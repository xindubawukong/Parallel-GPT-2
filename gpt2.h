#ifndef GPT2_H_
#define GPT2_H_

#include "gpt2_config.h"
#include "parlay/internal/get_time.h"
#include "parlay/parallel.h"
#include "transformer_block.h"
#include "utils.h"

/*
def gpt2(inputs, wte, wpe, blocks, ln_f, n_head):  # [n_seq] -> [n_seq, n_vocab]
    # token + positional embeddings
    x = wte[inputs] + wpe[range(len(inputs))]  # [n_seq] -> [n_seq, n_embd]

    # forward pass through n_layer transformer blocks
    for block in blocks:
        x = transformer_block(x, **block, n_head=n_head)  # [n_seq, n_embd] -> [n_seq, n_embd]

    # projection to vocab
    x = layer_norm(x, **ln_f)  # [n_seq, n_embd] -> [n_seq, n_embd]
    return x @ wte.T  # [n_seq, n_embd] -> [n_seq, n_vocab]
*/
struct GPT2 {
  GPT2Config config;
  float* wte;  // vocab_size * n_embd
  float* wpe;  // n_ctx * n_embd
  std::vector<TransformerBlock> blocks;
  LayerNorm ln_f;
  Linear lm_head;
  float* x;  // temporary variable
  std::vector<std::vector<float>> hidden_states;

  GPT2(GPT2Config config_)
      : config(config_),
        ln_f(config.n_embd),
        lm_head(config.n_embd, config.vocab_size) {
    wte = new float[config.vocab_size * config.n_embd]();
    wpe = new float[config.n_ctx * config.n_embd]();
    for (int i = 0; i < config.n_layer; i++) {
      blocks.emplace_back(config);
      blocks[i].layer_id = i;
    }
    x = new float[config.n_ctx * config.n_embd]();
  }

  void operator()(int64_t* inputs, int n_seq, float* outputs,
                  bool output_hidden_states = false) {
    std::cout << "\ninputs: ";
    for (int i = 0; i < n_seq; i++) std::cout << inputs[i] << " ";
    std::cout << std::endl;
    parlay::internal::timer t;
    parlay::parallel_for(0, n_seq, [&](int i) {
      parlay::parallel_for(0, config.n_embd, [&](int j) {
        x[i * config.n_embd + j] =
            wte[inputs[i] * config.n_embd + j] + wpe[i * config.n_embd + j];
      });
    });
    if (output_hidden_states) {
      hidden_states.clear();
      hidden_states.push_back(std::vector<float>(x, x + n_seq * config.n_embd));
    }
    t.next("token + positional embeddings");
    // go through the transformer blocks
    for (auto& block : blocks) {
      std::cout << "\ntransformer block" + std::to_string(block.layer_id) +
                       " start"
                << std::endl;
      block(x, n_seq);
      if (output_hidden_states) {
        hidden_states.push_back(
            std::vector<float>(x, x + n_seq * config.n_embd));
      }
      t.next("transformer block " + std::to_string(block.layer_id));
    }
    ln_f(x, n_seq);
    if (output_hidden_states) {
      hidden_states.pop_back();
      hidden_states.push_back(std::vector<float>(x, x + n_seq * config.n_embd));
    }
    std::cout << std::endl;
    t.next("layer_norm");
    lm_head(x, outputs, n_seq);
    t.next("lm_head");
    t.total();
  }

  void LoadParameters(auto& params) {
    std::cout << "\nLoading GPT2 parameters" << std::endl;
    Load2D(wte, params["model.transformer.wte.weight"], config.vocab_size,
           config.n_embd);
    Load2D(wpe, params["model.transformer.wpe.weight"], config.n_ctx,
           config.n_embd);
    std::cout << "wte and wpe loaded" << std::endl;
    for (auto& block : blocks) {
      block.LoadParameters(
          params, "model.transformer.h." + std::to_string(block.layer_id));
    }
    ln_f.LoadParameters(params, "model.transformer.ln_f");
    lm_head.weight = wte;
    parlay::parallel_for(0, config.vocab_size,
                         [&](int i) { lm_head.bias[i] = 0; });
    std::cout << "Successfully loaded GPT2 parameters!\n" << std::endl;
  }
};

#endif  // GPT2_H_
