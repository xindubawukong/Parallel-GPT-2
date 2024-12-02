#ifndef TRANSFORMER_BLOCK_H_
#define TRANSFORMER_BLOCK_H_

#include <string>

#include "gpt2_config.h"
#include "layer_norm.h"
#include "mlp.h"
#include "multi_head_attention.h"
#include "parlay/internal/get_time.h"
#include "utils.h"

/*
def transformer_block(x, mlp, attn, ln_1, ln_2, n_head):  # [n_seq, n_embd] -> [n_seq, n_embd]
    # multi-head causal self attention
    x = x + mha(layer_norm(x, **ln_1), **attn, n_head=n_head)  # [n_seq, n_embd] -> [n_seq, n_embd]

    # position-wise feed forward network
    x = x + ffn(layer_norm(x, **ln_2), **mlp)  # [n_seq, n_embd] -> [n_seq, n_embd]

    return x
*/
struct TransformerBlock {
  GPT2Config config;
  int layer_id;
  LayerNorm ln_1;
  MultiHeadAttention attn;
  LayerNorm ln_2;
  MLP mlp;
  float* x1;

  TransformerBlock(GPT2Config config_)
      : config(config_),
        layer_id(-1),
        ln_1(config.n_embd),
        attn(config_),
        ln_2(config.n_embd),
        mlp(config_) {
    x1 = new float[config.n_ctx * config.n_embd]();
  }

  // go through the transformer block and store the result back to x
  void operator()(float* x, int n_seq) {
    parlay::internal::timer t;
    std::copy(x, x + n_seq * config.n_embd, x1);
    ln_1(x1, n_seq);
    t.next("ln_1");
    attn(x1, n_seq);
    t.next("attn");
    for (int i = 0; i < n_seq * config.n_embd; i++) {
      x[i] += x1[i];
    }
    std::copy(x, x + n_seq * config.n_embd, x1);
    ln_2(x1, n_seq);
    t.next("ln_2");
    mlp(x1, n_seq);
    t.next("mlp");
    for (int i = 0; i < n_seq * config.n_embd; i++) {
      x[i] += x1[i];
    }
  }

  void LoadParameters(auto& params, std::string prefix) {
    std::cout << "Loading parameters from " << prefix << std::endl;
    ln_1.LoadParameters(params, prefix + ".ln_1");
    attn.LoadParameters(params, prefix + ".attn");
    ln_2.LoadParameters(params, prefix + ".ln_2");
    mlp.LoadParameters(params, prefix + ".mlp");
  }
};

#endif  // TRANSFORMER_BLOCK_H_
