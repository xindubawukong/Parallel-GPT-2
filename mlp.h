#ifndef MLP_H_
#define MLP_H_

#include <cmath>
#include <string>

#include "gpt2_config.h"
#include "linear.h"
#include "parlay/internal/get_time.h"
#include "parlay/parallel.h"

/*
def gelu(self, input: Tensor) -> Tensor:
    return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))


def mlp(x, c_fc, c_proj):  # [n_seq, n_embd] -> [n_seq, n_embd]
    # project up
    a = gelu(linear(x, **c_fc))  # [n_seq, n_embd] -> [n_seq, 4*n_embd]

    # project back down
    x = linear(a, **c_proj)  # [n_seq, 4*n_embd] -> [n_seq, n_embd]

    return x
*/

struct MLP {
  GPT2Config config;
  Linear c_fc;
  Linear c_proj;
  float* buf;

  MLP(GPT2Config config_)
      : config(config_),
        c_fc(config.n_embd, 4 * config.n_embd),
        c_proj(4 * config.n_embd, config.n_embd) {
    buf = new float[config.n_ctx * 4 * config.n_embd]();
  }

  float GeLU(float x) {
    static const float sqrt_2_pi = std::sqrt(2.0 / M_PI);
    static const float coef = 0.044715;
    return 0.5 * x *
           (1.0 + std::tanh(sqrt_2_pi * (x + coef * std::pow(x, 3.0))));
  }

  // go through the MLP and store the result back to x
  void operator()(float* x, int n_seq) {
    parlay::internal::timer t;
    c_fc(x, buf, n_seq);
    parlay::parallel_for(0, n_seq * 4 * config.n_embd,
                         [&](int i) { buf[i] = GeLU(buf[i]); });
    c_proj(buf, x, n_seq);
  }

  void LoadParameters(auto& params, std::string prefix) {
    c_fc.LoadParameters(params, prefix + ".c_fc");
    c_proj.LoadParameters(params, prefix + ".c_proj");
  }
};

#endif  // MLP_H_
