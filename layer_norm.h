#ifndef LAYER_NORM_H_
#define LAYER_NORM_H_

#include <string>

#include "utils.h"

/*
def layer_norm(x, g, b, eps: float = 1e-5):
    mean = np.mean(x, axis=-1, keepdims=True)
    variance = np.var(x, axis=-1, keepdims=True)
    x = (x - mean) / np.sqrt(variance + eps)  # normalize x to have mean=0 and var=1 over last axis
    return g * x + b  # scale and offset with gamma/beta params
*/
struct LayerNorm {
  int n_embd;
  float* weight;
  float* bias;

  LayerNorm(int n_embd_) : n_embd(n_embd_) {
    weight = new float[n_embd]();
    bias = new float[n_embd]();
  }

  // go through the layer norm and store the result back to x
  // x is [n_seq, n_embd]
  void operator()(float* x, int n_seq) {
    parlay::parallel_for(0, n_seq, [&](int i) {
      float mean = 0;
      float variance = 0;
      float* a = x + i * n_embd;
      for (int j = 0; j < n_embd; j++) {
        mean += a[j];
        variance += a[j] * a[j];
      }
      mean /= n_embd;
      variance = variance / n_embd - mean * mean;
      float inv_std = 1.0 / sqrt(variance + 1e-5);
      for (int j = 0; j < n_embd; j++) {
        a[j] = (a[j] - mean) * inv_std;
      }
      for (int j = 0; j < n_embd; j++) {
        a[j] = weight[j] * a[j] + bias[j];
      }
    });
  }

  void LoadParameters(auto& params, std::string prefix) {
    Load1D(weight, params[prefix + ".weight"], n_embd);
    Load1D(bias, params[prefix + ".bias"], n_embd);
  }
};

#endif  // LAYER_NORM_H_
