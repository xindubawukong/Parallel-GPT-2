#ifndef LINEAR_H_
#define LINEAR_H_

#include <string>

#include "utils.h"

/*
def linear(x, w, b):  # [m, in], [in, out], [out] -> [m, out]
    return x @ w + b
*/
struct Linear {
  int in, out;
  float* weight;  // out * in
  float* bias;

  Linear(int in_, int out_) : in(in_), out(out_) {
    weight = new float[in * out]();
    bias = new float[out]();
  }

  // input: [m, in]
  // output: [m, out]
  void operator()(float* input, float* output, int m) {
    static const int BLOCK = 8;
    parlay::parallel_for(0, m / BLOCK, [&](int i) {
      i *= BLOCK;
      parlay::parallel_for(0, out, [&](int j) {
        float sum[BLOCK];
        for (int t = 0; t < BLOCK; t++) sum[t] = bias[j];
        for (int k = 0; k < in; k++) {
          float b = weight[j * in + k];
          for (int t = 0; t < BLOCK; t++) {
            sum[t] += input[(i + t) * in + k] * b;
          }
        }
        for (int t = 0; t < BLOCK; t++) {
          output[(i + t) * out + j] = sum[t];
        }
      });
    });
    for (int i = m / BLOCK * BLOCK; i < m; i++) {
      parlay::parallel_for(0, out, [&](int j) {
        float sum = bias[j];
        for (int k = 0; k < in; k++) {
          sum += input[i * in + k] * weight[j * in + k];
        }
        output[i * out + j] = sum;
      });
    }
  }

  void LoadParameters(auto& params, std::string prefix) {
    Load2D(weight, params[prefix + ".weight"], in, out);
    Load1D(bias, params[prefix + ".bias"], out);

    // transpose the weight matrix
    float* tmp = new float[in * out];
    parlay::parallel_for(0, in, [&](int i) {
      parlay::parallel_for(
          0, out, [&](int j) { tmp[j * in + i] = weight[i * out + j]; });
    });
    parlay::parallel_for(0, in * out, [&](int i) { weight[i] = tmp[i]; });
    delete[] tmp;
  }
};

#endif  // LINEAR_H_
