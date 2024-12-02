#ifndef MULTI_HEAD_ATTENTION_H_
#define MULTI_HEAD_ATTENTION_H_

#include <string>

#include "gpt2_config.h"
#include "linear.h"
#include "parlay/internal/get_time.h"
#include "utils.h"

/*
def mha(x, c_attn, c_proj, n_head):  # [n_seq, n_embd] -> [n_seq, n_embd]
    # qkv projection
    x = linear(x, **c_attn)  # [n_seq, n_embd] -> [n_seq, 3*n_embd]

    # split into qkv
    qkv = np.split(x, 3, axis=-1)  # [n_seq, 3*n_embd] -> [3, n_seq, n_embd]

    # split into heads
    qkv_heads = list(map(lambda x: np.split(x, n_head, axis=-1), qkv))  # [3, n_seq, n_embd] -> [3, n_head, n_seq, n_embd/n_head]

    # causal mask to hide future inputs from being attended to
    causal_mask = (1 - np.tri(x.shape[0], dtype=x.dtype)) * -1e10  # [n_seq, n_seq]

    # perform attention over each head
    out_heads = [attention(q, k, v, causal_mask) for q, k, v in zip(*qkv_heads)]  # [3, n_head, n_seq, n_embd/n_head] -> [n_head, n_seq, n_embd/n_head]

    # merge heads
    x = np.hstack(out_heads)  # [n_head, n_seq, n_embd/n_head] -> [n_seq, n_embd]

    # out projection
    x = linear(x, **c_proj)  # [n_seq, n_embd] -> [n_seq, n_embd]

    return x
*/

struct Attention {
  GPT2Config config;
  int head_dim;
  float* q;
  float* k;
  float* v;
  float* y;
  float* scores;
  float* softmax_scores;

  Attention(GPT2Config config_) : config(config_) {
    assert(config.n_embd % config.n_head == 0);
    head_dim = config.n_embd / config.n_head;
    q = new float[config.n_ctx * head_dim];
    k = new float[config.n_ctx * head_dim];
    v = new float[config.n_ctx * head_dim];
    y = new float[config.n_ctx * head_dim];
    scores = new float[config.n_ctx * config.n_ctx];
    softmax_scores = new float[config.n_ctx * config.n_ctx];
  }

  void operator()(int n_seq) {
    parlay::internal::timer t;
    parlay::parallel_for(0, n_seq, [&](int i) {
      parlay::parallel_for(0, n_seq, [&](int j) {
        if (j > i) {
          scores[i * n_seq + j] = -1e10f;
          return;
        }
        float score = 0.0f;
        for (int t = 0; t < head_dim; t++) {
          score += q[i * head_dim + t] * k[j * head_dim + t];
        }
        scores[i * n_seq + j] = score / std::sqrt(static_cast<float>(head_dim));
      });
    });
    // t.next("loop1");

    parlay::parallel_for(0, n_seq, [&](int i) {
      float max_score = -1e10;
      for (int j = 0; j < n_seq; j++) {
        max_score = std::max(max_score, scores[i * n_seq + j]);
      }
      float sum_exp = 0.0;
      for (int j = 0; j < n_seq; j++) {
        softmax_scores[i * n_seq + j] =
            std::exp(scores[i * n_seq + j] - max_score);
        sum_exp += softmax_scores[i * n_seq + j];
      }
      for (int j = 0; j < n_seq; j++) {
        softmax_scores[i * n_seq + j] /= sum_exp;
      }
    });
    // t.next("loop2");

    // transpose v
    parlay::parallel_for(0, n_seq, [&](int i) {
      parlay::parallel_for(
          0, head_dim, [&](int j) { k[j * n_seq + i] = v[i * head_dim + j]; });
    });
    std::swap(k, v);
    // t.next("transpose");

    parlay::parallel_for(0, n_seq, [&](int i) {
      parlay::parallel_for(0, head_dim, [&](int j) {
        float sum = 0;
        for (int t = 0; t < n_seq; t++) {
          sum += softmax_scores[i * n_seq + t] * v[j * n_seq + t];
        }
        y[i * head_dim + j] = sum;
      });
    });
    // t.next("loop3");
  }
};

struct MultiHeadAttention {
  GPT2Config config;
  Linear c_attn;
  Linear c_proj;
  std::vector<Attention> attentions;
  float* QKV;
  float* Q;
  float* K;
  float* V;
  float* y;

  MultiHeadAttention(GPT2Config config_)
      : config(config_),
        c_attn(config_.n_embd, 3 * config_.n_embd),
        c_proj(config_.n_embd, config_.n_embd) {
    int n_ctx = config.n_ctx;
    int n_embd = config.n_embd;
    int n_head = config.n_head;

    assert(n_embd % n_head == 0);

    QKV = new float[n_ctx * 3 * n_embd];
    Q = new float[n_ctx * n_embd];
    K = new float[n_ctx * n_embd];
    V = new float[n_ctx * n_embd];
    y = new float[n_ctx * n_embd];

    for (int i = 0; i < n_head; i++) {
      attentions.push_back(Attention(config));
    }
  }

  void operator()(float* x, int n_seq) {
    parlay::internal::timer t;
    int n_embd = config.n_embd;
    int n_head = config.n_head;
    int head_dim = n_embd / n_head;

    c_attn(x, QKV, n_seq);
    // t.next("c_attn");

    parlay::parallel_for(0, n_seq, [&](int i) {
      for (int j = 0; j < n_embd; j++)
        Q[i * n_embd + j] = QKV[i * 3 * n_embd + j];
      for (int j = 0; j < n_embd; j++)
        K[i * n_embd + j] = QKV[i * 3 * n_embd + n_embd + j];
      for (int j = 0; j < n_embd; j++)
        V[i * n_embd + j] = QKV[i * 3 * n_embd + 2 * n_embd + j];
    });
    // t.next("copy1");

    parlay::parallel_for(0, n_head, [&](int i) {
      auto& attention = attentions[i];
      parlay::parallel_for(0, n_seq, [&](int j) {
        for (int k = 0; k < head_dim; k++) {
          attention.q[j * head_dim + k] = Q[j * n_embd + i * head_dim + k];
          attention.k[j * head_dim + k] = K[j * n_embd + i * head_dim + k];
          attention.v[j * head_dim + k] = V[j * n_embd + i * head_dim + k];
        }
      });
    });
    // t.next("copy2");

    parlay::parallel_for(0, n_head, [&](int i) {
      auto& attention = attentions[i];
      attention(n_seq);
    });
    // t.next("heads");

    parlay::parallel_for(0, n_head, [&](int i) {
      auto& attention = attentions[i];
      parlay::parallel_for(0, n_seq, [&](int j) {
        for (int k = 0; k < head_dim; k++) {
          y[j * n_embd + i * head_dim + k] = attention.y[j * head_dim + k];
        }
      });
    });
    // t.next("copy3");

    c_proj(y, x, n_seq);
    // t.next("c_proj");
  }

  void LoadParameters(auto& params, std::string prefix) {
    c_attn.LoadParameters(params, prefix + ".c_attn");
    c_proj.LoadParameters(params, prefix + ".c_proj");
  }
};

#endif  // MULTI_HEAD_ATTENTION_H_
