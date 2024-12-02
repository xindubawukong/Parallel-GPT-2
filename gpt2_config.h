#ifndef GPT2_CONFIG_H_
#define GPT2_CONFIG_H_

#include <iostream>

#include "nlohmann/json.hpp"

struct GPT2Config {
  GPT2Config(nlohmann::json config) {
    std::cout << "GPT2 config:" << std::endl
              << std::setw(4) << config << std::endl;
    n_ctx = config["n_ctx"];
    n_embd = config["n_embd"];
    n_head = config["n_head"];
    n_layer = config["n_layer"];
    n_positions = config["n_positions"];
    vocab_size = config["vocab_size"];
    use_cache = config["use_cache"];
  }
  int n_ctx;
  int n_embd;
  int n_head;
  int n_layer;
  int n_positions;
  int vocab_size;
  bool use_cache;
};

#endif  // GPT2_CONFIG_H_
