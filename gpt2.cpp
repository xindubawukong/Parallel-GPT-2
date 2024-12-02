#include "gpt2.h"

#include <iostream>

#include "gflags/gflags.h"
#include "gpt2_config.h"
#include "nlohmann/json.hpp"
#include "torch/script.h"

using namespace std;

DEFINE_string(model_name, "gpt2", "Model name");
DEFINE_int32(benchmark, 0, "# of tokens for benchmark");
DEFINE_string(prompt, "Apple is located", "Prompt");
DEFINE_int32(n, 10, "Number of tokens to generate");

void CompareOutput(GPT2Config config, torch::jit::script::Module& gpt2_torch,
                   GPT2& gpt2) {
  // Example input
  torch::Tensor input_ids = torch::tensor({{12345}});
  std::vector<torch::jit::IValue> inputs;
  inputs.push_back(input_ids);

  // (logits, past_key_values, hidden_states)
  auto output = gpt2_torch.forward(inputs).toTuple();
  auto logits = output->elements()[0].toTensor();
  cout << "logits: " << logits.sizes() << endl;
  // past_key_values: cached k,v for each layer
  auto past_key_values = output->elements()[1].toTuple();
  cout << "past_key_values: " << past_key_values->size() << endl;
  auto layer0_kv = past_key_values->elements()[0].toTuple();
  // [1, 12, 20, 64] because of multi-head attention
  cout << "layer0_k: " << layer0_kv->elements()[0].toTensor().sizes() << endl;
  cout << "layer0_v: " << layer0_kv->elements()[1].toTensor().sizes() << endl;
  // hidden_states[0] is the x after wte and wpe.
  // hidden_states[1-12] are layer outputs
  auto hidden_states = output->elements()[2].toTuple();
  cout << "hidden_states: " << hidden_states->size() << endl;
  auto layer0_output = hidden_states->elements()[0].toTensor();
  cout << "layer0_output: " << layer0_output.sizes() << endl;

  // compare the output of the our implementation with the PyTorch model
  int64_t* input_ids_data = input_ids.data_ptr<int64_t>();
  float* outputs = new float[1 * config.vocab_size];
  gpt2(input_ids_data, 1, outputs, true);
  cout << endl;
  // for (int t = 0; t <= 12; t++) {
  //   cout << "hidden states " << t << ":" << endl;
  //   for (int i = 0; i < 20; i++) {
  //     cout << fixed << setprecision(6) << gpt2.hidden_states[t][i] << ' '
  //          << hidden_states->elements()[t].toTensor().data_ptr<float>()[i]
  //          << endl;
  //   }
  // }
  cout << "outputs:" << endl;
  for (int i = 0; i < 20; i++) {
    cout << fixed << setprecision(6) << outputs[i] << " "
         << logits.data_ptr<float>()[i] << endl;
  }
}

int main(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  string model_path = "../model/" + FLAGS_model_name + "_model.pt";
  string config_path = "../model/" + FLAGS_model_name + "_config.json";
  torch::jit::script::Module gpt2_torch = torch::jit::load(model_path);
  GPT2Config config = nlohmann::json::parse(ifstream(config_path));

  GPT2 gpt2(config);

  unordered_map<string, at::Tensor> params;
  cout << "parameters:" << endl;
  for (const auto& param : gpt2_torch.named_parameters()) {
    std::cout << param.name << ' ' << param.value.sizes() << std::endl;
    params[param.name] = param.value;
  }
  gpt2.LoadParameters(params);

  CompareOutput(config, gpt2_torch, gpt2);

  if (FLAGS_benchmark > 0) {
    int n_seq = FLAGS_benchmark;
    vector<int64_t> input(n_seq);
    float* output = new float[config.n_ctx * config.vocab_size];
    double sum = 0;
    int rounds = 5;
    for (int r = 0; r < rounds; r++) {
      for (int i = 0; i < n_seq; i++) input[i] = rand() % config.vocab_size;
      parlay::internal::timer t;
      gpt2(input.data(), n_seq, output);
      double duration = t.stop();
      cout << "\nround " << r << " time: " << duration << endl;
      if (r > 0) sum += duration;
    }
    cout << "\nours average time: " << sum / (rounds - 1) << endl;

    // sum = 0;
    // for (int r = 0; r < rounds; r++) {
    //   torch::Tensor input_ids =
    //       torch::randint(0, config.vocab_size, {1, n_seq});
    //   std::vector<torch::jit::IValue> inputs;
    //   inputs.push_back(input_ids);
    //   parlay::internal::timer t;
    //   gpt2_torch.forward(inputs);
    //   double duration = t.stop();
    //   cout << "\nround " << r << " time: " << duration << endl;
    //   if (r > 0) sum += duration;
    // }
    // cout << "\nlibtorch average time: " << sum / (rounds - 1) << endl;
  } else {
    cout << "\nRunning on prompt: " << FLAGS_prompt << endl << endl;
    auto input_ids = Encode(FLAGS_model_name, FLAGS_prompt);
    std::cout << std::endl;
    float* output = new float[config.n_ctx * config.vocab_size];
    for (int i = 0; i < FLAGS_n; i++) {
      int n_seq = input_ids.size();
      gpt2(input_ids.data(), n_seq, output);
      int best = 0;
      for (int j = 1; j < config.vocab_size; j++) {
        if (output[(n_seq - 1) * config.vocab_size + j] >
            output[(n_seq - 1) * config.vocab_size + best]) {
          best = j;
        }
      }
      input_ids.push_back(best);
      cout << "best: " << best << endl;
    }
    auto output_text = Decode(FLAGS_model_name, input_ids);
    cout << "\noutput: " << output_text << endl;
  }
  return 0;
}
