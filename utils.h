#ifndef UTILS_H_
#define UTILS_H_

#include <string>

#include "parlay/parallel.h"
#include "torch/script.h"

template <typename T>
void Load1D(T* dst, const torch::Tensor& tensor, int n) {
  const T* src = tensor.data_ptr<T>();
  std::copy(src, src + n, dst);
}

template <typename T>
void Load2D(T* dst, const torch::Tensor& tensor, int n_row, int n_col) {
  if ((int)tensor.size(0) == n_row && (int)tensor.size(1) == n_col) {
    Load1D(dst, tensor, n_row * n_col);
  } else if ((int)tensor.size(0) == n_col && (int)tensor.size(1) == n_row) {
    Load1D(dst, tensor.t(), n_row * n_col);
  } else {
    assert(0);
  }
}

std::string ExecCommand(std::string cmd) {
  cmd += " 2>/dev/null";
  std::array<char, 128> buffer;
  std::string result;
  // Open a pipe to the command
  std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd.c_str(), "r"),
                                                pclose);
  if (!pipe) {
    throw std::runtime_error("popen() failed!");
  }
  // Read the output of the command
  while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
    result += buffer.data();
  }

  return result;
}

std::vector<int64_t> Encode(std::string model_name, std::string text) {
  std::string cmd = "cd .. && python3 gpt2.py --model_name " + model_name +
                    " --encode \"" + text + "\"";
  std::string output = ExecCommand(cmd);
  std::vector<int64_t> ids;
  auto IsDigit = [](char c) { return c >= '0' && c <= '9'; };
  for (int l = 0, r = 0; l < (int)output.length(); l = r + 1) {
    while (l < (int)output.length() && !IsDigit(output[l])) l++;
    if (l == (int)output.length()) break;
    r = l;
    while (r + 1 < (int)output.length() && IsDigit(output[r + 1])) r++;
    int id = 0;
    for (int i = l; i <= r; i++) id = id * 10 + output[i] - '0';
    ids.push_back(id);
  }
  return ids;
}

std::string Decode(std::string model_name, std::vector<int64_t> ids) {
  std::string cmd =
      "cd .. && python3 gpt2.py --model_name " + model_name + " --decode \"";
  for (int i = 0; i < (int)ids.size(); i++) {
    if (i > 0) cmd += " ";
    cmd += std::to_string(ids[i]);
  }
  cmd += "\"";
  return ExecCommand(cmd.c_str());
}

#endif  // UTILS_H_
