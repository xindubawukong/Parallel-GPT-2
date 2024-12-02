# Parallel GPT-2 Inference on CPUs

A pure C++ implemented gpt-2 inference, running on multi-CPUs in parallel.

Faster than <a href="https://github.com/karpathy/llm.c">llm.c</a> on 64 cores!

Authors:
- <a href="https://github.com/xindubawukong">Xiangyun Ding</a>
- <a href="https://github.com/sdeng006">Shuhuai Deng</a>
- <a href="https://github.com/ez022">Elisa Zhang</a>

## Requirements

This project requires the following software.

- python3
- cmake
- libtorch

### Download the Repositories

```
git clone --recurse-submodules git@github.com:xindubawukong/cs-228-project.git
```

### Download libtorch

- For linux:
https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-2.5.1%2Bcpu.zip
- For mac: https://download.pytorch.org/libtorch/cpu/libtorch-macos-arm64-2.5.1.zip

Unzip the file and save `libtorch` folder under `third_party/`.

### Install libraries

```
pip install -r requirements.txt 
```

## Usage

### Save GPT2 Model

```
python3 gpt2.py --save_model --model_name=gpt2
```

This will save parameters to `model/gpt2_model.pt` and config to `model/gpt2_config.json`.

Supported models: `gpt2|gpt2-medium|gpt2-large|gpt2-xl`.

### Compile

```
mkdir -p build && cd build
cmake ..
make
```

### Execute

```
./gpt2 --model_name=gpt2
```

It will read model parameters from the saved files.

```
./gpt2 --prompt "any prompt" --model_name=gpt2
```

You can change the prompt to generate other text.

You can also run a benchmark, which gives a input and run several rounds to test the performance:
```
./gpt2 --benchmark 1024 --modelname=gpt2
```

## References

- https://jaykmody.com/blog/gpt-from-scratch
- https://github.com/karpathy/llm.c
