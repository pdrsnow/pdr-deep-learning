<https://llama-cpp-python.readthedocs.io/en/latest/>

## `llama.cpp`

## `GGUF`和`GGML`

## 量化

```toml
[tool.poetry]
name = "llamaproject"
version = "0.1.0"
description = ""
authors = ["pdrsnow <229269179@qq.com>"]
readme = "README.md"

[tool.poetry.dependencies]
python = "^3.10"

llama-cpp-python = { version = "^0.3.6", source = "aliyun" }
torch = { version = "^2.5.1+cu118", source = "torch" }
sentencepiece = { version = "^0.2.0", source = "aliyun" }
transformers = { version = "^4.48.0", source = "aliyun" }
gguf = { version = "^0.14.0", source = "aliyun" }
protobuf = { version = "^5.29.3", source = "aliyun" }

[[tool.poetry.source]]
name = "aliyun"
url = "https://mirrors.aliyun.com/pypi/simple"
priority = "primary"

[[tool.poetry.source]]
name = "torch"
url = "https://download.pytorch.org/whl/cu118/"
priority = "supplemental"

[build-system]
requires = ["poetry-core"]
build-backend = "poetry.core.masonry.api"
```

```shell
poetry shell && poetry show

# 模型转换
git clone https://github.com/ggerganov/llama.cpp
python llama.cpp/convert_hf_to_gguf.py Qwen/Qwen2-VL-7B-Instruct --outfile Qwen/qwen2-vl-7b-instruct-f16.gguf
python llama.cpp/convert_hf_to_gguf.py Qwen/Qwen2-VL-7B-Instruct --outfile Qwen/qwen2-vl-7b-instruct-q8_0.gguf --outtype q8_0

# Q4_K_M
bin/llama-quantize Qwen/qwen2-vl-7b-instruct-f16.gguf qwen2-vl-7b-instruct-q4_k_m.gguf q4_k_m
```

## `llama.cpp`下载

> 下载地址: <https://github.com/ggml-org/llama.cpp/releases>

### 文件说明

```text
llama-<version>-bin-<os>-<feature>-<arch>.zip
```

> `llama.cpp`二进制文件，解压即可使用

- `<version>`：llama.cpp的版本。建议使用最新版本，有bug时，请尝试之前的版本直到找到能正常工作的为止。
- `<os>`：操作系统。`win`代表Windows；`macos`代表macOS；`linux`代表Linux。
- `<arch>`：系统架构。`x64`对应`x86_64`；`arm64`对应`arm64`。
- `<feature>`: 架构特性，适配的指令信息：

### `CPU`运行

+ `x86_64 CPU`建议首先尝试`avx2`。
    - `noavx`：完全无AVX硬件加速。
    - `avx2`，`avx`，`avx512`：基于SIMD的加速。大多数现代桌面CPU应该支持AVX2，部分CPU支持AVX512。
    - `openblas`：依赖OpenBLAS加速提示词(prompt)处理，但不涉及生成过程。

+ `arm64 CPU`建议首先尝试`llvm`。
    - `llvm`和`msvc`是不同的编译器

### `GPU`运行

+ `vulcan`：支持某些NVIDIA和AMD GPU
+ `kompute`：支持某些NVIDIA和AMD GPU；AMD GPU 先尝试
+ `sycl`：支持Intel GPU，包含oneAPI运行时；Intel GPU 先尝试`sycl`。
+ `cu<cuda_verison>`；NVIDIA GPU先尝试`
    - 当未包含`CUDA`运行时，解压`cudart-llama-bin-win-cu<cuda_version>-x64.zip`到`llama.cpp`的路径下

### `Linux`&`macOS`

+ `Linux`：
    - 仅有一个预构建的二进制文件`llama-<version>-bin-linux-x64.zip`，支持CPU。
+ `macOS`：
    - 对于Intel Mac，使用`llama-<version>-bin-macos-x64.zip`（不支持GPU）；
    - 对于Apple Silicon，使用`llama-<version>-bin-macos-arm64.zip`（支持GPU）。
