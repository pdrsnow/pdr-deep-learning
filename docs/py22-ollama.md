## 下载安装`Ollama`

> + 下载地址: <https://ollama.com/download/windows>
> + 参考文档：<https://www.cnblogs.com/obullxl/p/18295202/NTopic2024071001>
> + <https://ollama.readthedocs.io/quickstart/#quickstart>
> + <https://github.com/ollama/ollama/blob/main/docs/modelfile.md>
> + <https://github.com/ollama/ollama/blob/main/docs/import.md>

## 环境变量

```shell
# 服务地址
OLLAMA_HOST=127.0.0.1:11434

# 模型存放地址(默认: 用户目录/.ollama/models)
OLLAMA_MODELS=~/.ollama/models
```

## 使用`Ollama`进行LLM推理

### `Ollama`仓库模型

```shell
# 使用GLM-4-9b(没有会自动下载)
ollama run glm4:9b

# 使用千问2.5 7b(没有会自动下载)
ollama run qwen2.5:7b
```

### `huggingface`GGUF模型

```shell
ollama run hf.co/{username}/{reponame}:latest
```

### 本地模型

```shell
$HF_NAME='Qwen/Qwen2-VL-2B-Instruct'
$GF_NAME='Qwen/Qwen2-VL-2B-Instruct-F16.gguf'

# 模型转换
poetry shell && poetry show
python llama.cpp/convert_hf_to_gguf.py $HF_NAME --outfile $GF_NAME

echo "FROM $GF_NAME" > Modelfile

ollama create --quantize q4_K_M qwen2:vl_2b -f Modelfile
```

1) Q2_K
2) Q4_K_S
3) Q4_K_M
4) Q8_K_S
5) Q8_K_M
