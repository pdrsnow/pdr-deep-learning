<https://transformers.run/c2/2021-12-17-transformers-note-4/>
<https://uqoo.cc/vllmhe-transformersde-xiang-xi-dui-bi/>

## 介绍

1. 功能定义：一种神经网络架构，通用NLP框架，支持训练、微调、推理
2. 支持模型：BERT, GPT, T5, BART, RoBERTa etc.
3. 底层框架：TensorFlow, PyTorch, Flax etc.

| 特性   | 	vLLM                | 	Transformers        |
|------|----------------------|----------------------|
| 主要用途 | 	高效推理、大模型推理          | 	通用 NLP 任务，训练、微调和推理  |
| 性能优化 | 	内存和并发优化，低延迟、高吞吐量    | 	依赖外部工具进行推理优化        |
| 模型支持 | 	主要支持 GPT 类生成模型      | 	支持多种模型架构和任务类型       |
| 易用性  | 	针对推理简化 API，适合生产环境   | 	丰富的社区支持，广泛的文档和教程    |
| 扩展性  | 	单机多 GPU 优化，有限的分布式支持 | 	强大的分布式支持，适合大规模训练和推理 |

