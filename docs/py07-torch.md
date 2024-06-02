# `PyTorch`学习

## 1. 官网集合

> + PyTorch官网：<https://pytorch.org/get-started/locally/>
> + PyTorch中文：<https://pytorch.apachecn.org/2.0/tutorials/recipes/recipes_index/>
> + PyTorch中文：<https://pytorch-cn.readthedocs.io/zh/latest/>

> + 入门课程：<https://www.bilibili.com/video/BV1iv41117Zg>

> + <https://mofanpy.com/tutorials/machine-learning/torch/>
> + <https://github.com/jdb78/pytorch-forecasting>

## 2.安装`PyTorch`

```shell
# 3.8 <= python
python -m pip install torch -f https://download.pytorch.org/whl/torch_stable.html
# python -m pip install -r requirements.txt
```

**验证`PyTorch`**

```python
# 验证 PyTorch
import torch

x = torch.rand(5, 3)
print(x)

torch.cuda.is_available()
```

## 3.时间预测`pytorch-forecasting`

> + 项目地址：<https://github.com/jdb78/pytorch-forecasting>

> 路径[examples](examples)给出三个数据样例，建议先运行起来以对整个项目有一个整体的印象后，\
> 在根据项目文档进行下一步学习；

```shell
# 3.8 <= python
python -m pip install torch -f https://download.pytorch.org/whl/torch_stable.html
# 3.8 <= python <3.11
python -m pip install pytorch-forecasting
# 项目运行中会提示找不到`tensorboard`或`tensorboardX`模块，这里提前安装
python -m pip install tensorboard
python -m pip install optuna-integration
```

> + 构建模型：<https://pytorch-forecasting.readthedocs.io/en/stable/getting-started.html>

```shell
# 运行案例
python examples/nbeats.py
```

> 案例运行顺利会有新的文件生成，类似`lightning_logs/version_*/checkpoints/*.ckpt`；\
> `*.ckpt`就是学习过程中生成的模型数据，后续会运用该模型进行时间维度数据预测。
> + 应用模型：<https://pytorch-forecasting.readthedocs.io/en/stable/tutorials.html#id1>
