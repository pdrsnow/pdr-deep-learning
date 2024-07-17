#!/usr/bin/python
# -*- coding: UTF-8 -*-

import matplotlib as mpl
import matplotlib.pyplot as plt
# python 的可视化模块, 我有教程 (https://mofanpy.com/tutorials/data-manipulation/plt/)
import numpy
import torch
from torch.autograd import Variable

mpl.use("TkAgg")


# https://mofanpy.com/tutorials/machine-learning/torch/activation

def test_01_numpy_or_torch():
    # Torch 自称为神经网络界的 Numpy,
    # 因为他能将 torch 产生的 tensor 放在 GPU 中加速运算 (前提是你有合适的 GPU),
    # 就像 Numpy 会把 array 放在 CPU 中加速运算.
    # 所以神经网络的话, 当然是用 Torch 的 tensor 形式数据最好咯.
    # 就像 Tensorflow 当中的 tensor 一样.
    numpy_data = numpy.arange(6).reshape((2, 3))
    torch_data = torch.from_numpy(numpy_data)  # numpy => torch
    numpy_data2 = torch_data.numpy()  # torch => numpy
    print(
        '\nnumpy array:', numpy_data,
        '\ntorch tensor:', torch_data,
        '\ntensor to array:', numpy_data2,
    )


def test_o2_math():
    # abs 绝对值计算
    data = [-1, -2, 1, 2]
    tensor = torch.FloatTensor(data)  # 转换成32位浮点 tensor
    print(
        '\nabs',
        '\nnumpy: ', numpy.abs(data),  # [1 2 1 2]
        '\ntorch: ', torch.abs(tensor)  # [1 2 1 2]
    )

    # sin   三角函数 sin
    print(
        '\nsin',
        '\nnumpy: ', numpy.sin(data),  # [-0.84147098 -0.90929743  0.84147098  0.90929743]
        '\ntorch: ', torch.sin(tensor)  # [-0.8415 -0.9093  0.8415  0.9093]
    )

    # mean  均值
    print(
        '\nmean',
        '\nnumpy: ', numpy.mean(data),  # 0.0
        '\ntorch: ', torch.mean(tensor)  # 0.0
    )


def test_03_linear_algebra():
    # matrix multiplication 矩阵叉乘
    data = [[1, 2], [3, 4]]
    tensor = torch.FloatTensor(data)  # 转换成32位浮点 tensor
    # correct method
    print(
        '\nmatrix multiplication (matmul)',
        '\nnumpy: ', numpy.matmul(data, data),  # [[7, 10], [15, 22]]
        '\ntorch: ', torch.matmul(tensor, tensor)  # [[7, 10], [15, 22]]
    )

    # !!!!  下面是错误的方法 !!!!
    # matrix multiplication 矩阵点乘
    data = numpy.array(data)
    print(
        '\nmatrix multiplication (dot)',
        '\nnumpy: ', data.dot(data),
        '\ntorch: ', tensor.dot(tensor)  # torch 会转换成 [1,2,3,4].dot([1,2,3,4) = 30.0
    )


def test_04_variable():
    tensor = torch.FloatTensor([[1, 2, 3], [4, 5, 6]])
    variable = Variable(tensor, requires_grad=True)
    print(variable)


def test_05_functional():
    import torch.nn.functional as F
    x = torch.linspace(-5, 5, 200)  # x data (tensor), shape=(100, 1)
    x = Variable(x)

    x_np = x.data.numpy()  # 换成 numpy array, 出图时用
    # 几种常用的 激励函数
    y_relu = F.relu(x).data.numpy()
    y_sigmoid = F.sigmoid(x).data.numpy()
    y_tanh = F.tanh(x).data.numpy()
    y_softplus = F.softplus(x).data.numpy()
    # y_softmax = F.softmax(x)  softmax 比较特殊, 不能直接显示, 不过他是关于概率的, 用于分类

    # matplotlib.pyplot
    # 可视化模块教程 (https://mofanpy.com/tutorials/data-manipulation/plt/)
    plt.figure(1, figsize=(8, 6))
    plt.subplot(221)
    plt.plot(x_np, y_relu, c='red', label='relu')
    plt.ylim((-1, 5))
    plt.legend(loc='best')

    plt.subplot(222)
    plt.plot(x_np, y_sigmoid, c='red', label='sigmoid')
    plt.ylim((-0.2, 1.2))
    plt.legend(loc='best')

    plt.subplot(223)
    plt.plot(x_np, y_tanh, c='red', label='tanh')
    plt.ylim((-1.2, 1.2))
    plt.legend(loc='best')

    plt.subplot(224)
    plt.plot(x_np, y_softplus, c='red', label='softplus')
    plt.ylim((-0.2, 6))
    plt.legend(loc='best')

    # 需要支持交互显示环境
    plt.show()
    # 输出到图片
    # plt.savefig("./test_05_functional.png")


def test_06_unsqueeze():
    x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)  # x data (tensor), shape=(100, 1)
    y = x.pow(2) + 0.2 * torch.rand(x.size())  # noisy y data (tensor), shape=(100, 1)

    # 画图
    plt.scatter(x.data.numpy(), y.data.numpy())
    # plt.savefig("./test_06_unsqueeze.png")
    plt.show()


if __name__ == '__main__':
    print(torch.cuda.is_available())
    test_01_numpy_or_torch()
