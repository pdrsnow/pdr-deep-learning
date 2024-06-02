import os as os_

os_.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os_.environ["KERAS_BACKEND"] = "torch"
# os_.environ["KERAS_BACKEND"] = "tensorflow"

import matplotlib.pyplot as plt
import numpy as np


# https://www.bilibili.com/video/BV1Ct411H7rm

def demo_01_sequential():
    """
    线性回归
    """
    from keras import layers, models
    # 生成100个随机点
    x_data = np.random.rand(100)
    noise = np.random.normal(0, 0.01, x_data.shape)
    # y_data = x_data * 0.1 + 0.2
    y_data = x_data * 0.1 + 0.2 + noise

    # 显示数据
    plt.scatter(x_data, y_data)
    plt.show()

    # 构建顺序模型
    model = models.Sequential()
    # 添加 Dense 全连接层
    model.add(layers.Dense(units=1, input_dim=1))
    # 编译模型
    # sgd(Stochastic Gradient Descent): 随机梯度下降法
    # mse(Mean Squared Error): 均方误差
    model.compile(optimizer='sgd', loss="mse")

    # 训练模型(迭代模型)
    for step in range(1001):
        # 每次训练一个批次
        cost = model.train_on_batch(x_data, y_data)
        # 每500个batch打印一次cost值
        if (step % 100) == 0:
            print(f'step: {step}, cost: {cost}')

    # 打印权值和偏置值
    W, b = model.layers[0].get_weights()
    print(f'W: {W}, b: {b}')

    # x_data 输入网络中，得到预测值y_pred
    y_pred = model.predict(x_data)

    # 显示随机点
    plt.scatter(x_data, y_data)
    # 显示预测结果
    plt.plot(x_data, y_pred, 'r-', lw=3)
    plt.show()


def demo_02_square():
    """
    非线性回归
    """
    from keras import layers, models, optimizers, activations
    x_data = np.linspace(-0.5, 0.5, 200)
    noise = np.random.normal(0, 0.02, x_data.shape)
    y_data = np.square(x_data) + noise

    plt.scatter(x_data, y_data)
    plt.show()

    # 构建顺序模型
    model = models.Sequential()
    # 添加 Dense 全连接层
    # input_dim 输入, units 输出, activation 默认线性激活函数
    # model.add(layers.Dense(units=10, input_dim=1, activation=activations.tanh))
    model.add(layers.Dense(units=10, input_dim=1))
    model.add(layers.Activation('tanh'))
    # 添加第二 Dense 全连接层，input_dim可以不设置，默认使用上一层的units
    model.add(layers.Dense(units=1, activation=activations.tanh))
    # 也可以单独设置激活函数
    model.add(layers.Activation('tanh'))

    # 自定义优化器
    # 这里增大学习率lr有默认0.01，调整为0.3
    sgd = optimizers.SGD(learning_rate=0.3)

    # 编译模型, optimizer优化器
    # sgd(Stochastic Gradient Descent): 随机梯度下降法
    # mse(Mean Squared Error): 均方误差
    model.compile(optimizer=sgd, loss="mse")

    # 训练模型(迭代模型)
    for step in range(1001):
        # 每次训练一个批次
        cost = model.train_on_batch(x_data, y_data)
        # 每500个batch打印一次cost值
        if (step % 50) == 0:
            print(f'step: {step}, cost: {cost}')

    # 打印权值和偏置值
    W, b = model.layers[0].get_weights()
    print(f'W: {W}, b: {b}')

    x_pred = np.random.rand(150)
    # x_data 输入网络中，得到预测值y_pred
    y_pred = model.predict(x_pred)

    # 显示随机点
    plt.scatter(x_data, y_data)
    # 显示预测结果
    plt.plot(x_pred, y_pred, 'r-', lw=3)
    plt.show()


def demo_03_mnist():
    """
    MNIST数据集
    """
    from keras import datasets, utils, models, layers, optimizers
    # mnist数据集 生成虚拟数据(60000, 28, 28)
    (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
    print(f"x_train shape:{x_train.shape}, y_train shape:{y_train.shape}")
    # 显示数据集
    # plt.matshow(x_train[0], cmap='Grays')
    # plt.matshow(x_test[0], cmap='Oranges')
    # plt.show()

    # Scale images to the [0, 1] range
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255
    # Make sure images have shape (28, 28, 1)
    # x_train = np.expand_dims(x_train, -1)
    # x_test = np.expand_dims(x_test, -1)
    # plt.matshow(x_train.shape, cmap='Grays')
    # plt.matshow(x_test.shape, cmap='Oranges')
    # plt.show()

    # (60000, 28, 28) -> (60000, 784)
    x_train = x_train.reshape(x_train.shape[0], -1)
    x_test = x_test.reshape(x_test.shape[0], -1)
    print(f"x_train shape:{x_train.shape}, y_train shape:{y_train.shape}")
    # plt.matshow(x_train, cmap='Grays')
    # plt.matshow(x_test, cmap='Oranges')
    # plt.show()

    # 换one host格式
    y_train = utils.to_categorical(y_train, num_classes=10)
    y_test = utils.to_categorical(y_test, num_classes=10)
    # plt.matshow(y_train, cmap='Grays')
    # plt.matshow(y_test, cmap='Oranges')
    # plt.show()

    # 创建模型 输入784, 输出10
    model = models.Sequential([
        layers.Dense(units=10, input_dim=784, bias_initializer='one', activation='softmax')
    ])

    # 定义优化器(指定学习率)
    sgd = optimizers.SGD(learning_rate=0.2)
    # 使用自定义优化器编译模型, metrics
    model.compile(optimizer=sgd, loss='mse', metrics=['accuracy'])

    # 训练(填充)模型, batch_size:批次大小, epochs:迭代周期(每个批次循环次数)
    model.fit(x_train, y_train, batch_size=30, epochs=10)

    # 评估模型
    loss, accuracy = model.evaluate(x_test, y_test)
    print(f'test loss: {loss}, accuracy: {accuracy}')

    # 模型预测
    # y_perd = model.predict(x_test)
    # print("y_perd shape:", y_perd.shape)
    # plt.matshow(y_perd, cmap='Grays')
    # plt.show()


def demo_04_cross_entropy():
    """
    交叉熵
    """
    from keras import datasets, utils, models, layers, optimizers
    # mnist数据集 生成虚拟数据(60000, 28, 28)
    (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
    print(f"x_train shape:{x_train.shape}, y_train shape:{y_train.shape}")

    # (60000, 28, 28) -> (60000, 784)
    x_train = x_train.reshape(x_train.shape[0], -1) / 255
    x_test = x_test.reshape(x_test.shape[0], -1) / 255
    print(f"x_train shape:{x_train.shape}, y_train shape:{y_train.shape}")
    y_train = utils.to_categorical(y_train, num_classes=10)
    y_test = utils.to_categorical(y_test, num_classes=10)

    model = models.Sequential([
        layers.Dense(units=10, input_dim=784, bias_initializer='one', activation='softmax')
    ])

    sgd = optimizers.SGD(learning_rate=0.2)
    # 交叉商： categorical_crossentropy
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(x_train, y_train, batch_size=30, epochs=10)
    loss, accuracy = model.evaluate(x_test, y_test)
    print(f'test loss: {loss}, accuracy: {accuracy}')


def demo_05_over_fitting():
    """
    欠拟合/正确拟合/过拟合
    欠拟合(underfitting): 说明需要调整学习率, 学习次数等参数
    过拟合(overfitting): 干扰数据(如流量超过网卡极限)也纳入到模型中了, 可以增大训练数据集，迭代次数

    Early Stopping: 连续10个Epoch没有超过目前为止记录的最好`validation accuracy`，
    停止迭代，可以认为达到最佳，防止过拟合
    Dropout: 训练的时候随机关闭某些神经元，并`validation accuracy`(开启全部神经元)
    正则化项：
    """
    pass


def demo_06_dropout():
    """
    Dropout
    """
    from keras import datasets, utils, models, layers, optimizers
    # mnist数据集 生成虚拟数据(60000, 28, 28)
    (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
    print(f"x_train shape:{x_train.shape}, y_train shape:{y_train.shape}")

    # (60000, 28, 28) -> (60000, 784)
    x_train = x_train.reshape(x_train.shape[0], -1) / 255
    x_test = x_test.reshape(x_test.shape[0], -1) / 255
    print(f"x_train shape:{x_train.shape}, y_train shape:{y_train.shape}")
    y_train = utils.to_categorical(y_train, num_classes=10)
    y_test = utils.to_categorical(y_test, num_classes=10)

    model = models.Sequential([
        layers.Input(shape=(784,)),
        layers.Dense(units=200, bias_initializer='one', activation='tanh'),
        layers.Dropout(0.4),  # 40% 的神经元不工作(根据场景选择是否使用, 有些场景不适合)
        layers.Dense(units=100, bias_initializer='one', activation='tanh'),  # 默认使用上一层的units
        layers.Dropout(0.4),  # 40% 的神经元不工作(根据场景选择是否使用, 有些场景不适合)
        layers.Dense(units=10, bias_initializer='one', activation='softmax'),  # 默认使用上一层的units
    ])

    sgd = optimizers.SGD(learning_rate=0.2)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(x_train, y_train, batch_size=32, epochs=10)

    loss, accuracy = model.evaluate(x_test, y_test)
    print(f'test loss: {loss}, accuracy: {accuracy}')
    loss, accuracy = model.evaluate(x_train, y_train)
    print(f'train loss: {loss}, accuracy: {accuracy}')
    pass


def demo_07_regularizers():
    """
    正则化应用
    """
    from keras import datasets, utils, models, layers, optimizers, regularizers
    # mnist数据集 生成虚拟数据(60000, 28, 28)
    (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
    print(f"x_train shape:{x_train.shape}, y_train shape:{y_train.shape}")

    # (60000, 28, 28) -> (60000, 784)
    x_train = x_train.reshape(x_train.shape[0], -1) / 255
    x_test = x_test.reshape(x_test.shape[0], -1) / 255
    print(f"x_train shape:{x_train.shape}, y_train shape:{y_train.shape}")
    y_train = utils.to_categorical(y_train, num_classes=10)
    y_test = utils.to_categorical(y_test, num_classes=10)

    model = models.Sequential([
        layers.Input(shape=(784,)),
        layers.Dense(units=200, bias_initializer='one', activation='tanh',
                     kernel_regularizer=regularizers.L1(0.0003)),  # 正则化项
        layers.Dense(units=100, bias_initializer='one', activation='tanh'),  # 默认使用上一层的units
        layers.Dense(units=10, bias_initializer='one', activation='softmax'),  # 默认使用上一层的units
    ])

    sgd = optimizers.SGD(learning_rate=0.2)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(x_train, y_train, batch_size=32, epochs=10)

    loss, accuracy = model.evaluate(x_test, y_test)
    print(f'test loss: {loss}, accuracy: {accuracy}')
    loss, accuracy = model.evaluate(x_train, y_train)
    print(f'train loss: {loss}, accuracy: {accuracy}')
    pass


def demo_08_optimizers():
    """
    optimizers(优化器)：
    1. SGD(Stochastic Gradient Descent): 梯度下降法
    1.1. 标准-梯度下降法: 计算所有样本汇总误差, 根据总误差更新权值
    1.2. 随机-梯度下降法: 随机抽取一个样本计算误差，然后更新权值
    1.3. 批量-梯度下降法: 折中方案，总样本选一个批次(随机抽取若干样本作为一个批次)，计算这个批次的总误差，根据这个总误差更新权值
    2. Momentum: 梯度累加(类比运动中的加速度对速率的影响)
    3. NAG(Nesterov Accelerated Gradient): 预测梯度累加
    4. Adagrad: 平方梯度(梯度的平方值，学习率会越来越低，不宜叠加太多层)
    5. RMSprop: Adagrad的改进, 引入γ=0.9, 可自动调整学习率
    6. Adadelta: Adagrad的改进, 引入Δ, 可不设置学习率
    7. Adam: β1=0.9，β2=0.99, 根据梯度衰减优化学习率

    个人认为可以分为三类(SGD, Momentum, NAG), (Adagrad, RMSprop, Adadelta), (Adam)
    """
    from keras import datasets, utils, models, layers, optimizers, regularizers
    # mnist数据集 生成虚拟数据(60000, 28, 28)
    (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
    print(f"x_train shape:{x_train.shape}, y_train shape:{y_train.shape}")

    # (60000, 28, 28) -> (60000, 784)
    x_train = x_train.reshape(x_train.shape[0], -1) / 255
    x_test = x_test.reshape(x_test.shape[0], -1) / 255
    print(f"x_train shape:{x_train.shape}, y_train shape:{y_train.shape}")
    y_train = utils.to_categorical(y_train, num_classes=10)
    y_test = utils.to_categorical(y_test, num_classes=10)

    model = models.Sequential([
        layers.Input(shape=(784,)),
        layers.Dense(units=200, bias_initializer='one', activation='tanh',
                     kernel_regularizer=regularizers.L1(0.0003)),
        layers.Dense(units=100, bias_initializer='one', activation='tanh'),
        layers.Dense(units=10, bias_initializer='one', activation='softmax'),
    ])

    # sgd = optimizers.SGD(learning_rate=0.2)
    adam = optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(x_train, y_train, batch_size=32, epochs=10)

    loss, accuracy = model.evaluate(x_test, y_test)
    print(f'test loss: {loss}, accuracy: {accuracy}')
    loss, accuracy = model.evaluate(x_train, y_train)
    print(f'train loss: {loss}, accuracy: {accuracy}')
    pass


def demo_09_cnn():
    """
    BP神经网络：输入与输出神经元间全连接，每个连接一个权值；神经元过多时，计算的权值指数增长，计算很过大；
    CNN(Convolution Neural Network)；卷积神经网络思路，局部感受野+权值共享；[个人感觉像是一个树，枝叶汇聚到枝干，枝干在汇聚到主干,
    这个过程中记录下枝叶数(权值), 本质是对数据的抽象化]
    具体过程：(卷积->激活->池化)利用"卷积核/滤波器"进行"卷积计算"得到"特征图(feature map)"[局部感受野], 然后对”特征图“进行"池化"
    卷积(Convolution)："卷积核" + "卷积步长" + “卷积Padding”，生成的 特征图 也不一样
    池化(Pooling): 减少卷积层提取的特征个数，样本数据抽象化；
        最大池化(max-pooling)：提取区块内的最大值
        平均池化(mean-pooling): 提取区块内的平均值
        随机池化(stochastic-pooling): 提取区块内的平均值
    卷积Padding：卷积边界，SAME-PADDING 平面外补0，完整采样；VALID-PADDING 不会超出平面，但采样不全
    池化Padding：SAME-PADDING/VALID-PADDING

    LeNet-5
    :return:
    """
    from keras import datasets, utils, models, layers, optimizers
    # mnist数据集 生成虚拟数据(60000, 28, 28)
    (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
    print(f"x_train shape:{x_train.shape}, y_train shape:{y_train.shape}")

    # (60000, 28, 28) -> (60000, 784)
    x_train = x_train.reshape(-1, 28, 28, 1) / 255
    x_test = x_test.reshape(-1, 28, 28, 1) / 255
    print(f"x_train shape:{x_train.shape}, y_train shape:{y_train.shape}")
    y_train = utils.to_categorical(y_train, num_classes=10)
    y_test = utils.to_categorical(y_test, num_classes=10)

    model = models.Sequential([
        layers.Input(shape=(28, 28, 1)),
        # filters:卷积核数量, kernel_size:卷积窗口大小, strides:步长, padding:边界, activation:激活
        layers.Conv2D(filters=32, kernel_size=5, strides=1, padding='same', activation='relu'),
        # 池化层
        layers.MaxPool2D(pool_size=2, strides=2, padding='same'),
        # 第二层卷积与池化
        layers.Conv2D(filters=64, kernel_size=5, strides=1, padding='same', activation='relu'),
        layers.MaxPool2D(pool_size=2, strides=2, padding='same'),
        # 第二池化层输出扁平化1维
        layers.Flatten(),
        # 第一全连接层(启用随机神经元)
        layers.Dense(units=1024, activation='relu'),
        layers.Dropout(0.5),
        # 第二全连接层输出10(softmax)
        layers.Dense(units=10, activation='softmax'),
    ])

    # sgd = optimizers.SGD(learning_rate=0.2)
    adam = optimizers.Adam(learning_rate=1e-4)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    model.fit(x_train, y_train, batch_size=64, epochs=10)

    loss, accuracy = model.evaluate(x_test, y_test)
    print(f'test loss: {loss}, accuracy: {accuracy}')
    loss, accuracy = model.evaluate(x_train, y_train)
    print(f'train loss: {loss}, accuracy: {accuracy}')
    pass


def demo_10_rnn():
    """
    RNN(Recurrent Neural Network): 递归(循环)神经网络
    :return:
    """
    from keras import datasets, utils, models, layers, optimizers
    # mnist数据集 生成虚拟数据(60000, 28, 28)
    (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
    print(f"x_train shape:{x_train.shape}, y_train shape:{y_train.shape}")

    # (60000, 28, 28)
    x_train = x_train / 255.0
    x_test = x_test / 255.0
    print(f"x_train shape:{x_train.shape}, y_train shape:{y_train.shape}")
    y_train = utils.to_categorical(y_train, num_classes=10)
    y_test = utils.to_categorical(y_test, num_classes=10)

    # 数据长度一行28像素
    input_size = 28
    # 序列长度一共28行
    time_steps = 28
    # 隐藏层cell个数
    cell_size = 50
    model = models.Sequential([
        layers.Input(shape=(time_steps, input_size)),
        layers.SimpleRNN(units=cell_size),
        # 输出层10(softmax)
        layers.Dense(units=10, activation='softmax'),
    ])

    # sgd = optimizers.SGD(learning_rate=0.2)
    adam = optimizers.Adam(learning_rate=1e-4)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    model.fit(x_train, y_train, batch_size=64, epochs=10)

    loss, accuracy = model.evaluate(x_test, y_test)
    print(f'test loss: {loss}, accuracy: {accuracy}')
    loss, accuracy = model.evaluate(x_train, y_train)
    print(f'train loss: {loss}, accuracy: {accuracy}')
    pass


if __name__ == '__main__':
    # #线性回归
    # demo_01_sequential()
    # #非线性回归
    # demo_02_square()
    # #MNIST数据集
    # demo_03_mnist()
    # #交叉熵
    # demo_04_cross_entropy()
    # #dropout
    # demo_06_dropout()
    # #regularizers
    # demo_07_regularizers()
    # demo_08_optimizers()
    # demo_09_cnn()
    demo_10_rnn()
