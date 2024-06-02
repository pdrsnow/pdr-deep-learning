#!/usr/bin/python
# -*- coding: UTF-8 -*-

import argparse
import datetime
import json
import logging.config
import os as os_
import sys
import warnings
from datetime import datetime

import pytz
import requests
import yaml


class LoggerWriter:
    def __init__(self, logger, level):
        self.logger = logger
        self.level = level

    def write(self, message):
        if message.rstrip():  # Avoid logging blank lines
            self.logger.log(self.level, message.rstrip())

    def flush(self):
        pass


# print(sys.getdefaultencoding())
os_.environ["KERAS_BACKEND"] = "tensorflow"
os_.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
APP_HOME = os_.environ.setdefault('APP_HOME', os_.path.abspath('..'))
# 设置上海时区为全局时区
shanghai_tz = pytz.timezone("Asia/Shanghai")
session = requests.session()

# logging.config.fileConfig(f"{PAPP_HOME}/conf/logging.conf", encoding="utf-8")
with open(file=f"{APP_HOME}/conf/logging.yaml", mode='r', encoding="utf-8") as file:
    logConf = file.read().replace("${APP_HOME}", APP_HOME)
    logging.config.dictConfig(yaml.load(logConf, Loader=yaml.FullLoader))
    # Add the handlers to the logger
    logger = logging.getLogger()
    # logger.addHandler(file_handler)
    # logger.addHandler(console_handler)
    # hide warnings
    warnings.filterwarnings("ignore")
    # Redirect stdout and stderr
    sys.stdout = LoggerWriter(logger, logging.INFO)
    sys.stderr = LoggerWriter(logger, logging.ERROR)

import matplotlib.pyplot as plt
import pandas as pd
# import tensorflow as tf
from keras import models, layers
from sklearn import metrics, preprocessing, model_selection
from tensorflow.python.keras import backend as K


def get_time():
    res = datetime.now(shanghai_tz)
    return res


def get_date():
    curr_time = get_time()
    res = datetime.date(curr_time)
    return res


def get_taskdata_by_http(stime='', etime='', host_ip=''):
    task = None
    try:
        logging.info(f"[taskdata]历史数据取值区间: {stime} 到 {etime}")
        resp = session.post('https://pdrsnow.top/cesu/views/cs_cnn_task.action', json={
            'stime': stime, 'etime': etime, 'host_ip': host_ip
        })
        # 将结果转换为 DataFrame
        task = json.loads(resp.text)
    except (Exception) as error:
        print("Error while connecting to HTTP (taskdata)", error)
        task = {}
    finally:
        return task


def get_flowdata_by_http(stime='', etime='', host_ip=''):
    df = None
    try:
        logging.info(f"[flowdata]历史数据取值区间: {stime} 到 {etime}")
        resp = session.post('https://pdrsnow.top/cesu/views/cs_cnn_traffic.action', json={
            'stime': stime, 'etime': etime, 'host_ip': host_ip
        })
        # 将结果转换为 DataFrame
        df = pd.DataFrame(json.loads(resp.text))

        # plt.figure(figsize=(16, 8))
        # plt.plot(df['tx_mbs'], label='tx_mbs')
        # plt.plot(df['rx_mbs'], label='rx_mbs')
        # plt.legend()
        # plt.show()
    except (Exception) as error:
        print("Error while connecting to HTTP (flowdata)", error)
    finally:
        return df


def get_cpudata_by_http(stime='', etime='', host_ip=''):
    df_cpu = None
    try:
        logging.info(f"[cpudata]历史数据取值区间: {stime} 到 {etime}")
        resp = session.post('https://pdrsnow.top/cesu/views/cs_cnn_base_cpu.action', json={
            'stime': stime, 'etime': etime, 'host_ip': host_ip
        })
        # 将结果转换为 DataFrame
        df_cpu = pd.DataFrame(json.loads(resp.text))
    except (Exception) as error:
        print("Error while connecting to HTTP (cpudata)", error)
    finally:
        return df_cpu


def get_memdata_by_http(stime='', etime='', host_ip=''):
    df_men = None
    try:
        logging.info(f"[memdata]历史数据取值区间: {stime} 到 {etime}")
        resp = session.post('https://pdrsnow.top/cesu/views/cs_cnn_base_mem.action', json={
            'stime': stime, 'etime': etime, 'host_ip': host_ip
        })
        # 将结果转换为 DataFrame
        df_men = pd.DataFrame(json.loads(resp.text))
        df_men['used_G'] = df_men['used'].astype(int) / 1024 / 1024
    except (Exception) as error:
        print("Error while connecting to HTTP (memdata)", error)
    finally:
        return df_men


def send_taskdata_by_http(y_pred_data=None):
    df_men = None
    try:
        y_pred_data.to_json(orient='records')

        logging.info(f"[send_taskdata]历史数据取值区间: {stime} 到 {etime}")
        resp = session.post('https://pdrsnow.top/cesu/upload/cs_cnn_pred_traffic.action', json={
            'pred_time': stime, 'etime': etime, 'host_ip': host_ip
        })
        print(resp.text)
    except (Exception) as error:
        print("Error while connecting to HTTP (send_taskdata)", error)
    finally:
        return df_men

def data_preprocess(args, scaledData1, n_steps_in=60, n_steps_out=1):
    # 监督学习数据构造
    processedData1 = time_series_to_supervised(scaledData1, n_steps_in, n_steps_out)

    features = args.feature - 1
    begin_x = '0(t-' + str(n_steps_in) + ')'
    end_x = str(features) + '(t-1)'
    begin_y = '0(t+1)'
    end_y = str(features) + '(t+1)'

    data_x = processedData1.loc[:, begin_x:end_x]
    data_y = processedData1.loc[:, begin_y:end_y]

    # 划分测试集和训练集
    train_X1, test_X1, train_y, test_y = model_selection.train_test_split(
        data_x.values, data_y.values, test_size=0.4, random_state=343)
    # reshape input to be 3D [samples, timesteps, features]
    train_X = train_X1.reshape((train_X1.shape[0], n_steps_in, scaledData1.shape[1]))
    test_X = test_X1.reshape((test_X1.shape[0], n_steps_in, scaledData1.shape[1]))

    return train_X, train_y, test_X, test_y


def time_series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    """
    :param data:作为列表或2D NumPy数组的观察序列。需要。
    :param n_in:作为输入的滞后观察数（X）。值可以在[1..len（数据）]之间可选。默认为1。
    :param n_out:作为输出的观测数量（y）。值可以在[0..len（数据）]之间。可选的。默认为1。
    :param dropnan:Boolean是否删除具有NaN值的行。可选的。默认为True。
    :return:
    """
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    origNames = df.columns
    cols, names = list(), list()
    cols.append(df.shift(0))  # 不进行偏移,返回原始数据
    names += [('%s' % origNames[j]) for j in range(n_vars)]
    n_in = max(0, n_in)
    for i in range(n_in, 0, -1):
        time = '(t-%d)' % i
        cols.append(df.shift(i))
        names += [('%s%s' % (origNames[j], time)) for j in range(n_vars)]
    n_out = max(n_out, 0)
    for i in range(1, n_out + 1):
        time = '(t+%d)' % i
        cols.append(df.shift(-i))
        names += [('%s%s' % (origNames[j], time)) for j in range(n_vars)]
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    if dropnan:
        agg.dropna(inplace=True)
    return agg


def attention_model_2(INPUT_DIMS=13, TIME_STEPS=20, hidden_size=200, kernel_sizes=2):
    inputs = layers.Input(shape=(TIME_STEPS, INPUT_DIMS))
    # CNN: 卷积神经网络
    x = layers.Conv1D(filters=64, kernel_size=kernel_sizes, activation='relu')(inputs)
    x = layers.Dropout(0.3)(x)
    # BiLSTM: 双向LSTM
    lstm_out = layers.Bidirectional(layers.LSTM(hidden_size, return_sequences=True))(x)
    lstm_out = layers.Dropout(0.3)(lstm_out)
    # 注意力算法重排数据后，扁平化输出
    attention_mul = attention_3d_block(lstm_out)
    attention_mul = layers.Flatten()(attention_mul)
    # 全连接输出
    output = layers.Dense(INPUT_DIMS, activation='sigmoid')(attention_mul)
    model = models.Model(inputs=[inputs], outputs=output)
    return model


def attention_3d_block(inputs, single_attention_vector=False):
    time_steps = K.int_shape(inputs)[1]
    input_dim = K.int_shape(inputs)[2]
    a = layers.Permute((2, 1))(inputs)
    a = layers.Dense(time_steps, activation='softmax')(a)
    if single_attention_vector:
        a = layers.Lambda(lambda x: K.mean(x, axis=1))(a)
        a = layers.RepeatVector(input_dim)(a)

    a_probs = layers.Permute((2, 1))(a)
    # element-wise
    output_attention_mul = layers.Multiply()([inputs, a_probs])
    return output_attention_mul


if __name__ == '__main__':
    import numpy as np

    # 检测GPU是否可用
    # gpus = tf.config.list_physical_devices('GPU')
    # if gpus:
    # 设置仅使用第一块GPU
    # tf.config.set_visible_devices(gpus[0], 'GPU')

    task = get_taskdata_by_http()
    args = argparse.Namespace()
    setattr(args, 'description', 'Time Series forecast')  # 注意力机制混合模型
    # data
    setattr(args, 'model', task.get('model', 'CNN-BiLSTM-Attention'))  # 注意力机制混合模型
    setattr(args, 'window_size', int(task.get('window_size', 60)))  # 时间窗口大小, window_size > pre_len
    setattr(args, 'pre_len', int(task.get('pre_len', 1)))  # 预测未来数据长度
    # data
    setattr(args, 'date_length', int(task.get('date_length', 5)))  # 历史数据时间长度,单位是天
    setattr(args, 'target', task.get('target', 'tx_mbs'))  # 你需要预测的特征列，这个值会最后保存在csv文件里
    setattr(args, 'type', task.get('type', 'S'))  # [M, S],多元预测,单元预测
    setattr(args, 'feature', int(task.get('feature', 1)))  # 历史数据特征列，需要根据上一个type参数定义
    # learning
    setattr(args, 'lr', float(task.get('lr', 0.001)))  # 学习率
    setattr(args, 'drop_out', float(task.get('drop_out', 0.03)))  # 随机丢弃概率,防止过拟合
    setattr(args, 'epochs', int(task.get('epochs', 20)))  # 训练轮次
    setattr(args, 'batch_size', int(task.get('batch_size', 32)))  # 批次大小
    # model
    setattr(args, 'hidden_size', int(task.get('hidden_size', 200)))  # 隐藏层单元数
    setattr(args, 'kernel_sizes', int(task.get('kernel_sizes', 2)))  # 批次大小
    setattr(args, 'save', bool(task.get('save', False)))  # 模型权重是否保存,保存了就不用训练了

    #
    endtime = datetime.combine(get_date(), datetime.min.time())
    endtime = int(endtime.timestamp()) - 1
    starttime = endtime - (args.date_length * 24 * 60 * 60) + 1
    stime = task.get('stime', str(datetime.fromtimestamp(starttime)))
    etime = task.get('etime', str(datetime.fromtimestamp(endtime)))
    host_ip = task.get('host_ip', '172.53.32.6')
    # 需要历史数据的天数
    df_ = get_flowdata_by_http(stime, etime, host_ip)
    cpu_ = get_cpudata_by_http(stime, etime, host_ip)
    men_ = get_memdata_by_http(stime, etime, host_ip)

    df = df_.loc[df_['host_ip'] == host_ip, :]
    cpu = cpu_.loc[cpu_['host_ip'] == host_ip, :]
    men = men_.loc[men_['host_ip'] == host_ip, :]

    df.sort_values('create_time', inplace=True)
    cpu.sort_values('create_time', inplace=True)
    men.sort_values('create_time', inplace=True)

    df['key_time'] = pd.to_datetime(df['create_time'], format='%Y-%m-%d %H:%M:%S').dt.strftime('%Y-%m-%d %H:%M')
    cpu['key_time'] = pd.to_datetime(cpu['create_time'], format='%Y-%m-%d %H:%M:%S').dt.strftime('%Y-%m-%d %H:%M')
    men['key_time'] = pd.to_datetime(men['create_time'], format='%Y-%m-%d %H:%M:%S').dt.strftime('%Y-%m-%d %H:%M')

    v = df.merge(cpu, how="inner", on=['key_time'])
    v = v.merge(men, how="inner", on=['key_time'])
    v.sort_values('key_time', inplace=True)

    colunms = ['tx_mbs', 'used_G', 'si']
    data = v[colunms]

    data['tx_mbs'] = v['tx_mbs'].astype('float64')
    data['si'] = v['si'].astype('float64')

    # 定义可配置参数
    n_steps_in = args.window_size
    n_steps_out = args.pre_len
    epochs = args.epochs
    batch_size = args.batch_size

    # 数据标准化
    scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
    if args.type == 'S':
        scaledData1 = scaler.fit_transform(data[[args.target]])
    else:
        scaledData1 = scaler.fit_transform(data)

    train_X, train_y, test_X, test_y = data_preprocess(args, scaledData1, n_steps_in, n_steps_out)

    print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

    # # 训练是为了保存对应指标的模型权重值, 方便后期调用
    is_train = args.save
    if not is_train:
        # 训练模型
        model = attention_model_2(INPUT_DIMS=scaledData1.shape[1], TIME_STEPS=n_steps_in,
                                  hidden_size=args.hidden_size,
                                  kernel_sizes=args.kernel_sizes)
        model.compile(loss='mse', optimizer='adam')
        model.summary()

        history = model.fit([train_X], train_y, epochs=epochs, batch_size=batch_size, validation_split=0.01)
        # plot history
        plt.plot(history.history['loss'], label='train')
        plt.plot(history.history['val_loss'], label='test')
        plt.legend()

        # 设置坐标标签标注和字体大小
        plt.xlabel("Training loss", fontsize=10)
        plt.ylabel("Training epoch", fontsize=10)

        # 设置坐标刻度字体大小
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.show()

        # 保存模型
        model.save(f'{APP_HOME}/model/my_model-{host_ip}.keras')

    # 加载模型
    model = models.load_model(f'{APP_HOME}/model/my_model-{host_ip}.keras')

    # 预测
    yhat = model.predict(test_X)

    # 反归一化
    inv_forecast_y = scaler.inverse_transform(yhat)
    inv_test_y = scaler.inverse_transform(test_y)

    # 计算均方根误差
    # model evaluate
    index = colunms.index(args.target)
    y_true = inv_test_y[:, index]  # 真实值
    y_pred = inv_forecast_y[:, index]  # 预测值
    print("mae：", metrics.mean_absolute_error(y_true, y_pred))
    print("mse：", metrics.mean_squared_error(y_true, y_pred))
    print("rmse：", np.sqrt(metrics.mean_squared_error(y_true, y_pred)))
    print("mape：", np.sqrt(metrics.mean_absolute_percentage_error(y_true, y_pred)))
    print("r2：", np.sqrt(metrics.r2_score(y_true, y_pred)))

    y_pred_data = pd.DataFrame(y_pred[-n_steps_out:], columns=[args.target])
    send_taskdata_by_http(y_pred_data)

    # 画图
    plt.figure(figsize=(16, 8))
    plt.plot(inv_test_y[:, index], label='true')
    plt.plot(inv_forecast_y[:, index], label='pred')
    plt.legend()
    plt.show()
    print("结束")
