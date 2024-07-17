#!/usr/bin/python
# -*- coding: UTF-8 -*-

import logging.config
import os as os_
import sys
import warnings

import yaml

from rest.mapper import MyMapper

port: int = 8000
my_mapper: MyMapper


def load_config():
    APP_HOME = os_.environ.get('APP_HOME', os_.path.abspath('.'))
    with (open(file=f"{APP_HOME}/conf/application.yaml", mode='r', encoding="utf-8") as file):
        app_conf = yaml.load(file.read(), Loader=yaml.FullLoader)
        print(app_conf)
    global port, my_mapper
    port = app_conf['server']['port']
    mysql = app_conf['ds']['mysql']
    #
    my_mapper = MyMapper(**mysql)
    return app_conf


class LoggerWriter:
    def __init__(self, logger, level):
        self.logger = logger
        self.level = level

    def write(self, message):
        if message.rstrip():  # Avoid logging blank lines
            self.logger.log(self.level, message.rstrip())

    def flush(self):
        pass


def init_logs(logspath: str = ''):
    APP_HOME = os_.environ.get('APP_HOME', os_.path.abspath('.'))
    with (open(file=f"{APP_HOME}/conf/logging.yaml", mode='r', encoding="utf-8") as file):
        logConf = file.read().replace("${APP_HOME}", APP_HOME)
        logConf = logConf if not logspath else logConf.replace('/logs/model', f'/data/model/{logspath}')
        logging.config.dictConfig(yaml.load(logConf, Loader=yaml.FullLoader))
        # Add the handlers to the logger
        logger = logging.getLogger('root' if not logspath else 'modelLogger')
        # hide warnings
        warnings.filterwarnings("ignore")
        # Redirect stdout and stderr
        sys.stdout = LoggerWriter(logger, logging.INFO)
        sys.stderr = LoggerWriter(logger, logging.ERROR)


if __name__ == '__main__':
    load_config()
