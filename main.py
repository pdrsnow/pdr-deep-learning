#!/usr/bin/python
# -*- coding: UTF-8 -*-

from rest import config

config.load_config()
from rest.server import RestServer
from web import urls, views

if __name__ == '__main__':
    config.init_logs()

    view = views.RestView()

    routes = urls.load_routes(view)
    server = RestServer(config.port, routes)
    server.start()
