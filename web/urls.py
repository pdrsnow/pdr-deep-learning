#!/usr/bin/python
# -*- coding: UTF-8 -*-

from rest.server import RestRoute as route
from web.views import RestView


def load_routes(view: RestView):
    return [
        route("/hello", view.hello, name="你好"),
    ]
