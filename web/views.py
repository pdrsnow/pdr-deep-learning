#!/usr/bin/python
# -*- coding: UTF-8 -*-

import json as json_
import logging

from rest.server import RestRequest

logger = logging.getLogger()


class RestView:

    def hello(self, request: RestRequest):
        """
        """
        body = json_.loads(request.body)
        logger.info('req param: %s', body)
        body = {} if body is None else body

        return {'code': 200, 'msg': 'success', 'data': 'hello world!', **body}
