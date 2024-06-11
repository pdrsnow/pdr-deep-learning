#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""
https://docs.python.org/zh-cn/3/library/http.server.html
"""
import json as js
import threading
from http import HTTPStatus, HTTPMethod
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Tuple, Optional


class RestRequest:
    def __init__(self, method: HTTPMethod, path: str, headers=None, body=None):
        self.method = method
        self.path = path
        self.headers = headers
        self.body = body


class RestResponse:
    def __init__(self, code: HTTPStatus, headers: dict, body: str, json: dict):
        self.code = code
        self.headers = headers
        self.body = body
        self.json = json


def _send_success(handler: BaseHTTPRequestHandler, response: RestResponse):
    headers = response.headers if response.headers else {}
    if response.body:
        _send_msg(handler, response.code, headers=headers, body=response.body)
    elif response.json:
        _send_msg(handler, response.code, headers=headers, body=js.dumps(response.json, ensure_ascii=False))
    else:
        _send_msg(handler, response.code, headers=headers, body='')


def _send_msg(handler: BaseHTTPRequestHandler, code: int, headers: dict, body: str):
    bs = body.encode(encoding='utf-8')
    length = len(bs)
    handler.send_response(code)
    for k, v in headers.items():
        handler.send_header(k, v)
    handler.send_header('Content-Type', 'application/json;charset=utf-8')
    handler.send_header('Conten-Length', str(length))
    handler.end_headers()
    if length > 0:
        handler.wfile.write(bs)
        handler.wfile.flush()
    pass


class RestRoute:
    def __init__(self, path: str, target: object, methods: Tuple[str | HTTPMethod] = (), name=''):
        self.methods = list()
        for method in methods:
            self.methods.append(HTTPMethod(method))
        self.path = path
        self._name_ = name
        if not callable(target):
            raise ValueError('expect is function of func')
        self._target_ = target
        self._args_ = target.__code__.co_varnames

    def _target_kwargs(self, request: RestRequest, response: RestResponse) -> dict:
        kwargs = dict()
        if 'request' in self._args_:
            kwargs.setdefault('request', request)
        if 'response' in self._args_:
            kwargs.setdefault('response', response)
        if 'method' in self._args_:
            kwargs.setdefault('method', request.method)
        if 'path' in self._args_:
            kwargs.setdefault('path', request.path)
        if 'headers' in self._args_:
            kwargs.setdefault('headers', request.headers)
        if 'body' in self._args_:
            kwargs.setdefault('body', request.body)
        return kwargs

    def handler(self, request: RestRequest, response: RestResponse):
        try:
            kwargs = self._target_kwargs(request, response)
            result = self._target_(**kwargs)
            if isinstance(result, str):
                response.body = result
            elif isinstance(result, dict):
                response.json = result
            setattr(response, 'code', HTTPStatus.OK)
        except Exception as e:
            print(f'handler exec except: {e}')
            response.code = HTTPStatus.BAD_REQUEST


class RestHTTPRequestHandler(BaseHTTPRequestHandler):
    """
    可以理解为java的 Servlet
    """

    def __init__(self, request, client_address, server):
        super().__init__(request, client_address, server)

    def do_GET(self):
        if isinstance(self.server, RestServer):
            self.server._handle_rest(self)

    def do_POST(self):
        if isinstance(self.server, RestServer):
            self.server._handle_rest(self)

    def do_PUT(self):
        if isinstance(self.server, RestServer):
            self.server._handle_rest(self)


class RestServer(ThreadingHTTPServer):
    routes = {}

    def __init__(self, port=8080, routes: Optional[list] = None):
        super().__init__(('0.0.0.0', port), RestHTTPRequestHandler)
        self.routes['/'] = RestRoute(path='/', target=self._home)
        for route in routes:
            self.routes[route.path] = route

    def _home(self):
        return 'Hello Word!'

    def _handle_rest(self, handler: BaseHTTPRequestHandler):
        path, method, headers, body = handler.path, HTTPMethod(handler.command), {}, ''
        route: RestRoute = self.routes.get(path.split('?')[0])
        if not route:
            return handler.send_error(HTTPStatus.NOT_FOUND)
        if route.methods and (method not in route.methods):
            return handler.send_error(HTTPStatus.METHOD_NOT_ALLOWED)
        try:
            for k, v in handler.headers.items():
                headers[k.lower()] = v
            limit = int(headers.get('content-length', '0'))
            body = handler.rfile.readline(limit).decode(encoding='utf-8')
            request = RestRequest(method=method, path=path, headers=headers, body=body)
            response = RestResponse(code=HTTPStatus.OK, headers={}, body='', json={})
            route.handler(request, response)
            _send_success(handler=handler, response=response)
        except Exception as e:
            print(f'server except: {e}')
            handler.send_error(HTTPStatus.BAD_REQUEST)

    def start(self):
        print('server starting......')
        td = threading.Thread(target=self.serve_forever)
        td.start()
        print('server started......')
        td.join()
