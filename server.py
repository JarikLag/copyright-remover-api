import json
import os
import time
from io import BytesIO

import cherrypy
from cherrypy import log
import yaml
from PIL import Image
import psutil
import base64

import application.manager as manager

process = psutil.Process(os.getpid())  # for monitoring and debugging purposes

config = yaml.safe_load(open("config.yml"))


def process_copyrighted_request(body):
    image = read_image(body.get('image'))
    start = time.time()

    try:
        label = manager.classify_image(image)
        result = {
            'label': label
        }
    except Exception as e:
        result = {
            'error': str(e)
        }

    time_spent = time.time() - start
    log("Completed api call. Time spent {0:.3f} s".format(time_spent))

    return json.dumps(result)


def process_cut_object_request(body):
    image = read_image(body.get('image'))
    start = time.time()

    try:
        cutout = write_image(manager.cut_object(image))
        result = {
            'image': cutout
        }
    except Exception as e:
        result = {
            'error': str(e)
        }

    time_spent = time.time() - start
    log("Completed api call. Time spent {0:.3f} s".format(time_spent))

    return json.dumps(result)


def process_modify_request(body):
    image = read_image(body.get('image'))
    start = time.time()

    try:
        modified_image = write_image(manager.modify_image(image))

        result = {
            'image': modified_image
        }
    except Exception as e:
        result = {
            'error': str(e)
        }

    time_spent = time.time() - start
    log("Completed api call. Time spent {0:.3f} s".format(time_spent))

    return json.dumps(result)


class ApiServerController(object):
    @cherrypy.expose('/health')
    def health(self):
        result = {
            "status": "OK",  # TODO when is status not ok?
            "info": {
                "mem": "{0:.3f} MiB".format(process.memory_info().rss / 1024.0 / 1024.0),
                "cpu": process.cpu_percent(),
                "threads": len(process.threads())
            }
        }
        return json.dumps(result).encode("utf-8")

    @cherrypy.expose('/copyrighted')
    def copyrighted(self):
        cl = cherrypy.request.headers['Content-Length']
        raw = cherrypy.request.body.read(int(cl))
        body = json.loads(raw)
        return process_copyrighted_request(body).encode("utf-8")

    @cherrypy.expose('/modify')
    def modify(self):
        cl = cherrypy.request.headers['Content-Length']
        raw = cherrypy.request.body.read(int(cl))
        body = json.loads(raw)
        return process_modify_request(body).encode("utf-8")

    @cherrypy.expose('/cut-object')
    def cut_object(self):
        cl = cherrypy.request.headers['Content-Length']
        raw = cherrypy.request.body.read(int(cl))
        body = json.loads(raw)
        return process_cut_object_request(body).encode("utf-8")


def read_image(image_raw):
    return Image\
        .open(BytesIO(base64.b64decode(image_raw)))\
        .convert('RGB')


def write_image(image):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')


if __name__ == '__main__':
    cherrypy.tree.mount(ApiServerController(), '/')

    cherrypy.config.update({
        'server.socket_port': config["app"]["port"],
        'server.socket_host': config["app"]["host"],
        'server.thread_pool': config["app"]["thread_pool"],
        'log.access_file': "access1.log",
        'log.error_file': "error1.log",
        'log.screen': True,
        'tools.response_headers.on': True,
        'tools.encode.encoding': 'utf-8',
        'tools.response_headers.headers': [('Content-Type', 'application/json;encoding=utf-8')],
    })

    try:
        cherrypy.engine.start()
        cherrypy.engine.block()
    except KeyboardInterrupt:
        cherrypy.engine.stop()
