import json
import os
import time
from io import BytesIO

import cherrypy
from cherrypy import log
import yaml
import psutil
import base64

import application.manager as manager

process = psutil.Process(os.getpid())  # for monitoring and debugging purposes

config = yaml.safe_load(open("config.yml"))


def process_copyrighted_request(body):
    image_raw = body.get('image')
    start = time.time()

    try:
        label = manager.classify_image(image_raw)
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


def process_modify_request(body):
    image_raw = body.get('image')
    label = body.get('label')

    start = time.time()

    try:
        image = manager.modify_image(image_raw, label)
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        encoded = base64.b64encode(buffered.getvalue()).decode('utf-8')
        result = {
            'image': encoded
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
