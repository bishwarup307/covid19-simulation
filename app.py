"""
__author__: bishwarup
created: Friday, 27th March 2020 3:32:56 am
"""

from flask import Flask, Response, request, render_template, jsonify
import json
import time
from redis import Redis

app = Flask(__name__)

r = Redis()


def process_stream(d):
    l = [json.loads(x) for x in d.strip("\n").split("\n")]
    l = dict(zip(range(len(l)), l))
    return l


def get_data():
    time.sleep(1.0)
    l = r.get("city19").decode()
    return process_stream(l)


# def shutdown_server():
#     func = request.environ.get('werkzeug.server.shutdown')
#     if func is None:
#         raise RuntimeError('Not running with the Werkzeug Server')
#     func()


@app.route("/")
@app.route("/index")
def index():
    return render_template("index.html")


@app.route("/stream")
def stream():
    def eventStream():
        while True:
            try:
                yield f"event:update\ndata:{json.dumps(get_data())}\n\n"
            except AttributeError:
                yield f"event:BLANK_DB"

    return Response(eventStream(), mimetype="text/event-stream")


if __name__ == "__main__":
    app.run(debug=True)
