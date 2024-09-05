from flask import Flask, jsonify, request
from flask_cors import CORS
import threading
import queue
import time
from api.utils import get_datastore

app = Flask(__name__)
CORS(app)

class Item():
    def __init__(self, query=None, query_embed=None, domains="all") -> None:
        self.query = query
        self.query_embed = query_embed
        self.domains = domains
        self.searched_results = None

class BackgroundThread(threading.Thread):
    def __init__(self):
        super().__init__()
        self.queue = queue.Queue()
        self.daemon = True
        self.datastore = get_datastore('MassiveDS-1.4T')

    def run(self):
        while True:
            try:
                item = self.queue.get(timeout=1)
                print(f"Processing item: {item}")
                search_results = self.datastore.search(item)
                self.queue.task_done()
            except queue.Empty:
                pass

    def search(self, item):
        self.queue.put(item)

background_thread = BackgroundThread()
background_thread.start()

@app.route('/search', methods=['POST'])
def search():
    item = Item(
        query=request.json['query'],
        domains=request.json['domains']
    )
    background_thread.search(item)
    return jsonify({"message": f"Searchd passages for '{item.query}' from {item.domains} added to queue"}), 200

@app.route('/queue_size')
def queue_size():
    size = background_thread.queue.qsize()
    return jsonify({"queue_size": size}), 200


if __name__ == '__main__':
    app.run(debug=True)