import json
import datetime
import hydra
from flask import Flask, jsonify, request
from flask_cors import CORS
import threading
import queue
import hydra
from omegaconf import OmegaConf
from hydra.core.global_hydra import GlobalHydra

from pes2o_ds_ivfpq import get_datastore


def load_config():
    # Ensuring Hydra is not already initialized which can cause issues in notebooks or multiple initializations
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    # Initialize Hydra and set the path to the config directory
    hydra.initialize(config_path="conf")

    # Compose the configuration (this loads the configuration files and merges them)
    cfg = hydra.compose(config_name="pes2o")

    # Print or use the configuration as needed
    print(OmegaConf.to_yaml(cfg))
    return cfg


app = Flask(__name__)
CORS(app)


class Item:
    def __init__(self, query=None, query_embed=None, domains="pes2o", n_docs=1) -> None:
        self.query = query
        self.query_embed = query_embed
        self.domains = domains
        self.n_docs = n_docs
        self.searched_results = None
    
    def get_dict(self,):
        dict_item = {
            'query': self.query,
            'query_embed': self.query_embed,
            'domains': self.domains,
            'n_docs': self.n_docs,
            'searched_results': self.searched_results,
        }
        return dict_item


class SearchQueue:
    def __init__(self, log_queries=True):
        self.queue = queue.Queue()
        self.lock = threading.Lock()
        self.current_search = None
        self.cfg = load_config()
        self.datastore = get_datastore(self.cfg)

        self.log_queries = log_queries
        self.query_log = '/gscratch/scrubbed/rulins/api_query/2024_10_21_queries.jsonl'
    
    def search(self, item):
        with self.lock:
            if self.current_search is None:
                self.current_search = item
                if self.log_queries:
                    now = datetime.datetime.now()
                    formatted_time = now.strftime('%Y-%m-%d %H:%M:%S')
                    with open(self.query_log, 'a+') as fin:
                        fin.write(json.dumps({'time': formatted_time, 'query': item.query})+'\n')
                results = self.datastore.search(item.query, item.n_docs)
                self.current_search = None
                return results
            else:
                future = threading.Event()
                self.queue.put((item, future))
                future.wait()
                return item.searched_results
    
    def process_queue(self):
        while True:
            item, future = self.queue.get()
            with self.lock:
                self.current_search = item
                item.searched_results = self.datastore.search(item)
                self.current_search = None
            future.set()
            self.queue.task_done()

search_queue = SearchQueue()
threading.Thread(target=search_queue.process_queue, daemon=True).start()

@app.route('/search', methods=['POST'])
def search():
    item = Item(
        query=request.json['query'],
        domains=request.json['domains'],
        n_docs=request.json['n_docs'],
    )
    # Perform the search synchronously, but queue if another search is in progress
    results = search_queue.search(item)
    print(results)
    return jsonify({
        "message": f"Search completed for '{item.query}' from {item.domains}",
        "query": item.query,
        "n_docs": item.n_docs,
        "results": results,
    }), 200

@app.route('/current_search')
def current_search():
    with search_queue.lock:
        current = search_queue.current_search
        if current:
            return jsonify({
                "current_search": current.query,
                "domains": current.domains,
                "n_docs": current.n_docs,
            }), 200
        else:
            return jsonify({"message": "No search currently in progress"}), 200

@app.route('/queue_size')
def queue_size():
    size = search_queue.queue.qsize()
    return jsonify({"queue_size": size}), 200

@app.route("/")
def home():
    return jsonify("Hello! What you are looking for?")


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5005)