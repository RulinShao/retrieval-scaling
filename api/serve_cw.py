import os
import json
import datetime
import hydra
import socket
from flask import Flask, jsonify, request
from flask_cors import CORS
import threading
import queue
import hydra
from omegaconf import OmegaConf
from hydra.core.global_hydra import GlobalHydra

from massive_ds_ivf_flat import get_datastore


def load_config():
    # Ensuring Hydra is not already initialized which can cause issues in notebooks or multiple initializations
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    # Initialize Hydra and set the path to the config directory
    hydra.initialize(config_path="conf")

    # Compose the configuration (this loads the configuration files and merges them)
    cfg = hydra.compose(config_name="ivf_flat")

    # Print or use the configuration as needed
    print(OmegaConf.to_yaml(cfg))
    return cfg


app = Flask(__name__)
CORS(app)


class Item:
    def __init__(self, query=None, query_embed=None, domains="MassiveDS", n_docs=1) -> None:
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
        self.datastore = get_datastore(self.cfg, int(os.getenv('CHUNK_ID')))

        self.log_queries = log_queries
        self.query_log = '/checkpoint/amaia/explore/rulin/api_query_cache/2024_11_14_queries.jsonl'
    
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
    try:
        results = search_queue.search(item)
        print(results)
        return jsonify({
            "message": f"Search completed for '{item.query}' from {item.domains}",
            "query": item.query,
            "n_docs": item.n_docs,
            "results": results,
        }), 200
    except Exception as e:
        return jsonify({"message": f"An error occured: {e}"}), -1

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


def find_free_port():
    # https://stackoverflow.com/a/36331860
    with socket.socket() as s:
        s.bind(("", 0))  # Bind to a free port provided by the host.
        return s.getsockname()[1]  # Return the port number assigned.


def main():
    port = find_free_port()
    server_id = socket.gethostname()
    chunk_id = os.getenv('CHUNK_ID')
    serve_info = {'server_id': server_id, 'port': port, 'chunk_id': int(os.getenv('CHUNK_ID'))}
    endpoint = f'rulin@{server_id}:{port}/search'
    print(f'Running at {endpoint}')
    with open('running_ports_c4_wiki.txt', 'a+') as fout:
        fout.write(f'Chunk: {chunk_id}\n')
        fout.write(endpoint)
        fout.write('\n')
        
    
    app.run(host='0.0.0.0', port=port)


if __name__ == '__main__':
    main()
    
    
    """
    curl -X POST rulin@cw-h100-211-159:49253/search -H "Content-Type: application/json" -d '{"query": "Where was Marie Curie born?", "n_docs": 1, "domains": "MassiveDS", "subsample_ratio": 0.5}'
    curl -X POST rulin@cw-h100-192-171:36109/search -H "Content-Type: application/json" -d '{"query": "Where was Marie Curie born?", "n_docs": 1, "domains": "rpj_c4"}'
    curl -X POST rulin@cw-h100-205-027:32943/search -H "Content-Type: application/json" -d '{"query": "Where was Marie Curie born?", "n_docs": 1, "domains": "rpj_c4 (nprobe=128)"}'
    curl -X POST rulin@cw-h100-219-147:55199/search -H "Content-Type: application/json" -d '{"query": "How much money, in euros, was the surgeon held responsible for Stella Obasanjo death ordered to pay her son?", "n_docs": 1, "domains": "rpj_c4 (nprobe=128)"}'
    """