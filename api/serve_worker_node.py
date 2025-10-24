import os
import json
import traceback
import datetime
import hydra
import socket
from flask import Flask, jsonify, request
from flask_cors import CORS
import threading
import queue
import time
import hydra
from omegaconf import OmegaConf
from hydra.core.global_hydra import GlobalHydra

from api.api_index import get_datastore


DS_DOMAIN = os.getenv('DS_DOMAIN')
NUM_SHARDS = int(os.getenv('NUM_SHARDS'))
NUM_SHARDS_PER_WORKER = int(os.getenv('NUM_SHARDS_PER_WORKER'))
WORKER_ID = int(os.getenv('WORKER_ID'))

shard_ids = [i for i in range(WORKER_ID*NUM_SHARDS_PER_WORKER, (WORKER_ID+1)*NUM_SHARDS_PER_WORKER)]


def load_config():
    # Ensuring Hydra is not already initialized which can cause issues in notebooks or multiple initializations
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    # Initialize Hydra and set the path to the config directory
    hydra.initialize(config_path="conf")

    # Get overrides from environment variables
    overrides = []
    for key, value in os.environ.items():
        if key.startswith('HYDRA_OVERRIDE_'):
            # Convert HYDRA_OVERRIDE_MODEL__DATASET to model.dataset
            config_key = key.replace('HYDRA_OVERRIDE_', '').lower().replace('__', '.')
            overrides.append(f"{config_key}={value}")

    # Compose the configuration with overrides
    cfg = hydra.compose(config_name="aws_h200", overrides=overrides)

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
    def __init__(self, log_queries=False):
        self.queue = queue.Queue()
        self.lock = threading.Lock()
        self.current_search = None
        self.cfg = load_config()
        self.cfg.datastore.domain = DS_DOMAIN
        self.cfg.datastore.embedding.num_shards = NUM_SHARDS
        self.datastore = get_datastore(self.cfg, shard_ids)

        self.log_queries = log_queries
        self.query_log = 'cached_queries.jsonl'
    
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
    try:
        item = Item(
            query=request.json['query'],
            domains=request.json['domains'],
            n_docs=request.json['n_docs'],
        )
        # Perform the search synchronously with 60s timeout
        timer = threading.Timer(60.0, lambda: (_ for _ in ()).throw(TimeoutError('Search timed out after 60 seconds')))
        timer.start()
        try:
            results = search_queue.search(item)
            timer.cancel()
            print(results)
            return jsonify({
                "message": f"Search completed for '{item.query}' from {item.domains}",
                "query": item.query,
                "n_docs": item.n_docs,
                "results": results,
            }), 200
        except TimeoutError as e:
            timer.cancel()
            return jsonify({
                "message": str(e),
                "query": item.query,
                "error": "timeout"
            }), 408
    except Exception as e:
        tb_lines = traceback.format_exception(e.__class__, e, e.__traceback__)
        error_message = f"An error occurred: {str(e)}\n{''.join(tb_lines)}"
        return jsonify({"message": error_message}), 500

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
    chunk_id = '-'.join([str(id) for id in shard_ids])
    domain_name = DS_DOMAIN
    serve_info = {'server_id': server_id, 'port': port, 'chunk_id': chunk_id}
    endpoint = f'rulin@{server_id}:{port}/search'  # replace with your username
    print(f'Running at {endpoint}')
    with open('running_ports_massiveds.jsonl', 'a+') as fout:
        info = {
            'domain_name': f'{domain_name}',
            'chunk_id': f'{chunk_id}',
            'endpoint': f'{endpoint}',
        }
        fout.write(json.dumps(info)+'\n')
    
    app.run(host='0.0.0.0', port=port)


if __name__ == '__main__':
    main()   
    
    """
    ##### Test #####
    curl -X POST rulin@cr1-h200-p5en48xlarge-712:33959/search -H "Content-Type: application/json" -d '{"query": "Where was Marie Curie born?", "n_docs": 1, "domains": "rpj_c4"}'
    """