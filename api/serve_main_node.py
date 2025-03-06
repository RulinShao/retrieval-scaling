

import requests
import pdb
import os
import json
import datetime
import hydra
import socket
from flask import Flask, jsonify, request
from flask_cors import CORS
import threading
import multiprocessing
import concurrent.futures
import queue
import hydra
from omegaconf import OmegaConf
from hydra.core.global_hydra import GlobalHydra


def extract_running_endpoints(file_path='running_ports_massiveds.txt'):
    """
    Extracts information from a text file and returns it as a list.
    Args:
        file_path (str): The path to the text file.
    Returns:
        list: A list of extracted information.
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()
    endpoints = []
    for line in lines:
        if '@' in line:
            endpoints.append(line.strip())
    return endpoints



def rerank_elements(element_list, k=-1):
    """
    Reranks a list of elements based on their scores in descending order.
    Args:
        element_list (list): A list of elements, where each element contains 'IDs', 'passages', and 'scores'.
    Returns:
        dict: The reranked element.
    """
    # Concatenate the results according to the keys
    concatenated_element = {
        'ID': [],
        'passage': [],
        'score': []
    }
    for element in element_list:
        for i in range(len(element['IDs'][0])):
            concatenated_element['ID'].append(element['IDs'][0][i])
            concatenated_element['passage'].append(element['passages'][0][i])
            concatenated_element['score'].append(element['scores'][0][i])
    # Rerank the values based on the scores in descending order
    sorted_indices = sorted(range(len(concatenated_element['score'])), key=lambda k: concatenated_element['score'][k], reverse=True)
    reranked_element = {
        'IDs': [[concatenated_element['ID'][i] for i in sorted_indices][:k]],
        'passages': [[concatenated_element['passage'][i] for i in sorted_indices][:k]],
        'scores': [[concatenated_element['score'][i] for i in sorted_indices][:k]]
    }
    return reranked_element


def test_main_node():
    json_data = {
        'query': 'retrieval-augmented language model',
        "n_docs": 2,
        "domains": "rpj_c4 (nprobes=128)"
    }
    headers = {"Content-Type": "application/json"}


    endpoints = extract_running_endpoints()
    print(endpoints)

    all_responses = []
    for endpoint in endpoints:
        # Add 'http://' to the URL if it is not SSL/TLS secured, otherwise use 'https://'
        response = requests.post('http://'+endpoint, json=json_data, headers=headers)
        all_responses.append(response.json()['results'])

        print(response.status_code)
        print(response.json())

    sorted_elements = rerank_elements(all_responses, k=response.json()['n_docs'])
    updated_response = {'n_docs': response.json()['n_docs'], 
                        'query': response.json()['query'], 
                        'results': sorted_elements,
                        }

    print(updated_response)



def load_config():
    # Ensuring Hydra is not already initialized which can cause issues in notebooks or multiple initializations
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    # Initialize Hydra and set the path to the config directory
    hydra.initialize(config_path="conf")

    # Compose the configuration (this loads the configuration files and merges them)
    cfg = hydra.compose(config_name="ivf_flat")

    # # Print or use the configuration as needed
    # print(OmegaConf.to_yaml(cfg))
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


def main_node_search(query, n_docs):
    json_data = {
        'query': query,
        "n_docs": n_docs,
        "domains": "rpj_c4 (nprobes=128)"
    }
    headers = {"Content-Type": "application/json"}


    endpoints = extract_running_endpoints()
    print(endpoints)

    all_responses = []
    for endpoint in endpoints:
        # Add 'http://' to the URL if it is not SSL/TLS secured, otherwise use 'https://'
        response = requests.post('http://'+endpoint, json=json_data, headers=headers)
        all_responses.append(response.json()['results'])

    sorted_elements = rerank_elements(all_responses, k=response.json()['n_docs'])
    updated_response = {'n_docs': response.json()['n_docs'], 
                        'query': response.json()['query'], 
                        'results': sorted_elements,
                        }

    print(updated_response)
    return updated_response


def fetch_endpoint(endpoint, json_data, headers):
    """
    Fetch data from a single endpoint.
    
    Args:
        endpoint (str): The endpoint URL.
        json_data (dict): The JSON data to send in the request body.
        headers (dict): The request headers.
    
    Returns:
        dict: The response JSON data.
    """
    response = requests.post('http://' + endpoint, json=json_data, headers=headers)
    return response.json()['results']

def main_node_multithread_search(query, n_docs):
    """
    Search multiple nodes and aggregate results.
    
    Args:
        query (str): The search query.
        n_docs (int): The number of documents to retrieve.
    
    Returns:
        dict: The aggregated search results.
    """
    json_data = {
        'query': query,
        "n_docs": n_docs,
        "domains": "rpj_c4 (nprobes=128)"
    }
    headers = {"Content-Type": "application/json"}
    endpoints = extract_running_endpoints()
    print(endpoints)
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(fetch_endpoint, endpoint, json_data, headers) for endpoint in endpoints]
        all_responses = [future.result() for future in futures]
    
    print(f"Searched from {len(all_responses)} shards.")
    sorted_elements = rerank_elements(all_responses, k=n_docs)
    updated_response = {'n_docs': n_docs, 
                        'query': query, 
                        'results': sorted_elements,
                        }
    print(updated_response)
    return updated_response


class SearchQueue:
    def __init__(self, log_queries=True):
        self.queue = queue.Queue()
        self.lock = threading.Lock()
        self.current_search = None
        self.cfg = load_config()

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
                results = main_node_multithread_search(item.query, item.n_docs)
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


def find_free_port():
    # https://stackoverflow.com/a/36331860
    with socket.socket() as s:
        s.bind(("", 0))  # Bind to a free port provided by the host.
        return s.getsockname()[1]  # Return the port number assigned.


if __name__ == '__main__':
    port = find_free_port()
    server_id = socket.gethostname()
    endpoint = f'rulin@{server_id}:{port}/search'
    print(endpoint)
    app.run(host='0.0.0.0', port=port)
    
    