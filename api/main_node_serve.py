

import requests
import traceback
import pdb
import os
import json
import random
import pdb
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

endpoints_list = '/checkpoint/amaia/explore/rulin/retrieval-scaling/running_ports_c4_wiki_ip_fixed.txt'

def extract_running_endpoints(file_path=endpoints_list):
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
    sorted_indices = sorted(range(len(concatenated_element['score'])), key=lambda k: concatenated_element['score'][k], reverse=True) # set to True if the higher the better and vice versa
    reranked_element = {
        'IDs': [[concatenated_element['ID'][i] for i in sorted_indices][:k]],
        'passages': [[concatenated_element['passage'][i] for i in sorted_indices][:k]],
        'scores': [[concatenated_element['score'][i] for i in sorted_indices][:k]]
    }
    return reranked_element


def subsample_by_coin_flip(data, probability):
    """Subsample data by flipping a coin with a given probability."""
    return [item for item in data if random.random() < probability]

def apply_coin_flip_subsampling(reranked_element, probability):
    # Assuming the lists are of the same length
    length = len(reranked_element['IDs'][0])
    
    # Generate a mask for which indices to keep
    mask = [random.random() < probability for _ in range(length)]
    
    # Apply the mask to each list
    subsampled_ids = [id_ for id_, keep in zip(reranked_element['IDs'][0], mask) if keep]
    subsampled_passages = [passage for passage, keep in zip(reranked_element['passages'][0], mask) if keep]
    subsampled_scores = [score for score, keep in zip(reranked_element['scores'][0], mask) if keep]
    
    # Update the reranked_element with the subsampled lists
    reranked_element['IDs'] = subsampled_ids
    reranked_element['passages'] = subsampled_passages
    reranked_element['scores'] = subsampled_scores
    
    print(f"Before subsampling: {length}, after subsampling: {len(subsampled_ids)}")
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
    def __init__(self, query=None, query_embed=None, domains="MassiveDS", n_docs=1, subsample_ratio=1.0) -> None:
        self.query = query
        self.query_embed = query_embed
        self.domains = domains
        self.n_docs = n_docs
        self.subsample_ratio = subsample_ratio
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

def main_node_multithread_search(query, n_docs, subsample_ratio=1.0):
    """
    Search multiple nodes and aggregate results.
    
    Args:
        query (str): The search query.
        n_docs (int): The number of documents to retrieve.
    
    Returns:
        dict: The aggregated search results.
    """
    if subsample_ratio >= 0.5:
        K = 300
    elif subsample_ratio >= 0.25:
        K = 600
    elif subsample_ratio >= 0.125:
        K = 1500
    elif subsample_ratio >= 0.0625:
        K = 2500
    elif subsample_ratio >= 0.03125:
        K = 5000
    else:
        K = 10000
    
    json_data = {
        'query': query,
        "n_docs": n_docs if subsample_ratio == 1.0 else K,
        "domains": "MassiveDS"
    }
    headers = {"Content-Type": "application/json"}
    endpoints = extract_running_endpoints()
    print(endpoints)
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(fetch_endpoint, endpoint, json_data, headers) for endpoint in endpoints]
        all_responses = [future.result() for future in futures]
    
    print(f"Searched from {len(all_responses)} shards.")
    if subsample_ratio > 1 or subsample_ratio < 0:
        sorted_elements = {'Error message': f'Error: subsample ratio should be set between 0 and 1, but got {subsample_ratio}.'}
    else:
        sorted_elements = rerank_elements(all_responses, k=n_docs if subsample_ratio == 1.0 else K)
    
    if subsample_ratio != 1:
        print(f"Subsampling {subsample_ratio} documents.")
        sorted_elements = apply_coin_flip_subsampling(sorted_elements, subsample_ratio)
        sorted_elements = {
                'IDs': [sorted_elements['IDs'][:n_docs]],
                'passages': [sorted_elements['passages'][:n_docs]],
                'scores': [sorted_elements['scores'][:n_docs]],
            }
        
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
                results = main_node_multithread_search(item.query, item.n_docs, item.subsample_ratio)
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
        request_json = request.json
        if 'subsample_ratio' in request_json:
            item = Item(
                query=request_json['query'],
                domains=request_json['domains'],
                n_docs=request_json['n_docs'],
                subsample_ratio=request_json['subsample_ratio']
            )
            print(item)
        else:
            item = Item(
                query=request_json['query'],
                domains=request_json['domains'],
                n_docs=request_json['n_docs'],
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


if __name__ == '__main__':
    port = find_free_port()
    server_id = socket.gethostname()
    endpoint = f'rulin@{server_id}:{port}/search'
    print(endpoint)
    
    with open('running_ports_main_node.txt', 'a+') as fout:
        fout.write(f'Endpoints: {endpoints_list}\n')
        fout.write(endpoint)
        fout.write('\n')
    
    app.run(host='0.0.0.0', port=port)
    
    