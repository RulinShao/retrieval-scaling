

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


def check_endpoint(endpoint):
    url = f"http://{endpoint}"
    headers = {"Content-Type": "application/json"}
    data = {
        "query": "Where was Marie Curie born?",
        "n_docs": 1,
        "domains": "MassiveDS"
    }
    try:
        # Set a short timeout to avoid hanging
        response = requests.post(url, json=data, headers=headers, timeout=5)
        return response.status_code == 200
    except (requests.exceptions.RequestException, requests.exceptions.Timeout):
        return False


def extract_running_endpoints(
    file_path='running_ports_massiveds.jsonl',
    check_endpoint_before_return=False,
    remove_invalid_endpoints_after_check=False,
):
    """
    Extracts information from a text file and returns it as a list.
    Args:
        file_path (str): The path to the text file.
    Returns:
        list: A list of extracted information.
    """
    endpoints = []
    with open(file_path, 'r') as file:
        for line in file:
            info = json.loads(line)
            endpoints.append(info['endpoint'])
    
    # Create a dictionary to track unique domain_name + chunk_id combinations
    unique_endpoints = {}
    with open(file_path, 'r') as file:
        for line in file:
            info = json.loads(line)
            key = (info['domain_name'], info['chunk_id'])
            
            # If we haven't seen this combination before, add it
            if key not in unique_endpoints:
                unique_endpoints[key] = info['endpoint']
            # If we have seen it, keep the valid endpoint
            else:
                # Check both endpoints and keep the valid one
                old_endpoint = unique_endpoints[key]
                new_endpoint = info['endpoint']
                
                old_valid = check_endpoint(old_endpoint)
                new_valid = check_endpoint(new_endpoint)
                
                if new_valid and not old_valid:
                    unique_endpoints[key] = new_endpoint

    # Update endpoints list with deduplicated values
    endpoints = list(unique_endpoints.values())

    num_endpoints_before_check = len(endpoints)
    print(f"Number of endpoints before check: {num_endpoints_before_check}")
    if check_endpoint_before_return:
        new_endpoints = []
        for endpoint in endpoints:
            if check_endpoint(endpoint):
                new_endpoints.append(endpoint)
        endpoints = new_endpoints
        
        num_endpoints_after_check = len(endpoints)
        print(f"Number of endpoints after check: {num_endpoints_after_check}")
        
        if num_endpoints_after_check != num_endpoints_before_check and remove_invalid_endpoints_after_check:
            with open('running_ports_massiveds.jsonl', 'w') as fout:
                for endpoint in endpoints:
                    fout.write(json.dumps(endpoint)+'\n')
    
    assert len(endpoints) == 13, f"Missing endpoints. Current alive endpoints: {len(endpoints)}"
        
    return endpoints


def test_extract_running_endpoints():
    endpoints = extract_running_endpoints(check_endpoint_before_return=True)
    print(endpoints)
    for i, endpoint in enumerate(endpoints):
        print(f"{i}: {endpoint}")


def rerank_elements(element_list, k=-1):
    """
    Reranks a list of elements based on their scores in descending order.
    Handles multiple batch sizes (bs>1).
    
    Args:
        element_list (list): A list of elements, where each element contains 'IDs', 'passages', and 'scores'.
        k (int): Maximum number of results to return per batch. If -1, return all.
        
    Returns:
        dict: The reranked elements, with each batch maintained separately.
    """
    # Create a new structure that preserves the batch dimension
    batch_size = max(len(element['scores']) for element in element_list)
    reranked_element = {
        'IDs': [[] for _ in range(batch_size)],
        'passages': [[] for _ in range(batch_size)],
        'scores': [[] for _ in range(batch_size)]
    }
    
    # Process each batch separately
    for batch_idx in range(batch_size):
        # Concatenate the results for this batch
        concatenated_batch = {
            'ID': [],
            'passage': [],
            'score': []
        }
        
        # Collect items from all elements for this batch index
        for element in element_list:
            # Skip if this element doesn't have data for this batch index
            if batch_idx >= len(element['scores']):
                continue
                
            for i in range(len(element['IDs'][batch_idx])):
                concatenated_batch['ID'].append(element['IDs'][batch_idx][i])
                concatenated_batch['passage'].append(element['passages'][batch_idx][i])
                concatenated_batch['score'].append(element['scores'][batch_idx][i])
        
        # Rerank based on scores in descending order for this batch
        sorted_indices = sorted(
            range(len(concatenated_batch['score'])), 
            key=lambda i: concatenated_batch['score'][i], 
            reverse=True
        )
        
        # Apply k limit if specified
        if k > 0:
            sorted_indices = sorted_indices[:k]
            
        # Add the sorted items for this batch
        reranked_element['IDs'][batch_idx] = [concatenated_batch['ID'][i] for i in sorted_indices]
        reranked_element['passages'][batch_idx] = [concatenated_batch['passage'][i] for i in sorted_indices]
        reranked_element['scores'][batch_idx] = [concatenated_batch['score'][i] for i in sorted_indices]
    
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
    
    try:
        endpoints = extract_running_endpoints()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(fetch_endpoint, endpoint, json_data, headers) for endpoint in endpoints]
            all_responses = [future.result() for future in futures]
    except:
        try:
            print(f"Main node search failed due to {e}, try to extract running endpoints again after 15 mins")
            time.sleep(15*60)
            endpoints = extract_running_endpoints(check_endpoint_before_return=True, remove_invalid_endpoints_after_check=True)
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [executor.submit(fetch_endpoint, endpoint, json_data, headers) for endpoint in endpoints]
                all_responses = [future.result() for future in futures]
        except Exception as e:
            print(f"Error: {e}")
            return {'Error': e}
    
    print(f"Searched from {len(all_responses)} shards.")
    sorted_elements = rerank_elements(all_responses, k=n_docs)
    updated_response = {'n_docs': n_docs, 
                        'query': query, 
                        'results': sorted_elements,
                        }
    print(updated_response)
    return updated_response


class SearchQueue:
    def __init__(self,):
        self.queue = queue.Queue()
        self.lock = threading.Lock()
        self.current_search = None
        self.cfg = load_config()
    
    def search(self, item):
        with self.lock:
            if self.current_search is None:
                self.current_search = item
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
    endpoint = f'rulin@{server_id}:{port}/search'  # NOTE: change to your own username; TODO: try user = getpass.getuser()
    print(endpoint)
    with open('running_ports_main_node.txt', 'a+') as fout:
        fout.write(f'Endpoints: \n')
        fout.write(endpoint)
        fout.write('\n')
    app.run(host='0.0.0.0', port=port)
    
    # test_extract_running_endpoints()