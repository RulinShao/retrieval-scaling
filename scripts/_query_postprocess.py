import os
import json
import pickle
import pdb


def load_jsonl(path):
    data = []
    with open(path, 'r') as fin:
        for line in fin:
            data.append(json.loads(line))
    return data

def save_jsonl(data, path):
    with open(path, 'w') as fout:
        for ex in data:
            fout.write(json.dumps(ex) + '\n')


def post_process_claims():
    all_query_save_path = "/mnt/md-256k/comem/all_queries.jsonl"
    data_root_dir = "/mnt/md-256k/comem/extracted_claims"
    retrieved_dir = "/mnt/md-256k/scaling_out/retrieved_results/post_processed"
    retrieved_path = os.path.join(retrieved_dir, "merged_all_queries_top100.jsonl")
    dedupped_retrieved_path = os.path.join(retrieved_dir, "dedup_merged_all_queries_top100.jsonl")


    retrieved_results = load_jsonl(retrieved_path)
    hashed_retrieved_results = {}
    for ex in retrieved_results:
        hashed_retrieved_results[ex['query']] = ex['ctxs']


    dedupped_retrieved_results = load_jsonl(dedupped_retrieved_path)
    hashed_dedupped_retrieved_results = {}
    count_less_than_10 = 0
    for ex in dedupped_retrieved_results:
        hashed_dedupped_retrieved_results[ex['query']] = ex['ctxs'][:min(10, len(ex['ctxs']))]
        pdb.set_trace()
        if len(ex['ctxs']) < 10:
            count_less_than_10 += 1
            print(len(ex['ctxs']))
        
    print(f"{count_less_than_10} out of {len(retrieved_results)} examples have fewer than 10 retrieved passages.")


    dedupped_hashed_query_save_path = os.path.join(retrieved_dir, "hashed_retrieval_results.pkl")
    with open(dedupped_hashed_query_save_path, 'wb') as fout:
        pickle.dump(hashed_dedupped_retrieved_results, fout)


def post_process_karthik(k=100):
    raw_data_path = "/mnt/md-256k/comem/karthik/250.jsonl"
    all_query_save_path = "/mnt/md-256k/comem/karthik/all_queries_250.jsonl"
    retrieved_dir = "/mnt/md-256k/scaling_out/retrieved_results/post_processed"
    retrieved_path = os.path.join(retrieved_dir, "merged_all_queries_250_top2000.jsonl")
    dedupped_retrieved_path = os.path.join(retrieved_dir, "dedup_merged_all_queries_250_top2000.jsonl")

    raw_data = load_jsonl(raw_data_path)
    
    dedupped_retrieved_results = load_jsonl(dedupped_retrieved_path)
    hashed_dedupped_retrieved_results = {}
    count_less_than_k = 0
    for ex in dedupped_retrieved_results:
        hashed_dedupped_retrieved_results[ex['query']] = ex['ctxs'][:min(k, len(ex['ctxs']))]
        if len(ex['ctxs']) < k:
            count_less_than_k += 1
            print(len(ex['ctxs']))
        
    print(f"{count_less_than_k} out of {len(raw_data)} examples have fewer than {k} retrieved passages.")
    
    for ex in raw_data:
        key = 'prompt' if 'prompt' in ex.keys() else 'question'
        docs = hashed_dedupped_retrieved_results[ex[key]]
        ex['ctxs'] = docs[:k]

    save_path = f"/mnt/md-256k/comem/karthik/250_retrieved_top_{k}.jsonl"
    save_jsonl(raw_data, save_path)

if __name__ == '__main__':
    post_process_karthik()