import os
import json
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


def preprocess_clamis():
    all_query_save_path = "/mnt/md-256k/comem/all_queries.jsonl"
    data_root_dir = "/mnt/md-256k/comem/extracted_claims"

    all_queries = []
    repeat_count, all_count = 0, 0
    for dirname in os.listdir(data_root_dir):
        data_dir = os.path.join(data_root_dir, dirname)
        for filename in os.listdir(data_dir):
            data_path = os.path.join(data_dir, filename)

            data = load_jsonl(data_path)

            for ex in data:
                for query in ex['all_claims']:
                    if {'query': query} in all_queries:
                        repeat_count += 1
                    else:
                        all_queries.append({'query': query})
                    all_count += 1

    print(f"Total repeated claims: {repeat_count} out of {all_count}")
    save_jsonl(all_queries, all_query_save_path)


def preprocess_prompt():
    all_query_save_path = "/mnt/md-256k/comem/karthik/all_queries_250.jsonl"
    data_path = "/mnt/md-256k/comem/karthik/250.jsonl"

    all_queries = []
    data = load_jsonl(data_path)
    for ex in data:
        ex['query'] = ex['prompt'] if 'prompt' in ex.keys() else ex['question']
        all_queries.append(ex)

    save_jsonl(all_queries, all_query_save_path)


if __name__ == '__main__':
    preprocess_prompt()