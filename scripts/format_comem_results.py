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


orig_data_dir = "/checkpoint/amaia/explore/mingdachen/analysis/extracted_claims"
out_data_dir = "/checkpoint/amaia/explore/rulin/extracted_claims"


hashed_retrieval_results_path = '/home/rulin/hashed_retrieval_results.pkl'
with open(hashed_retrieval_results_path, 'rb') as fin:
    hashed_retrieval_results = pickle.load(fin)


for subdirname in os.listdir(orig_data_dir):
    orig_data_subdir = os.path.join(orig_data_dir, subdirname)
    out_data_subdir = os.path.join(out_data_dir, subdirname)
    os.makedirs(out_data_subdir, exist_ok=True)
    for filename in os.listdir(orig_data_subdir):
        orig_data_path = os.path.join(orig_data_subdir, filename)
        out_data_path = os.path.join(out_data_subdir, filename.replace('claims_', 'evidence_'))

        data = load_jsonl(orig_data_path)
        for ex in data:
            claim_search_results = {}
            for claim in ex['all_claims']:

                ctxs = hashed_retrieval_results[claim]
                top_10_ctxs = []
                for ctx in ctxs[:min(10, len(ctxs))]:
                    passage_info = {
                        'title': '',
                        'snippet': ctx['retrieval text'],
                        'link': ctx['source'],
                    }
                    top_10_ctxs.append(passage_info)
                
                claim_search_results[claim] = top_10_ctxs
            
            ex['claim_search_results'] = claim_search_results
        
        save_jsonl(data, out_data_path)

