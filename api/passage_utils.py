import os
import pickle
import json
from tqdm import tqdm


def convert_pkl_to_jsonl(passage_dir):
    # write down to avoid confusion; modify to general function when integrating code.
    pkl_files = [
        "raw_passages-0-of-16.pkl",   "raw_passages-12-of-16.pkl",  "raw_passages-15-of-16.pkl", 
        "raw_passages-3-of-16.pkl",   "raw_passages-6-of-16.pkl",   "raw_passages-9-of-16.pkl",
        "raw_passages-10-of-16.pkl",  "raw_passages-13-of-16.pkl",  "raw_passages-1-of-16.pkl",  
        "raw_passages-4-of-16.pkl",   "raw_passages-7-of-16.pkl",
        "raw_passages-11-of-16.pkl",  "raw_passages-14-of-16.pkl",  "raw_passages-2-of-16.pkl",  
        "raw_passages-5-of-16.pkl",   "raw_passages-8-of-16.pkl"
    ]
    for file in tqdm(pkl_files):
        
        # Create the JSONL file name
        file_path = os.path.join(passage_dir, file)
        jsonl_file = file_path.replace('.pkl', '.jsonl')
        
        if os.path.exists(jsonl_file):
            continue
        
        # Load the pickle file
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        # Save the data to the JSONL file
        with open(jsonl_file, 'w') as f:
            for item in data:
                json.dump(item, f)
                f.write('\n')
    print("All pickle files have been converted to JSONL files.")


def get_passage_pos_ids(passage_dir, pos_map_save_path):
    if os.path.exists(pos_map_save_path):
        with open(pos_map_save_path, 'rb') as f:
            pos_id_map = pickle.load(f)
        return pos_id_map

    # write down to avoid confusion; modify to general function when integrating code.
    jsonl_files = [
        "raw_passages-0-of-16.jsonl",   "raw_passages-12-of-16.jsonl",  "raw_passages-15-of-16.jsonl", 
        "raw_passages-3-of-16.jsonl",   "raw_passages-6-of-16.jsonl",   "raw_passages-9-of-16.jsonl",
        "raw_passages-10-of-16.jsonl",  "raw_passages-13-of-16.jsonl",  "raw_passages-1-of-16.jsonl",  
        "raw_passages-4-of-16.jsonl",   "raw_passages-7-of-16.jsonl",
        "raw_passages-11-of-16.jsonl",  "raw_passages-14-of-16.jsonl",  "raw_passages-2-of-16.jsonl",  
        "raw_passages-5-of-16.jsonl",   "raw_passages-8-of-16.jsonl"
    ]
    pos_id_map = {}
    for shard_id in tqdm(range(len(jsonl_files))):
        filename = f"raw_passages-{shard_id}-of-16.jsonl"
        file_path = os.path.join(passage_dir, filename)
        file_pos_id_map = {}
        with open(file_path, 'r') as file:
            position = file.tell()
            line = file.readline()
            doc_id = 0
            while line:
                file_pos_id_map[doc_id] = [file_path, position]
                doc_id += 1
                position = file.tell()
                line = file.readline()
        pos_id_map[shard_id] = file_pos_id_map
    
    # Save the output map to a pickle file
    if pos_map_save_path is not None:
        with open(pos_map_save_path, 'wb') as f:
            pickle.dump(pos_id_map, f)
        print(f"Output map saved to {pos_map_save_path}")
    return pos_id_map
    


if __name__ == '__main__':
    passage_dir = '/gscratch/zlab/rulins/data/scaling_out/passages/pes2o_v3/16-shards'
    convert_pkl_to_jsonl(passage_dir)
    
    pos_map_save_path = '/gscratch/zlab/rulins/data/scaling_out/passages/pes2o_v3/16-shards/pos_map.pkl'
    get_passage_pos_ids(passage_dir, pos_map_save_path)