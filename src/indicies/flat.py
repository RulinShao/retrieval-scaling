import os
import json
import pickle
import faiss
import numpy as np
import torch

from src.indicies.index_utils import convert_pkl_to_jsonl, get_passage_pos_ids


os.environ["TOKENIZERS_PARALLELISM"] = "true"

device = 'cuda' if torch.cuda.is_available()  else 'cpu'


class FlatIndexer(object):

    def __init__(self, 
                index_path,
                meta_file,
                passage_dir=None,
                pos_map_save_path=None,
                ):
    
        self.index_path = index_path  # path to store the final index
        self.meta_file = meta_file  # path to save the index id to db id map
        self.passage_dir = passage_dir
        self.pos_map_save_path = pos_map_save_path
        self.cuda = False

        if os.path.exists(index_path) and os.path.exists(self.meta_file):
            print("Loading index...")
            self.index = faiss.read_index(index_path)
            self.index_id_to_db_id = self.load_index_id_to_db_id()
        else:
            raise NotImplementedError
        
        if self.pos_map_save_path is not None:
            self.psg_pos_id_map = self.load_psg_pos_id_map()

    def load_index_id_to_db_id(self,):
        with open(self.meta_file, "rb") as reader:
            index_id_to_db_id = pickle.load(reader)
        return index_id_to_db_id
    
    def build_passage_pos_id_map(self, ):
        convert_pkl_to_jsonl(self.passage_dir)
        passage_pos_ids = get_passage_pos_ids(self.passage_dir, self.pos_map_save_path)
        return passage_pos_ids

    def load_psg_pos_id_map(self,):
        if os.path.exists(self.pos_map_save_path):
            with open(self.pos_map_save_path, 'rb') as f:
                psg_pos_id_map = pickle.load(f)
        else:
            psg_pos_id_map = self.build_passage_pos_id_map()
        return psg_pos_id_map
    
    def _id2psg(self, shard_id, chunk_id):
        filename, position = self.psg_pos_id_map[shard_id][chunk_id]
        with open(filename, 'r') as file:
            file.seek(position)
            line = file.readline()
        return json.loads(line)
    
    def _get_passage(self, index_id):
        try:
            shard_id, chunk_id = self.index_id_to_db_id[index_id]
        except:
            shard_id, chunk_id = 0, self.index_id_to_db_id[index_id]
        return self._id2psg(shard_id, chunk_id)
    
    def get_retrieved_passages(self, all_indices):
        passages, db_ids = [], []
        for query_indices in all_indices:
            passages_per_query = [self._get_passage(int(index_id))["text"] for index_id in query_indices]
            db_ids_per_query = [self.index_id_to_db_id[int(index_id)] for index_id in query_indices]
            passages.append(passages_per_query)
            db_ids.append(db_ids_per_query)
        return passages, db_ids
    
    def search(self, query_embs, k=4096):
        all_scores, all_indices = self.index.search(query_embs.astype(np.float32), k)
        all_passages, db_ids = self.get_retrieved_passages(all_indices)
        return all_scores.tolist(), all_passages, db_ids
