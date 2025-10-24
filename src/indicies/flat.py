import os
import re
import json
import time
import pickle
import faiss
import numpy as np
import torch

from src.indicies.index_utils import convert_pkl_to_jsonl, get_passage_pos_ids


os.environ["TOKENIZERS_PARALLELISM"] = "true"

device = 'cuda' if torch.cuda.is_available()  else 'cpu'


class FlatIndexer(object):

    def __init__(self, 
                embed_paths=None,
                index_path=None,
                meta_file=None,
                passage_dir=None,
                pos_map_save_path=None,
                dimension=768,
                ):
    
        self.embed_paths = embed_paths
        self.index_path = index_path  # path to store the final index
        self.meta_file = meta_file  # path to save the index id to db id map
        self.passage_dir = passage_dir
        self.pos_map_save_path = pos_map_save_path
        self.dimension=dimension
        self.cuda = False

        if os.path.exists(index_path) and os.path.exists(self.meta_file):
            print("Loading index...")
            self.index = faiss.read_index(index_path)
            self.index_id_to_db_id = self.load_index_id_to_db_id()
        else:
            self.index = faiss.IndexFlatIP(dimension)
            self.index_id_to_db_id = []
            print ("Building index...")
            self._build_index()
        
        if self.pos_map_save_path is not None:
            self.psg_pos_id_map = self.load_psg_pos_id_map()

    def _build_index(self,):
        start_time = time.time()
        for embed_path in self.embed_paths:
            filename = os.path.basename(embed_path)
            match = re.search(r"passages_(\d+)\.pkl", filename)
            shard_id = int(match.group(1))
                
            to_add = self.get_embs(shard_id=shard_id).copy()
            self.index.add(to_add)
            ids_toadd = [[shard_id, chunk_id] for chunk_id in range(len(to_add))]  #TODO: check len(to_add) is correct usage
            self.index_id_to_db_id.extend(ids_toadd)
            print ('Added %d / %d shards, (%d min)' % (shard_id+1, len(self.embed_paths), (time.time()-start_time)/60))
        
        faiss.write_index(self.index, self.index_path)
        with open(self.meta_file, 'wb') as fout:
            pickle.dump(self.index_id_to_db_id, fout)
        print ('Adding took {} s'.format(time.time() - start_time))
        
        print(f'Total data indexed {len(self.index_id_to_db_id)}')
        faiss.write_index(self.index, self.index_path)
        with open(self.meta_file, mode='wb') as f:
            pickle.dump(self.index_id_to_db_id, f)
        
    def load_embeds(self, shard_id=None):
        all_ids, all_embeds = [], []
        offset = 0
        for embed_path in self.embed_paths:
            loaded_shard_id = int(re.search(r'_(\d+).pkl$', embed_path).group(1))
            if shard_id is not None and loaded_shard_id != shard_id:
                continue
            print(f"Loading pickle embedding from {embed_path}...")
            with open(embed_path, "rb") as fin:
                ids, embeddings = pickle.load(fin)
            all_ids.extend([i + offset for i in ids])
            all_embeds.extend(embeddings)
            offset += len(ids)
        all_embeds = np.stack(all_embeds).astype(np.float32)
        datastore_size = len(all_ids)
        return all_embeds

    def get_embs(self, indices=None, shard_id=None):
        if indices is not None:
            embs = self.embs[indices]
        elif shard_id is not None:
            embs = self.load_embeds(shard_id)
        return embs
    
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
