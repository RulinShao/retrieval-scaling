import os
import time
import json
from omegaconf.omegaconf import OmegaConf
import contriever.src.contriever
from transformers import AutoTokenizer, AutoModel
import hydra
import pickle
import json
import torch

from src.hydra_runner import hydra_runner
from src.indicies.flat import FlatIndexer
from src.search import embed_queries
import pdb


device = 'cuda' if torch.cuda.is_available()  else 'cpu'

class FlatDatastoreAPI():
    def __init__(self, shard_id, cfg) -> None:
        self.cfg = cfg
        self.index = self.load_flat_index(shard_id=shard_id)
        # TODO support other query encoders
        self.query_encoder, self.query_tokenizer, _ = contriever.src.contriever.load_retriever('facebook/contriever-msmarco')
        self.query_encoder = self.query_encoder.to(device)
        
    
    def search(self, query, n_docs=3):
        query_embedding = self.embed_query(query)
        searched_scores, searched_passages, db_ids  = self.index.search(query_embedding, n_docs)
        results = {'scores': searched_scores, 'passages': searched_passages, 'IDs': db_ids}
        return results
    
    def load_flat_index(self, shard_id=0):
        embed_dir = self.cfg.datastore.embedding.embedding_dir
        passage_dir = self.cfg.datastore.embedding.passages_dir
        index_path = os.path.join(embed_dir, 'index', str(shard_id), 'index.faiss')
        meta_file = os.path.join(embed_dir, 'index', str(shard_id), 'index_meta.faiss')
        pos_map_save_path = os.path.join(passage_dir, 'passage_pos_id_map.pkl')
        index = FlatIndexer(
            index_path,
            meta_file,
            passage_dir=passage_dir,
            pos_map_save_path=pos_map_save_path,
        )
        return index
    
    def embed_query(self, query):
        query_embbeding = embed_queries(self.cfg.evaluation.search, [query], self.query_encoder, self.query_tokenizer, self.cfg.model.query_encoder)
        return query_embbeding
    

def get_datastore(cfg, shard_id):
    ds = FlatDatastoreAPI(shard_id=shard_id, cfg=cfg)
    # test_search(ds)
    return ds

@hydra.main(config_path="/home/rulin/retrieval-scaling/ric/conf/", config_name="pes2o")
def main(cfg):
    get_datastore(cfg, 0)

def test_search(ds):
    query = 'when was the last time anyone was on the moon?'  # 'scores': array([[44.3889  , 44.770973, 45.956238]], dtype=float32), 'IDs': [[[5, 45516], [6, 2218998], [5, 897337]]
    query2 = "who wrote he ain't heavy he's my brother lyrics?"  # 'scores': array([[33.60194 , 41.798004, 43.465225]], dtype=float32) 'IDs': [[[2, 361677], [5, 1717105], [2, 361675]]]
    # query: 'scores': array([[3540.2017, 3541.064 , 3541.2776]] 'IDs': [[[4, 270110], [6, 1770473], [4, 270109]]]
    # query2: 'scores': array([[3537.0122, 3543.1533, 3544.3677]] 'IDs': [[[4, 270110], [6, 1770473], [3, 1408820]]]
    search_results = ds.search(query, 1)
    print(search_results)
    pdb.set_trace()

def profile_time(ds):
    for i in range(30):
        if i == 10:
            start = time.time()
        search_results = ds.search("Sunny San Diego days", 3)
    end = time.time()
    print(search_results)
    print(f"Averaged Time: {(end-start)/20:.2f} seconds per query")
    


if __name__ == '__main__':
    main()
    
    