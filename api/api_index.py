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
from src.indicies.ivf_flat import IVFFlatIndexer
from src.search import embed_queries
from src.indicies.base import Indexer


device = 'cuda' if torch.cuda.is_available()  else 'cpu'

class DatastoreAPI():
    def __init__(self, cfg, shard_id=None) -> None:
        if shard_id is not None:
            if isinstance(shard_id, list):
                cfg.datastore.index.index_shard_ids = shard_id
            else:
                cfg.datastore.index.index_shard_ids = [shard_id]
        self._index = Indexer(cfg)
        self.index = self._index.datastore
        
        model_name_or_path = cfg.model.query_encoder
        tokenizer_name_or_path = cfg.model.query_tokenizer
        if "contriever" in model_name_or_path:
            query_encoder, query_tokenizer, _ = contriever.src.contriever.load_retriever(model_name_or_path)
        elif "dragon" in model_name_or_path or "drama" in model_name_or_path:
            query_tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
            query_encoder = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True)
        elif "sentence-transformers" in model_name_or_path:
            query_tokenizer = None
            query_encoder = SentenceTransformer(model_name_or_path)
        elif "ReasonIR" in model_name_or_path or "GRIT" in model_name_or_path:
            from gritlm import GritLM
            query_tokenizer = None
            query_encoder = GritLM(model_name_or_path, torch_dtype="auto", mode="embedding")
        else:
            print(f"{model_name_or_path} is not supported!")
            raise AttributeError
        self.query_encoder = query_encoder
        self.query_tokenizer = query_tokenizer
        self.query_encoder = self.query_encoder.to(device)
        
        self.cfg = cfg
    
    def search(self, query, n_docs=3):
        query_embedding = self.embed_query(query)
        searched_scores, searched_passages, db_ids  = self.index.search(query_embedding, n_docs)
        results = {'scores': searched_scores, 'passages': searched_passages, 'IDs': db_ids}
        return results
    
    def embed_query(self, query):
        if isinstance(query, str):
            query_embedding = embed_queries(self.cfg.evaluation.search, [query], self.query_encoder, self.query_tokenizer, self.cfg.model.query_encoder)
        elif isinstance(query, list):
            query_embedding = embed_queries(self.cfg.evaluation.search, query, self.query_encoder, self.query_tokenizer, self.cfg.model.query_encoder)
        else:
            raise AttributeError(f"Query is not a string nor list!")
        return query_embedding
    

def get_datastore(cfg, shard_id=None):
    ds = DatastoreAPI(cfg=cfg, shard_id=shard_id)
    test_search(ds)
    return ds


@hydra.main(config_path="conf", config_name="aws_h200")
def main(cfg):
    get_datastore(cfg, 0)


def test_search(ds):
    query = 'when was the last time anyone was on the moon?'  # 'scores': array([[44.3889  , 44.770973, 45.956238]], dtype=float32), 'IDs': [[[5, 45516], [6, 2218998], [5, 897337]]
    query2 = "who wrote he ain't heavy he's my brother lyrics?"  # 'scores': array([[33.60194 , 41.798004, 43.465225]], dtype=float32) 'IDs': [[[2, 361677], [5, 1717105], [2, 361675]]]
    search_results = ds.search(query, 1)
    print(search_results)


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
    
    