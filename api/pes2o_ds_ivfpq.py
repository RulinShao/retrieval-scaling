import os
from omegaconf.omegaconf import OmegaConf
import contriever.src.contriever
import hydra
import pickle
import json
import torch

from src.hydra_runner import hydra_runner
from src.indicies.ivfpq import IVFPQIndexer
from src.search import embed_queries
import pdb


device = 'cuda' if torch.cuda.is_available()  else 'cpu'

class Pes2oIVFPQDatastoreAPI():
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        print(f'\n{OmegaConf.to_yaml(self.cfg)}')
        # todo: not loading the passages all at once because it's too large
        self.index, self.index_id_to_db_id = self.load_pes2o_index()
        self.psg_pos_id_map = self.load_psg_pos_id_map()
        self.query_encoder, self.query_tokenizer, _ = contriever.src.contriever.load_retriever(self.cfg.model.query_encoder)
        self.query_encoder = self.query_encoder.to(device)
    
    def search(self, query, n_docs=3):
        query_embedding = self.embed_query(query)
        top_ids_and_scores = self.index.search_knn(query_embedding, n_docs)
        searched_passages, searched_scores, pes2o_ids = self.get_retrieved_passages(top_ids_and_scores)
        return {'pes2o IDs': pes2o_ids, 'scores': searched_scores, 'passages': searched_passages}
    
    def load_pes2o_index(self,):
        index_path = '/gscratch/zlab/rulins/pes2o_v3_ivfpq_3M_4096_64.index'
        index = IVFPQIndexer(index_path)
        meta_file = '/gscratch/zlab/rulins/pes2o_v3_ivfpq_3M_4096_64.meta.index'
        with open(meta_file, "rb") as reader:
            index_id_to_db_id = pickle.load(reader)
        return index, index_id_to_db_id

    def load_psg_pos_id_map(self,):
        psg_pos_id_map_file = '/gscratch/zlab/rulins/data/scaling_out/passages/pes2o_v3/16-shards/pos_map.pkl'
        with open(psg_pos_id_map_file, 'rb') as f:
            psg_pos_id_map = pickle.load(f)
        return psg_pos_id_map

    def id2psg(self, shard_id, chunk_id):
        filename, position = self.psg_pos_id_map[shard_id][chunk_id]
        with open(filename, 'r') as file:
            file.seek(position)
            line = file.readline()
        return json.loads(line)
    
    def embed_query(self, query):
        query_embbeding = embed_queries(self.cfg.evaluation.search, [query], self.query_encoder, self.query_tokenizer, self.cfg.model.query_encoder)
        return query_embbeding
    
    def get_passage(self, index_id):
        shard_id, chunk_id = self.index_id_to_db_id[index_id]
        return self.id2psg(shard_id, chunk_id)
    
    def get_retrieved_passages(self, top_ids_and_scores):
        passages = [self.get_passage(int(index_id)) for index_id in top_ids_and_scores[0][0]]
        docs = [passage["text"] for passage in passages]
        pes2o_ids = [passage["raw_id"] for passage in passages]
        scores = [str(score) for score in top_ids_and_scores[1][0]]
        return docs, scores, pes2o_ids
    

@hydra.main(config_path="/gscratch/zlab/rulins/scaling-clean/ric/conf", config_name="pes2o_v3")
def get_datastore(cfg):
    ds = Pes2oIVFPQDatastoreAPI(cfg)
    return ds

    

if __name__ == '__main__':
    ds = get_datastore()
    