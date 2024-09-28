from omegaconf.omegaconf import OmegaConf
import contriever.src.contriever
import hydra
import pickle
import torch

from src.hydra_runner import hydra_runner
from src.index import Indexer, get_index_dir_and_passage_paths, get_index_passages_and_id_map
from src.search import embed_queries
import pdb


device = 'cuda' if torch.cuda.is_available()  else 'cpu'

class Pes2oDatastoreAPI():
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        print(f'\n{OmegaConf.to_yaml(self.cfg)}')
        self.index, self.passages, self.passage_id_map = self.load_pes2o_index()
        self.query_encoder, self.query_tokenizer, _ = contriever.src.contriever.load_retriever(self.cfg.model.query_encoder)
        self.query_encoder = self.query_encoder.to(device)
        self.pes2o_id_map = self.load_pes2o_id_mapping()
    
    def search(self, query, n_docs=3):
        query_embedding = self.embed_query(query)
        top_ids_and_scores = self.index.search_knn(query_embedding, n_docs)
        searched_passages, searched_scores, pes2o_ids = self.get_retrieved_passages(top_ids_and_scores)
        return {'pes2o IDs': pes2o_ids, 'scores': searched_scores, 'passages': searched_passages}
    
    def load_pes2o_index(self,):
        index_dir, _ = get_index_dir_and_passage_paths(self.cfg, self.cfg.datastore.index.index_shard_ids)
        index = Indexer(self.cfg.datastore.index.projection_size, self.cfg.datastore.index.n_subquantizers, self.cfg.datastore.index.n_bits)
        index.deserialize_from(index_dir)

        passages, passage_id_map = get_index_passages_and_id_map(self.cfg, self.cfg.datastore.index.index_shard_ids)
        assert len(passages) == index.index.ntotal, f"number of documents {len(passages)} and number of embeddings {index.index.ntotal} mismatch"
        return index, passages, passage_id_map
    
    def embed_query(self, query):
        query_embbeding = embed_queries(self.cfg.evaluation.search, [query], self.query_encoder, self.query_tokenizer, self.cfg.model.query_encoder)
        return query_embbeding
    
    def get_retrieved_passages(self, top_ids_and_scores):
        results_and_scores = top_ids_and_scores[0]
        docs = [self.passages[int(doc_id)]["text"] for doc_id in results_and_scores[0]]
        ids = [self.passages[int(doc_id)]["id"] for doc_id in results_and_scores[0]]
        pes2o_ids = [self.pes2o_id_map[_id][doc[:100]] for _id, doc in zip(ids, docs)]
        scores = [str(score) for score in results_and_scores[1]]
        return docs, scores, pes2o_ids
    
    def load_pes2o_id_mapping(self,):
        pes2o_mapping_file = '/gscratch/zlab/rulins/scaling-clean/api/updated_pes2o_id_mapping.pkl'
        with open(pes2o_mapping_file, 'rb') as file:
            id_mapping = pickle.load(file)
        return id_mapping
    

@hydra.main(config_path="/gscratch/zlab/rulins/scaling-clean/ric/conf", config_name="pes2o")
def get_datastore(cfg):
    ds = Pes2oDatastoreAPI(cfg)
    return ds


if __name__ == '__main__':
    ds = get_datastore()