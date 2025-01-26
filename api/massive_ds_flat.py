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

class DatastoreAPI():
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        print(f'\n{OmegaConf.to_yaml(self.cfg)}')
        self.index, self.passages, self.passage_id_map = self.load_index()
        self.query_encoder, self.query_tokenizer, _ = contriever.src.contriever.load_retriever(self.cfg.model.query_encoder)
        self.query_encoder = self.query_encoder.to(device)
    
    def search(self, query, n_docs=3):
        query_embedding = self.embed_query(query)
        top_ids_and_scores = self.index.search_knn(query_embedding, n_docs)
        searched_passages, searched_scores = self.get_retrieved_passages(top_ids_and_scores)
        return {'scores': searched_scores, 'passages': searched_passages}
    
    def load_index(self,):
        # index_dir, _ = get_index_dir_and_passage_paths(self.cfg, ['0'])
        index = Indexer(self.cfg.datastore.index.projection_size, self.cfg.datastore.index.n_subquantizers, self.cfg.datastore.index.n_bits)
        index.deserialize_from('/home/rulin/retrieval-scaling/scaling_out/embeddings/facebook/contriever-msmarco/fineweb_edu_1m/1-shards/index/0')

        passages, passage_id_map = get_index_passages_and_id_map(self.cfg, ['0'])
        assert len(passages) == index.index.ntotal, f"number of documents {len(passages)} and number of embeddings {index.index.ntotal} mismatch"
        return index, passages, passage_id_map
    
    def embed_query(self, query):
        query_embbeding = embed_queries(self.cfg.evaluation.search, [query], self.query_encoder, self.query_tokenizer, self.cfg.model.query_encoder)
        return query_embbeding
    
    def get_retrieved_passages(self, top_ids_and_scores):
        results_and_scores = top_ids_and_scores[0]
        docs = [self.passages[int(doc_id)]["text"] for doc_id in results_and_scores[0]]
        ids = [self.passages[int(doc_id)]["id"] for doc_id in results_and_scores[0]]
        scores = [str(score) for score in results_and_scores[1]]
        return docs, scores
    

@hydra.main(config_path="/home/rulin/retrieval-scaling/ric/conf", config_name="example_config")
def get_datastore(cfg):
    ds = DatastoreAPI(cfg)
    return ds


if __name__ == '__main__':
    ds = get_datastore()