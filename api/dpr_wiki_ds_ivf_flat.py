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
import pdb


device = 'cuda' if torch.cuda.is_available()  else 'cpu'

class IVFFlatDatastoreAPI():
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        print(f'\n{OmegaConf.to_yaml(self.cfg)}')
        self.index = self.load_dpr_wiki_ivf_flat_index()
        if "facebook/contriever" in self.cfg.model.query_encoder:
            self.query_encoder, self.query_tokenizer, _ = contriever.src.contriever.load_retriever(self.cfg.model.query_encoder)
        elif "dragon" in self.cfg.model.query_encoder:
            self.query_tokenizer = AutoTokenizer.from_pretrained(self.cfg.model.query_tokenizer)
            self.query_encoder = AutoModel.from_pretrained(self.cfg.model.query_encoder)
        self.query_encoder = self.query_encoder.to(device)
    
    def search(self, query, n_docs=3):
        query_embedding = self.embed_query(query)
        searched_scores, searched_passages  = self.index.search(query_embedding, n_docs)
        return {'scores': searched_scores, 'passages': searched_passages}
    
    def load_dpr_wiki_ivf_flat_index(self,):
        sample_train_size = 6000000
        projection_size = 768
        ncentroids = 8192
        probe = 128
        
        embed_dir = '/checkpoint/amaia/explore/comem/data/scaling_out/embeddings/facebook/dragon-plus-context-encoder/dpr_wiki/8-shards'
        embed_paths = [os.path.join(embed_dir, filename) for filename in os.listdir(embed_dir) if filename.endswith('.pkl')]
        formatted_index_name = f"index_ivf_flat.{sample_train_size}.{projection_size}.{ncentroids}.faiss"
        index_dir = '/checkpoint/amaia/explore/comem/data/scaling_out/embeddings/facebook/dragon-plus-context-encoder/dpr_wiki/8-shards/index_ivf_flat/'
        os.makedirs(index_dir, exist_ok=True)
        index_path = os.path.join(index_dir, formatted_index_name)
        meta_file = os.path.join(index_dir, formatted_index_name+'.meta')
        trained_index_path = os.path.join(index_dir, formatted_index_name+'.trained')
        passage_dir = '/checkpoint/amaia/explore/comem/data/massive_ds_1.4t/scaling_out/passages/dpr_wiki/8-shards'
        pos_map_save_path = os.path.join(index_dir, 'passage_pos_id_map.pkl')
        index = IVFFlatIndexer(
            embed_paths,
            index_path,
            meta_file,
            trained_index_path,
            passage_dir=passage_dir,
            pos_map_save_path=pos_map_save_path,
            sample_train_size=sample_train_size,
            dimension=projection_size,
            ncentroids=ncentroids,
            probe=probe,
        )
        return index
    
    def embed_query(self, query):
        query_embbeding = embed_queries(self.cfg.evaluation.search, [query], self.query_encoder, self.query_tokenizer, self.cfg.model.query_encoder)
        return query_embbeding
    

@hydra.main(config_path="/checkpoint/amaia/explore/rulin/retrieval-scaling/ric/conf", config_name="ivf_flat")
def get_datastore(cfg):
    ds = IVFFlatDatastoreAPI(cfg)
    return ds


def profile_time(ds):
    for i in range(30):
        if i == 10:
            start = time.time()
        search_results = ds.search('a sunny day', 3)
    end = time.time()
    print(search_results)
    print(f"Averaged Time: {(end-start)/20:.2f} seconds per query")
    

def nq_search(ds):
    query_path = '/checkpoint/amaia/explore/rulin/retrieval-scaling/examples/nq_open.jsonl'
    output_path = '/checkpoint/amaia/explore/rulin/retrieval-scaling/examples/nq_open_searched_128.jsonl'
    data = []
    with open(query_path, 'r') as fin:
        for line in fin:
            data.append(json.loads(line))
    
    searched_results = []
    for ex in data:
        query = ex["query"]
        search_results = ds.search(query, 3)
        documents = search_results['passages'][0]
        ctxs = []
        for doc in documents:
            ctxs.append({'retrieval text': doc})
        new_ex = {"query": query, "ctxs": ctxs}
        searched_results.append(new_ex)
    
    with open(output_path, 'w') as fout:
        for new_ex in searched_results:
            fout.write(json.dumps(new_ex)+'\n')

        


if __name__ == '__main__':
    ds = get_datastore()
    
    