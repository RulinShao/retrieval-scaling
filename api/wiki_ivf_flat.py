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
    def __init__(self, shard_id, cfg) -> None:
        self.index = self.load_ivf_flat_index(shard_id=shard_id)
        self.query_encoder, self.query_tokenizer, _ = contriever.src.contriever.load_retriever('facebook/contriever-msmarco')
        self.query_encoder = self.query_encoder.to(device)
        
        self.cfg = cfg
        # print(f'\n{OmegaConf.to_yaml(self.cfg)}')
        # self.index = self.load_ivf_flat_index(shard_id=shard_id)
        self.cfg.model.query_encoder = 'facebook/contriever-msmarco' # fixed to contriever before figuring out the issue with dragon query encoder
        # if "facebook/contriever" in self.cfg.model.query_encoder:
        #     self.query_encoder, self.query_tokenizer, _ = contriever.src.contriever.load_retriever(self.cfg.model.query_encoder)
        # elif "dragon" in self.cfg.model.query_encoder:
        #     self.query_tokenizer = AutoTokenizer.from_pretrained(self.cfg.model.query_tokenizer)
        #     self.query_encoder = AutoModel.from_pretrained(self.cfg.model.query_encoder)
        # self.query_encoder = self.query_encoder.to(device)
    
    def search(self, query, n_docs=3):
        query_embedding = self.embed_query(query)
        searched_scores, searched_passages, db_ids  = self.index.search(query_embedding, n_docs)
        results = {'scores': searched_scores, 'passages': searched_passages, 'IDs': db_ids}
        return results
    
    def load_ivf_flat_index(self, shard_id=0):
        domain = 'dpr_wiki'
        num_shards = 1
        # domain = 'rpj_c4'
        # num_shards = 32
        sample_train_size = 6000000
        projection_size = 768
        ncentroids = 4096
        probe = 128
        
        embed_dir = f'/checkpoint/amaia/explore/comem/data/scaling_out/embeddings/facebook/contriever-msmarco/{domain}/{num_shards}-shards'
        embed_paths = [os.path.join(embed_dir, filename) for filename in os.listdir(embed_dir) if filename.endswith('.pkl')]
        formatted_index_name = f"index_ivf_flat_ip.{sample_train_size}.{projection_size}.{ncentroids}.faiss"
        index_dir = f'/checkpoint/amaia/explore/comem/data/scaling_out/embeddings/facebook/contriever-msmarco/{domain}/{num_shards}-shards/index_ivf_flat_{shard_id}/'
        os.makedirs(index_dir, exist_ok=True)
        index_path = os.path.join(index_dir, formatted_index_name)
        meta_file = os.path.join(index_dir, formatted_index_name+'.meta')
        trained_index_path = os.path.join(index_dir, formatted_index_name+'.trained')
        passage_dir = f'/checkpoint/amaia/explore/comem/data/massive_ds_1.4t/scaling_out/passages/{domain}/{num_shards}-shards'
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
    

def get_datastore(cfg, shard_id):
    ds = IVFFlatDatastoreAPI(shard_id=shard_id, cfg=cfg)
    # test_search(ds)
    return ds

@hydra.main(config_path="/checkpoint/amaia/explore/rulin/retrieval-scaling/ric/conf", config_name="ivf_flat")
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
    

def nq_search(ds):
    query_path = '/checkpoint/amaia/explore/rulin/retrieval-scaling/examples/nq_open.jsonl'
    output_path = '/checkpoint/amaia/explore/rulin/retrieval-scaling/examples/nq_open_ip_searched_4096_contriever.jsonl'
    data = []
    with open(query_path, 'r') as fin:
        for line in fin:
            data.append(json.loads(line))
    
    for ex in data:
        query = ex["query"]
        search_results = ds.search(query, 3)
        documents = search_results['passages'][0]
        ctxs = []
        for doc in documents:
            ctxs.append({'retrieval text': doc})
        new_ex = {"query": query, "ctxs": ctxs}
        print("#"*10)
        print("##### Query #####")
        print(query)
        print("##### Top-1 Passage #####")
        print(ctxs[0])
        print("#"*10)
        with open(output_path, 'a+') as fout:
            fout.write(json.dumps(new_ex)+'\n')



if __name__ == '__main__':
    main()
    
    