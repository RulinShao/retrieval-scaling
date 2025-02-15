import os
import json
import random
import logging
import pickle
import time
import glob
from tqdm import tqdm
import pdb
from typing import List, Tuple, Any
from abc import ABC, abstractmethod
from omegaconf import ListConfig
import subprocess
import re

import faiss
import numpy as np
import torch
from transformers import GPTNeoXTokenizerFast

import contriever.src.contriever
import contriever.src.utils
import contriever.src.slurm
from contriever.src.evaluation import calculate_matches
import contriever.src.normalize_text

from src.indicies.index_utils import convert_pkl_to_jsonl, get_passage_pos_ids


os.environ["TOKENIZERS_PARALLELISM"] = "true"

device = 'cuda' if torch.cuda.is_available()  else 'cpu'


class IVFPQIndexer(object):

    def __init__(self, 
                embed_paths,
                index_path,
                meta_file,
                trained_index_path,
                passage_dir=None,
                pos_map_save_path=None,
                sample_train_size=1000000,
                prev_index_path=None,
                dimension=768,
                dtype=np.float16,
                ncentroids=4096,
                probe=2048,
                num_keys_to_add_at_a_time=1000000,
                DSTORE_SIZE_BATCH=51200000,
                n_subquantizers=16,
                code_size=8,
                ):
    
        self.embed_paths = embed_paths  # list of paths where saved the embedding of all shards
        self.index_path = index_path  # path to store the final index
        self.meta_file = meta_file  # path to save the index id to db id map
        self.prev_index_path = prev_index_path  # add new data to it instead of training new clusters
        self.trained_index_path = trained_index_path  # path to save the trained index
        self.passage_dir = passage_dir
        self.pos_map_save_path = pos_map_save_path
        self.cuda = False

        self.sample_size = sample_train_size
        self.dimension = dimension
        self.ncentroids = ncentroids
        self.probe = probe
        self.num_keys_to_add_at_a_time = num_keys_to_add_at_a_time
        self.n_subquantizers = n_subquantizers
        self.code_size = code_size

        if os.path.exists(index_path) and os.path.exists(self.meta_file):
            print("Loading index...")
            self.index = faiss.read_index(index_path)
            self.index_id_to_db_id = self.load_index_id_to_db_id()
            self.index.nprobe = self.probe
        else:
            self.index_id_to_db_id = []
            if not os.path.exists(self.trained_index_path):
                print ("Training index...")
                self._sample_and_train_index()

            print ("Building index...")
            self.index = self._add_keys(self.index_path, self.prev_index_path if self.prev_index_path is not None else self.trained_index_path)
        
        if self.pos_map_save_path is not None:
            self.psg_pos_id_map = self.load_psg_pos_id_map()

    def load_index_id_to_db_id(self,):
        with open(self.meta_file, "rb") as reader:
            index_id_to_db_id = pickle.load(reader)
        return index_id_to_db_id
    
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

    def get_knn_scores(self, query_emb, indices):
        assert self.embeds is not None
        embs = self.get_embs(indices) # [batch_size, k, dimension]
        scores = - np.sqrt(np.sum((np.expand_dims(query_emb, 1)-embs)**2, -1)) # [batch_size, k]
        return scores

    def _sample_and_train_index(self,):
        print(f"Sampling {self.sample_size} examples from {len(self.embed_paths)} files...")
        per_shard_sample_size = self.sample_size // len(self.embed_paths)
        all_sampled_embs = []
        for embed_path in self.embed_paths:
            print(f"Loading pickle embedding from {embed_path}...")
            with open(embed_path, "rb") as fin:
                _, embeddings = pickle.load(fin)
            shard_size = len(embeddings)
            print(f"Finished loading, sampling {per_shard_sample_size} from {shard_size} for training...")
            random_samples = np.random.choice(np.arange(shard_size), size=[min(per_shard_sample_size, shard_size)], replace=False)
            sampled_embs = embeddings[random_samples]
            all_sampled_embs.extend(sampled_embs)
        all_sampled_embs = np.stack(all_sampled_embs).astype(np.float32)
        
        print ("Training index...")
        start_time = time.time()
        self._train_index(all_sampled_embs, self.trained_index_path)
        print ("Finish training (%ds)" % (time.time()-start_time))

    def _train_index(self, sampled_embs, trained_index_path):
        quantizer = faiss.IndexFlatIP(self.dimension)
        start_index = faiss.IndexIVFPQ(quantizer,
                                       self.dimension,
                                       self.ncentroids,
                                       self.n_subquantizers,
                                       self.code_size,
                                       faiss.METRIC_INNER_PRODUCT
                                       )
        start_index.nprobe = self.probe
        np.random.seed(1)

        if self.cuda:
            # Convert to GPU index
            res = faiss.StandardGpuResources()
            co = faiss.GpuClonerOptions()
            co.useFloat16 = True
            gpu_index = faiss.index_cpu_to_gpu(res, 0, start_index, co)
            gpu_index.verbose = False

            # Train on GPU and back to CPU
            gpu_index.train(sampled_embs)
            start_index = faiss.index_gpu_to_cpu(gpu_index)
        else:
            # Faiss does not handle adding keys in fp16 as of writing this.
            start_index.train(sampled_embs)
        faiss.write_index(start_index, trained_index_path)

    def _add_keys(self, index_path, trained_index_path):
        index = faiss.read_index(trained_index_path)
        assert index.is_trained and index.ntotal == 0
        
        start_time = time.time()
        # NOTE: the shard id is a absolute id defined in the name
        for embed_path in self.embed_paths:
            filename = os.path.basename(embed_path)
            match = re.search(r"passages_(\d+)\.pkl", filename)
            shard_id = int(match.group(1))
                
            to_add = self.get_embs(shard_id=shard_id).copy()
            index.add(to_add)
            ids_toadd = [[shard_id, chunk_id] for chunk_id in range(len(to_add))]  #TODO: check len(to_add) is correct usage
            self.index_id_to_db_id.extend(ids_toadd)
            print ('Added %d / %d shards, (%d min)' % (shard_id+1, len(self.embed_paths), (time.time()-start_time)/60))
        
        faiss.write_index(index, index_path)
        with open(self.meta_file, 'wb') as fout:
            pickle.dump(self.index_id_to_db_id, fout)
        print ('Adding took {} s'.format(time.time() - start_time))
        return index
    
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
        shard_id, chunk_id = self.index_id_to_db_id[index_id]
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
        


def test_build_multi_shard_dpr_wiki():
    embed_dir = '/fsx-comem/rulin/data/truth_teller/scaling_out/embeddings/facebook/contriever-msmarco/dpr_wiki/1-shards'
    embed_paths = [os.path.join(embed_dir, filename) for filename in os.listdir(embed_dir) if filename.endswith('.pkl')]
    sample_train_size = 6000000
    projection_size = 768
    ncentroids = 4096
    n_subquantizers = 16
    code_size = 8
    formatted_index_name = f"index_ivf_pq_ip.{sample_train_size}.{projection_size}.{ncentroids}.faiss"
    index_dir = '/fsx-comem/rulin/data/truth_teller/scaling_out/embeddings/facebook/contriever-msmarco/dpr_wiki/1-shards/index_ivf_pq/'
    os.makedirs(index_dir, exist_ok=True)
    index_path = os.path.join(index_dir, formatted_index_name)
    meta_file = os.path.join(index_dir, formatted_index_name+'.meta')
    trained_index_path = os.path.join(index_dir, formatted_index_name+'.trained')
    pos_map_save_path = os.path.join(index_dir, 'passage_pos_id_map.pkl')
    passage_dir = '/fsx-comem/rulin/data/truth_teller/scaling_out/passages/dpr_wiki/1-shards'
    index = IVFPQIndexer(
        embed_paths,
        index_path,
        meta_file,
        trained_index_path,
        passage_dir=passage_dir,
        pos_map_save_path=pos_map_save_path,
        sample_train_size=sample_train_size,
        dimension=projection_size,
        ncentroids=ncentroids,
        n_subquantizers=n_subquantizers,
        code_size=code_size,
        )


def test_build_single_shard_dpr_wiki():
    embed_dir = '/checkpoint/amaia/explore/comem/data/scaling_out/embeddings/facebook/dragon-plus-context-encoder/dpr_wiki/8-shards'
    embed_paths = [os.path.join(embed_dir, filename) for filename in os.listdir(embed_dir) if filename.endswith('.pkl')]
    sample_train_size = 6000000
    projection_size = 768
    ncentroids = 4096
    n_subquantizers = 16
    code_size = 8
    formatted_index_name = f"index_ivf_pq_ip.{sample_train_size}.{projection_size}.{ncentroids}.faiss"
    index_dir = '/checkpoint/amaia/explore/comem/data/scaling_out/embeddings/facebook/dragon-plus-context-encoder/dpr_wiki/8-shards/index_ivf_pq/'
    os.makedirs(index_dir, exist_ok=True)
    index_path = os.path.join(index_dir, formatted_index_name)
    meta_file = os.path.join(index_dir, formatted_index_name+'.meta')
    trained_index_path = os.path.join(index_dir, formatted_index_name+'.trained')
    pos_map_save_path = os.path.join(index_dir, 'passage_pos_id_map.pkl')
    passage_dir = '/fsx-comem/rulin/data/truth_teller/scaling_out/passages/dpr_wiki/8-shards'
    index = IVFPQIndexer(
        embed_paths,
        index_path,
        meta_file,
        trained_index_path,
        passage_dir=passage_dir,
        pos_map_save_path=pos_map_save_path,
        sample_train_size=sample_train_size,
        dimension=projection_size,
        ncentroids=ncentroids,
        n_subquantizers=n_subquantizers,
        code_size=code_size,
        )

if __name__ == '__main__':
    test_build_multi_shard_dpr_wiki()