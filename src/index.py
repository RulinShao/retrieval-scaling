# Modified based on https://github.com/facebookresearch/contriever/blob/main/passage_retrieval.py
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import json
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

import faiss
import numpy as np
import torch
from transformers import GPTNeoXTokenizerFast
try:
    from pyserini.search.lucene import LuceneSearcher
except:
    logging.warning("Failed to import pyserini! Please install it from https://github.com/castorini/pyserini/tree/master.")

import contriever.src.contriever
import contriever.src.utils
import contriever.src.slurm
from contriever.src.evaluation import calculate_matches
import contriever.src.normalize_text

from src.data import fast_load_jsonl_shard
from src.indicies.base import Indexer
from src.indicies.index_utils import get_index_dir_and_embedding_paths


os.environ["TOKENIZERS_PARALLELISM"] = "true"


device = 'cuda' if torch.cuda.is_available()  else 'cpu'

def build_dense_index(cfg):
    index_args = cfg.datastore.index

    if isinstance(index_args.index_shard_ids[0], ListConfig):
        print(f"Multi-index mode: building {len(index_args.index_shard_ids)} index for {index_args.index_shard_ids} sequentially...")
        index_shard_ids_list = index_args.index_shard_ids
    else:
        print(f"Single-index mode: building a single index over {index_args.index_shard_ids} shards...")
        index_shard_ids_list = [index_args.index_shard_ids]
    
    for index_shard_ids in index_shard_ids_list:
        index = Indexer(cfg)


def get_index_passages_and_id_map(cfg, index_shard_ids=None):
    index_args = cfg.datastore.index

    index_shard_ids = index_shard_ids if index_shard_ids else index_args.get('index_shard_ids', None)
    assert index_shard_ids is not None
    
    index_shard_ids = [int(i) for i in index_shard_ids]

    passages = []
    passage_id_map = {}
    offset = 0
    for shard_id in index_shard_ids:
        shard_passages = fast_load_jsonl_shard(cfg.datastore.embedding, shard_id, return_all_passages=True)
        shard_id_map = {str(x["id"]+offset): x for x in shard_passages}
        
        offset += len(shard_passages)
        passages.extend(shard_passages)
        passage_id_map = {**passage_id_map, **shard_id_map}
        
    return passages, passage_id_map


class BM25Index(object):

    def __init__(self, index_dir, data_dir, stopwords):

        if not os.path.exists(index_dir):
            print ("Start building index for %s at %s" % (data_dir, index_dir))
            
            if stopwords:
                command = """python -m pyserini.index.lucene \
                --collection JsonCollection \
                --input '%s' \
                --index '%s' \
                --generator DefaultLuceneDocumentGenerator \
                --storeRaw --threads 1 \
                --stopwords '%s' """ % (data_dir, index_dir, stopwords)
            else:
                command = """python -m pyserini.index.lucene \
                --collection JsonCollection \
                --input '%s' \
                --index '%s' \
                --generator DefaultLuceneDocumentGenerator \
                --storeRaw --threads 1""" % (data_dir, index_dir)

            ret_code = subprocess.run([command],
                                    shell=True,
                                    #stdout=subprocess.DEVNULL,
                                    #stderr=subprocess.STDOUT
                                    )
            if ret_code.returncode != 0:
                print("Failed to build the index")
                exit()
            else:
                print("Successfully built the index")

        self.searcher = LuceneSearcher(index_dir)

    def search(self, query, k, continuation=False, shift=False, raw_only=True):
        # not used for simple raw text retrieval
        hits = self.searcher.search(query, k=k)
        out = []
        for hit in hits:
            docid = hit.docid

            if shift:
                docid = str(int(hit.docid)+1)
            
            raw = self.searcher.doc(docid).raw()
            
            if raw_only:
                if continuation:
                    next_item = self.searcher.doc(str(int(hit.docid)+1))
                    if next_item is not None:
                        next_raw = next_item.raw()
                        raw += next_raw  # todo: verify
                    else:
                        print ("The last block retrieved, so skipping continuation...")
                
                out.append(raw)
            
            else:
                input_ids = json.loads(raw)["input_ids"]

                if continuation:
                    next_item = self.searcher.doc(str(int(hit.docid)+1))
                    if next_item is not None:
                        next_raw = next_item.raw()
                        input_ids += json.loads(next_raw)["input_ids"]
                        raw += next_raw  # todo: verify
                    else:
                        print ("The last block retrieved, so skipping continuation...")

                out.append(input_ids)
        
        return out


def get_bm25_index_dir(cfg, index_shard_ids_list):
    shards_postfix = '_'.join([str(shard_id) for shard_id in index_shard_ids_list])
    index_dir = os.path.join(cfg.datastore.embedding.passages_dir, 'bm25')
    index_dir = os.path.join(index_dir, shards_postfix)
    return index_dir

def build_bm25_index(cfg):
    index_args = cfg.datastore.index
    stopwords = cfg.datastore.index.get("stopwords", None)

    if isinstance(index_args.index_shard_ids[0], ListConfig):
        print(f"Multi-index mode: building a BM25 index over {len(index_args.index_shard_ids)} shards...")
        index_shard_ids_list = [i for index_shards in index_args.index_shard_ids for i in index_shards]
    else:
        print(f"Single-index mode: building a BM25 index over {index_args.index_shard_ids} shards...")
        index_shard_ids_list = index_args.index_shard_ids
    
    bm25_base_path = get_bm25_index_dir(cfg, index_shard_ids_list)
    bm25_data_dir = os.path.join(bm25_base_path, 'data')
    bm25_index_dir = os.path.join(bm25_base_path, 'index')

    if not os.path.exists(bm25_index_dir):
        for index_shard_id in index_shard_ids_list:
            shard_passages, _ = get_index_passages_and_id_map(cfg, [index_shard_id])

            os.makedirs(bm25_data_dir, exist_ok=True)
            bm25_data_path = os.path.join(bm25_data_dir, f"data_{index_shard_id}.jsonl")
            if not os.path.exists(bm25_data_path):
                try:
                    with open(bm25_data_path, "w") as f:
                        for passage in tqdm(shard_passages):
                            f.write(json.dumps({
                                "id": str(passage["id"]),
                                "contents": passage["text"],
                            })+"\n")
                    logging.info(f"Saved passages to {bm25_data_path}.")
                except Exception as e:
                    logging.error(f"An error occurred: {e}")
                    os.remove(bm25_data_path)
                    logging.error(f"File '{bm25_data_path}' has been deleted due to an error.")
            else:
                logging.info(f"{bm25_data_path} exists, skipping..")
    
    logging.info(f'Loading/building bm25 search index from {bm25_index_dir}')
    searcher = BM25Index(bm25_index_dir, bm25_data_dir, stopwords)


def build_index(cfg):
    if cfg.model.get("sparse_retriever", None):
        build_bm25_index(cfg)
    else:
        build_dense_index(cfg)