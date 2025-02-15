import os
import logging
import numpy as np
import torch

from src.indicies.flat import FlatIndexer
from src.indicies.ivf_flat import IVFFlatIndexer
from src.indicies.ivf_pq import IVFPQIndexer
from src.indicies.index_utils import get_index_dir_and_embedding_paths


class Indexer(object):
    
    def __init__(self, cfg):
        self.cfg = cfg
        self.args = cfg.datastore.index
        self.index_type = self.args.index_type
        
        passage_dir = self.cfg.datastore.embedding.passages_dir
        index_dir, embedding_paths = get_index_dir_and_embedding_paths(cfg)
        os.makedirs(index_dir, exist_ok=True)
        logging.info(f"Indexing for passages: {embedding_paths}")
        if "IVF" in self.index_type:
            formatted_index_name = f"index_{self.index_type}.{self.args.sample_train_size}.{self.args.projection_size}.{self.args.ncentroids}.faiss"
            trained_index_path = os.path.join(index_dir, formatted_index_name+'.trained')
        else:
            formatted_index_name = f"index_{self.index_type}.faiss"
        index_path = os.path.join(index_dir, formatted_index_name)
        meta_file = os.path.join(index_dir, formatted_index_name+'.meta')
        pos_map_save_path = os.path.join(index_dir, 'passage_pos_id_map.pkl')
        
        if self.index_type == "Flat":
            self.datastore = FlatIndexer(
                embed_paths=embedding_paths,
                index_path=index_path,
                meta_file=meta_file,
                passage_dir=passage_dir,
                pos_map_save_path=pos_map_save_path,
                dimension=self.args.projection_size,
            )
        elif self.index_type == "IVFFlat":
            self.datastore = IVFFlatIndexer(
                embed_paths=embedding_paths,
                index_path=index_path,
                meta_file=meta_file,
                trained_index_path=trained_index_path,
                passage_dir=passage_dir,
                pos_map_save_path=pos_map_save_path,
                sample_train_size=self.args.sample_train_size,
                prev_index_path=None,
                dimension=self.args.projection_size,
                ncentroids=self.args.ncentroids,
                probe=self.args.probe,
            )
        elif self.index_type == "IVFPQ":
            self.datastore = IVFPQIndexer(
                embed_paths=embedding_paths,
                index_path=index_path,
                meta_file=meta_file,
                trained_index_path=trained_index_path,
                passage_dir=passage_dir,
                pos_map_save_path=pos_map_save_path,
                sample_train_size=self.args.sample_train_size,
                prev_index_path=None,
                dimension=self.args.projection_size,
                ncentroids=self.args.ncentroids,
                probe=self.args.probe,
                n_subquantizers=self.args.n_subquantizers,
                code_size=self.args.n_bits,
            )
        else:
            raise NotImplementedError
        
        
    def search(self, query_embs, k=5):
        all_scores, all_passages, db_ids = self.datastore.search(query_embs, k)
        return all_scores, all_passages, db_ids
    
