import os

import argparse
import csv
import logging
import pickle
import pdb
from tqdm import tqdm

import numpy as np
import torch

from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer

import contriever.src.slurm
import contriever.src.contriever
import contriever.src.utils
import contriever.src.normalize_text

from src.data import fast_load_jsonl_shard


def embed_passages(args, passages, model, tokenizer):
    if "sentence-transformers" in args.model_name_or_path:
        allids, alltext = [], []
        for k, p in tqdm(enumerate(passages)):
            allids.append(p["id"])
            if args.no_title or not "title" in p:
                text = p["text"]
            else:
                text = p["title"] + " " + p["text"]
            if args.lowercase:
                text = text.lower()
            if args.normalize_text:
                text = contriever.src.normalize_text.normalize(text)
            alltext.append(text)
        
        with torch.no_grad():
            allembeddings = model.encode(alltext, batch_size=64)  # default is 512, but got oom
        
    else:
        total = 0
        allids, allembeddings = [], []
        batch_ids, batch_text = [], []
        with torch.no_grad():
            for k, p in tqdm(enumerate(passages)):
                batch_ids.append(p["id"])
                if args.no_title or not "title" in p:
                    text = p["text"]
                else:
                    text = p["title"] + " " + p["text"]
                if args.lowercase:
                    text = text.lower()
                if args.normalize_text:
                    text = contriever.src.normalize_text.normalize(text)
                batch_text.append(text)

                if len(batch_text) == args.per_gpu_batch_size or k == len(passages) - 1:

                    encoded_batch = tokenizer.batch_encode_plus(
                        batch_text,
                        return_tensors="pt",
                        max_length=args.passage_maxlength,
                        padding=True,
                        truncation=True,
                    )

                    encoded_batch = {k: v.cuda() for k, v in encoded_batch.items()}
                    embeddings = model(**encoded_batch)  # shape: (per_gpu_batch_size, hidden_size)
                    if "contriever" not in args.model_name_or_path:
                        # assume in hf form
                        embeddings = embeddings.last_hidden_state[:, 0, :]

                    embeddings = embeddings.cpu()
                    
                    total += len(batch_ids)
                    allids.extend(batch_ids)
                    allembeddings.append(embeddings)

                    batch_text = []
                    batch_ids = []
                    if k % 10000 == 0 and k > 0:
                        print(f"Encoded passages {total}")
        
        allembeddings = torch.cat(allembeddings, dim=0).numpy()
    
    return allids, allembeddings


def get_sharded_passages(args, all_passages):
    total_num_passages = len(all_passages)
    shard_size = total_num_passages // args.num_shards
    start_idx = args.shard_id * shard_size
    end_idx = start_idx + shard_size
    if args.shard_id == args.num_shards - 1:
        end_idx = total_num_passages
    
    passages = all_passages[start_idx: end_idx]
    print(f"Using {len(passages)} passages from idx {start_idx} to {end_idx}.")
    return passages


def generate_passage_embeddings(cfg):
    if cfg.model.get("sparse_retriever", None):
        print(f"No need to run the embedding step for sparse retrieval, skipping...")

    else:
        args = cfg.datastore.embedding
        
        logging.info(f"Loading retriever model from {args.model_name_or_path}...")
        if "contriever" in args.model_name_or_path:
            model, tokenizer, _ = contriever.src.contriever.load_retriever(args.model_name_or_path)
        elif "dragon" in args.model_name_or_path:
            tokenizer_name_or_path = args.tokenizer if args.get('tokenizer', None) else args.model_name_or_path
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
            model = AutoModel.from_pretrained(args.model_name_or_path)
        elif "sentence-transformers" in args.model_name_or_path:
            tokenizer = None
            model = SentenceTransformer(args.model_name_or_path)
        else:
            print(f"{args.model_name_or_path} is not supported!")
            raise AttributeError
        
        model.eval()
        model = model.cuda()
        if not args.no_fp16:
            model = model.half()
        
        shard_ids = [int(i) for i in args.shard_ids]

        for shard_id in shard_ids:
            embedding_shard_save_path = os.path.join(args.embedding_dir, args.prefix + f"_{shard_id:02d}.pkl")
            
            if os.path.exists(embedding_shard_save_path) and args.get("use_saved_if_exists", "true"):
                print(f"Embeddings exist in {embedding_shard_save_path}")
                continue
            
            shard_passages = fast_load_jsonl_shard(args, shard_id)

            allids, allembeddings = embed_passages(args, shard_passages, model, tokenizer)

            os.makedirs(args.embedding_dir, exist_ok=True)
            print(f"Saving {len(allids)} passage embeddings to {embedding_shard_save_path}.")
            with open(embedding_shard_save_path, mode="wb") as file:
                pickle.dump((allids, allembeddings), file)

            print(f"Processed {len(allids)} passages in the {shard_id}-th (out of {args.num_shards}) shard.\nWritten to {embedding_shard_save_path}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--raw_data_path", type=str, default=None, help="Path to passages (.jsonl or .tsv file)")
    parser.add_argument("--embedding_dir", type=str, default="wikipedia_embeddings", help="dir path to save embeddings")
    parser.add_argument("--prefix", type=str, default="passages", help="prefix path to save embeddings")
    parser.add_argument("--shard_id", type=int, default=0, help="Id of the current shard")
    parser.add_argument("--num_shards", type=int, default=1, help="Total number of shards")
    parser.add_argument(
        "--per_gpu_batch_size", type=int, default=512, help="Batch size for the passage encoder forward pass"
    )
    parser.add_argument("--chunk_size", type=int, default=512, help="Maximum number of words in a passage, the length will be further cut by passage_maxlength")
    parser.add_argument("--passage_maxlength", type=int, default=512, help="Maximum number of tokens in a passage")
    parser.add_argument(
        "--model_name_or_path", type=str, help="path to directory containing model weights and config file"
    )
    parser.add_argument("--no_fp16", action="store_true", help="inference in fp32")
    parser.add_argument("--no_title", action="store_true", help="title not added to the passage body")
    parser.add_argument("--lowercase", action="store_true", help="lowercase text before encoding")
    parser.add_argument("--normalize_text", action="store_true", help="lowercase text before encoding")

    args = parser.parse_args()

    generate_passage_embeddings(args)