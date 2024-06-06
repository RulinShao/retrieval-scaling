import os
import argparse
import csv
import json
import logging
import pickle
import time
import glob
from pathlib import Path
from tqdm import tqdm
import pdb
import re

import numpy as np
import torch
import transformers

import contriever.src.index
import contriever.src.contriever
import contriever.src.utils
import contriever.src.slurm
from contriever.src.evaluation import calculate_matches
import contriever.src.normalize_text

from src.data import load_eval_data
from src.search import get_merged_search_output_path, get_search_output_path
from src.decontamination import check_below_lexical_overlap_threshold

os.environ["TOKENIZERS_PARALLELISM"] = "true"

device = 'cuda' if torch.cuda.is_available()  else 'cpu'


class PplEvalOutput:
    def __init__(self, cfg, average_loss, perplexity, bit_per_byte, no_enough_docs_count=None):
        self.cfg = cfg
        self.average_loss = average_loss
        self.perplexity = perplexity
        self.bit_per_byte = bit_per_byte
        self.no_enough_docs_count = no_enough_docs_count
    
    def log_message(self,):
        msg = f"Domain = {self.cfg.evaluation.domain}" \
            f"\t DS_domain = {self.cfg.datastore.domain}" \
            f"\tconcate_k = {self.cfg.evaluation.concate_k}" \
            f"\tavg Loss = {self.average_loss:.4f}" \
            f"\tperplexity = {self.perplexity.item():.4f}" \
            f"\tbpb = {self.bit_per_byte.item():.4f}" \
            f"\ttotal shards = {self.cfg.datastore.embedding.num_shards}" \
            f"\tsampled shards = {len(self.cfg.datastore.index.index_shard_ids)}" \
            f"\t#eval samples = {self.cfg.evaluation.data.num_eval_samples}" \
            f"\tds chunk size = {self.cfg.datastore.embedding.chunk_size}" \
            f"\teval chunk size = {self.cfg.evaluation.data.max_eval_data_seq_length}" \
            f"\teval stride = {self.cfg.evaluation.data.eval_stride}" \
            f"\tall shards = {self.cfg.datastore.index.index_shard_ids}"
        if self.no_enough_docs_count:
            msg += f"\tno enough docs = {self.no_enough_docs_count}"

        return msg
    
    def log_short_message(self,):
        msg = f"Domain = {self.cfg.evaluation.domain}" \
            f"\ttotal shards = {self.cfg.datastore.embedding.num_shards}" \
            f"\t#eval samples = {self.cfg.evaluation.data.num_eval_samples}" \
            f"\tconcate_k = {self.cfg.evaluation.concate_k}" \
            f"\tavg Loss = {self.average_loss:.4f}" \
            f"\tperplexity = {self.perplexity.item():.4f}" \
            f"\tbpb = {self.bit_per_byte.item():.4f}"
        return msg


def evaluate_perplexity(cfg):
    if cfg.tasks.eval.task_name == 'perplexity_calibration':
        outputs = evaluate_calibration(cfg)
        return outputs
    
    eval_args = cfg.evaluation
    lm_only  = False if eval_args.concate_k else True

    if lm_only:    
        eval_data = load_eval_data(cfg)
    
    else:
        # eval_data_path = os.path.join(eval_args.eval_output_dir, os.path.basename(eval_args.data.eval_data).replace('.jsonl', '_retrieved_results.jsonl'))
        if eval_args.search.get('merged_path', None):
            eval_data_path = eval_args.search.merged_path
        else:
            eval_data_path = get_merged_search_output_path(cfg)
        eval_data = []
        with open(eval_data_path, 'r') as file:
            for line in file:
                ex = json.loads(line)
                eval_data.append(ex)
    
    all_context, all_answer, no_enough_docs_count = build_doc_prompts(eval_data, eval_args)  # prompt_k+...+prompt_1+query

    tokenizer = transformers.AutoTokenizer.from_pretrained(cfg.model.lm_model)
    try:
        lm = transformers.AutoModelForCausalLM.from_pretrained(
            cfg.model.lm_model,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        ).to(device)
    except:
        lm = transformers.AutoModelForCausalLM.from_pretrained(
            cfg.model.lm_model,
            torch_dtype=torch.bfloat16,
        ).to(device)

    pad_token = tokenizer.pad_token_id if tokenizer.eos_token_id is None else tokenizer.eos_token_id

    total_loss = 0
    total_samples = 0
    for i, (context, answer) in enumerate(zip(all_context, all_answer)):
        if eval_args.debug_mode and i % 10 == 0:
            print(f"Debugging mode:\nContext:\n{context}\nAnswer:\n{answer}\n")
        
        # todo: batch
        answer_ids = tokenizer(answer, return_tensors='pt', truncation=False).to(device)['input_ids']
        context_ids = tokenizer(context, return_tensors='pt', truncation=False).to(device)['input_ids']

        input_ids = torch.cat((context_ids, answer_ids), dim=1)
        labels = torch.cat((torch.full(context_ids.size(), -100).to(device), answer_ids.clone()), dim=1)  # mask out retrieval tokens and query tokens
        labels = torch.where(labels == pad_token, torch.tensor(-100), labels)  # mask out padded tokens

        # truncate from left
        input_ids = input_ids[:, -lm.config.max_position_embeddings:]
        labels = labels[:, -lm.config.max_position_embeddings:]

        with torch.no_grad():
            try:
                outputs = lm(input_ids, labels=labels)
            except:
                continue

        loss = outputs.loss.cpu().detach()
        print(loss)
        total_loss += loss.item() * input_ids.size()[0]
        total_samples += input_ids.size()[0]
    
    average_loss = total_loss / total_samples
    perplexity = torch.exp(torch.tensor(average_loss))
    entropy_bits = torch.log2(perplexity)
    bit_per_byte = entropy_bits / 8

    outputs = PplEvalOutput(cfg, average_loss, perplexity, bit_per_byte, no_enough_docs_count)

    logging.info(outputs.log_message())
    return outputs


def build_doc_prompts(eval_data, args):
    num_docs = args.concate_k
    decontamination, contamination_threshold, decontamination_method = args.get('decontamination', False), args.get('contamination_threshold', 0.5), args.get('decontamination_method', 'longest')
    use_continuation, use_both_doc_and_continuation = args.get('use_continuation', False), args.get('use_both_doc_and_continuation', False)
    
    # concate the doc with query regardless of length constraint
    # make sure the number of tokens retrieved text + query is smaller than max_seq_len
    # prepend the doc in a reverse order wrt relevance such that we can truncate tokens from left
    all_contexts, all_answers = [], []
    for ex in eval_data[1:]:
        answer = extract_answer(ex['raw_inputs'], ex['raw_query'])
        doc = ''
        no_enough_docs_count = 0
        if num_docs > 0:
            try:
                if ex['ctxs'][0] is not None:
                    doc_added = 0
                    doc_index = 0
                    while doc_added < num_docs and doc_index < len(ex['ctxs']):
                        
                        if use_both_doc_and_continuation:
                            print("Prepending both ctx and its continuation")
                            retrieved_text = ex['ctxs'][doc_index]['retrieval text'] + ex['ctxs'][doc_index]['retrieval next text'] + ' \n'
                        
                        elif use_continuation:
                            print("Prepending ctx's continuation")
                            retrieved_text = ex['ctxs'][doc_index]['retrieval next text'] + ' \n'
                        
                        else:
                            print("Prepending ctx")
                            retrieved_text = ex['ctxs'][doc_index]['retrieval text'] + ' \n'
                        
                        if decontamination:
                            if check_below_lexical_overlap_threshold(retrieved_text, answer, contamination_threshold, decontamination_method):
                                doc = retrieved_text + doc
                                doc_added += 1
                        
                        else:
                            doc = retrieved_text + doc
                            doc_added += 1
                        
                        doc_index += 1
                    
                    if doc_added == 0:
                        print("No document prepended!")
                    if doc_added < num_docs:
                        no_enough_docs_count += 1
            except:
                print("No document prepended!")
        
        context = doc + ex['raw_query']
        all_contexts.append(context)
        all_answers.append(answer)
    return all_contexts, all_answers, no_enough_docs_count

def extract_answer(raw_inputs, raw_query):
    inputs = raw_inputs.replace('<|endoftext|>', '')
    query = raw_query.replace('<|endoftext|>', '')
    try:
        answer = inputs.replace(query, '')
    except:
        try:
            answer = inputs.replace(query[:-1], '')
        except:
            answer = inputs[-len(inputs)//2:]
    return answer

def evaluate_calibration(cfg):
    eval_args = cfg.evaluation

    # eval_data_path = os.path.join(eval_args.eval_output_dir, os.path.basename(eval_args.data.eval_data).replace('.jsonl', '_retrieved_results.jsonl'))
    if eval_args.search.get('merged_path', None):
        eval_data_path = eval_args.search.merged_path
    else:
        eval_data_path = get_merged_search_output_path(cfg)

    eval_data = []
    with open(eval_data_path, 'r') as file:
        for line in file:
            ex = json.loads(line)
            eval_data.append(ex)
    eval_data = eval_data[1:]

    tokenizer = transformers.AutoTokenizer.from_pretrained(cfg.model.lm_model)
    pad_token = tokenizer.pad_token_id if tokenizer.eos_token_id is None else tokenizer.eos_token_id
    try:
        lm = transformers.AutoModelForCausalLM.from_pretrained(
            cfg.model.lm_model,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        ).to(device)
    except:
        lm = transformers.AutoModelForCausalLM.from_pretrained(
            cfg.model.lm_model,
            torch_dtype=torch.bfloat16,
        ).to(device)

    output_dir = cfg.evaluation.get('calibration_out_dir', "out_calibration")
    os.makedirs(output_dir, exist_ok=True)

    decontamination, contamination_threshold, decontamination_method = eval_args.get('decontamination', False), eval_args.get('contamination_threshold', 0.5), eval_args.get('decontamination_method', 'longest')
    use_continuation = eval_args.get('use_continuation', False)
    
    all_losses_min = []
    all_lm_losses_and_retrieval_scores = []
    for ex_id, ex in enumerate(eval_data):
        all_doc_inputs, all_query_inputs, all_retrieval_scores = build_doc_prompts_for_calibration(ex, eval_args.search.n_docs, decontamination, contamination_threshold, decontamination_method, use_continuation)
        
        lm_losses = []
        loss_min = float('inf')
        for doc_id, (doc, text, score) in enumerate(zip(all_doc_inputs, all_query_inputs, all_retrieval_scores)):
            docs_ids = tokenizer(doc, return_tensors='pt', truncation=False).to(device)['input_ids']
            text_ids = tokenizer(text, return_tensors='pt', truncation=False).to(device)['input_ids']
            input_ids = torch.cat((docs_ids, text_ids), dim=1)
            labels = torch.cat((torch.full(docs_ids.size(), -100).to(device), text_ids), dim=1)
            labels = torch.where(labels == pad_token, torch.tensor(-100), labels)

            # truncate from left
            input_ids = input_ids[:, -lm.config.max_position_embeddings:]
            labels = labels[:, -lm.config.max_position_embeddings:]

            with torch.no_grad():
                outputs = lm(input_ids, labels=labels)

            loss = outputs.loss.cpu().detach().item()
            lm_losses.append(loss)

            print(loss, score)
            loss_min = min(loss, loss_min)
            # with open(os.path.join(output_dir, f'{cfg.evaluation.domain}_{cfg.evaluation.data.num_eval_samples}_losses.jsonl'), 'a') as file:
            #     file.write(json.dumps({"ex_id": ex_id, "doc_id": doc_id, "loss": loss, "retrieval_score": score}) + "\n")
        
        all_lm_losses_and_retrieval_scores.append([lm_losses, all_retrieval_scores])
        all_losses_min.append(loss_min)
    
    with open(os.path.join(output_dir, f'calibration_results_{cfg.evaluation.domain}_{cfg.evaluation.data.num_eval_samples}_samples.pkl'), 'wb') as file:
        pickle.dump(all_lm_losses_and_retrieval_scores, file)
    
    average_loss_min = sum(all_losses_min) / len(all_losses_min)
    perplexity = torch.exp(torch.tensor(average_loss_min))
    entropy_bits = torch.log2(perplexity)
    bit_per_byte = entropy_bits / 8

    outputs = PplEvalOutput(cfg, average_loss_min, perplexity, bit_per_byte)

    logging.info(outputs.log_message())

    return outputs


def build_doc_prompts_for_calibration(ex, n_docs, decontamination=False, contamination_threshold=1, decontamination_method='longest', use_continuation=False):
    contexts, answers, scores = [], [], []
    doc_added = 0
    doc_index = 0
    answer = extract_answer(ex['raw_inputs'], ex['raw_query'])
    while doc_added < n_docs and doc_index < len(ex['ctxs']):
        try:
            retrieved_text = ex['ctxs'][doc_index]['retrieval text'] + ' \n'
        except:
            pdb.set_trace()
        if decontamination:
            if check_below_lexical_overlap_threshold(retrieved_text, answer, contamination_threshold, decontamination_method):
                doc_added += 1
                answers.append(answer)
                contexts.append(retrieved_text + ex['raw_query'])
                scores.append(float(ex['ctxs'][doc_index]['retrieval score']))
        else:
            doc_added += 1
            answers.append(answer)
            contexts.append(retrieved_text + ex['raw_query'])
            scores.append(float(ex['ctxs'][doc_index]['retrieval score']))
        doc_index += 1
    return contexts, answers, scores