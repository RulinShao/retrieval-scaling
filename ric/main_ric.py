import os

import logging
from omegaconf.omegaconf import OmegaConf, open_dict

from src.hydra_runner import hydra_runner
from src.embed import generate_passage_embeddings
from src.index import build_index
from src.search import search_topk, post_hoc_merge_topk_multi_domain
from src.evaluate_perplexity import evaluate_perplexity


@hydra_runner(config_path="conf", config_name="default")
def main(cfg) -> None:
    
    logging.info("\n\n************** Experiment configuration ***********")
    logging.info(f'\n{OmegaConf.to_yaml(cfg)}')

    if cfg.tasks.datastore.get('embedding', False):
        logging.info("\n\n************** Building Embedding ***********")
        generate_passage_embeddings(cfg)
    
    if cfg.tasks.datastore.get('index', False):
        logging.info("\n\n************** Indexing ***********")
        build_index(cfg)
    
    if cfg.tasks.eval.get('search', False):
        logging.info("\n\n************** Running Search ***********")
        search_topk(cfg)
    
    if cfg.tasks.eval.get('merge_search', False):
        logging.info("\n\n************** Post Merging Searched Results from Multiple Domains ***********")
        post_hoc_merge_topk_multi_domain(cfg)
    
    if cfg.tasks.eval.get('inference', False):
        logging.info("\n\n************** Running Perplexity Evaluation ***********")
        outputs = evaluate_perplexity(cfg)
        log_results_separately(cfg, outputs)
    


def log_results_separately(cfg, outputs):
    if cfg.evaluation.get('results_only_log_file', None):
        with open(cfg.evaluation.results_only_log_file, "a+") as file:
            file.write('\n')
            file.write(outputs.log_message())

if __name__ == '__main__':
    main()
