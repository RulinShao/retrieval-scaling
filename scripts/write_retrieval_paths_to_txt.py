import os
import argparse


MMLU_DOMAINS = ['mmlu_econometrics', 'mmlu_moral_disputes', 'mmlu_marketing', 'mmlu_college_biology', 'mmlu_high_school_geography', 'mmlu_professional_medicine', 'mmlu_high_school_chemistry', 'mmlu_high_school_computer_science', 'mmlu_high_school_european_history', 'mmlu_international_law', 'mmlu_global_facts', 'mmlu_human_aging', 'mmlu_anatomy', 'mmlu_miscellaneous', 'mmlu_formal_logic', 'mmlu_conceptual_physics', 'mmlu_high_school_physics', 'mmlu_prehistory', 'mmlu_computer_security', 'mmlu_jurisprudence', 'mmlu_college_mathematics', 'mmlu_high_school_world_history', 'mmlu_business_ethics', 'mmlu_human_sexuality', 'mmlu_sociology', 'mmlu_clinical_knowledge', 'mmlu_high_school_biology', 'mmlu_college_chemistry', \
    'mmlu_medical_genetics', 'mmlu_high_school_mathematics', 'mmlu_professional_accounting', 'mmlu_high_school_government_and_politics', 'mmlu_logical_fallacies', 'mmlu_professional_law', 'mmlu_electrical_engineering', \
    'mmlu_elementary_mathematics', 'mmlu_public_relations', 'mmlu_moral_scenarios', 'mmlu_college_medicine', \
    'mmlu_high_school_microeconomics', 'mmlu_machine_learning', 'mmlu_world_religions', 'mmlu_high_school_statistics', 'mmlu_nutrition', \
    'mmlu_us_foreign_policy', 'mmlu_philosophy', 'mmlu_high_school_macroeconomics', 'mmlu_security_studies', 'mmlu_high_school_psychology', 'mmlu_high_school_us_history', 'mmlu_college_computer_science', 'mmlu_professional_psychology', 'mmlu_abstract_algebra', 'mmlu_astronomy', 'mmlu_virology', 'mmlu_college_physics', 'mmlu_management']


def check_single_shard_results_exist(domain, num_shards, eval_domain, n_docs):
    retrieved_dir = "/mnt/md-256k/scaling_out/retrieved_results/facebook/contriever-msmarco"
    for i in range(num_shards):
        if 'mmlu' == eval_domain:
            for mmlu_domain_name in MMLU_DOMAINS:
                single_shard_retrieved_path = os.path.join(retrieved_dir, f"{domain}_datastore-256_chunk_size-1of{num_shards}_shards/top_{n_docs}/{i}/{mmlu_domain_name}_retrieved_results.jsonl")
                if not os.path.exists(single_shard_retrieved_path):
                    print(f"{single_shard_retrieved_path} is missing!")
                
        else:
            single_shard_retrieved_path = os.path.join(retrieved_dir, f"{domain}_datastore-256_chunk_size-1of{num_shards}_shards/top_{n_docs}/{i}/{eval_domain}_retrieved_results.jsonl")
            if not os.path.exists(single_shard_retrieved_path):
                print(f"{single_shard_retrieved_path} is missing!")


def get_searched_paths(domain, num_shards, eval_domain, output_file, first_n_shards=8, n_docs=100, assert_no_missing=True):
    retrieved_dir = "/mnt/md-256k/scaling_out/retrieved_results/facebook/contriever-msmarco"

    all_paths = []
    no_missing = True
    if num_shards == 8:
        for i in range(8):
            if i+1==first_n_shards:
                if eval_domain == 'mmlu':
                    for mmlu_domain in MMLU_DOMAINS:
                        merged_shard_retrieved_path = os.path.join(retrieved_dir, f"{domain}_datastore-256_chunk_size-1of{num_shards}_shards/top_{n_docs}/{'-'.join([str(x) for x in range(i+1)])}/{mmlu_domain}_retrieved_results.jsonl")
                        if os.path.exists(merged_shard_retrieved_path):
                            all_paths.append(merged_shard_retrieved_path)
                        else:
                            print(f"{merged_shard_retrieved_path} is missing!")
                            no_missing = False
                else:
                    merged_shard_retrieved_path = os.path.join(retrieved_dir, f"{domain}_datastore-256_chunk_size-1of{num_shards}_shards/top_{n_docs}/{'-'.join([str(x) for x in range(i+1)])}/{eval_domain}_retrieved_results.jsonl")
                    if os.path.exists(merged_shard_retrieved_path):
                        all_paths.append(merged_shard_retrieved_path)
                    else:
                        print(f"{merged_shard_retrieved_path} is missing!")
                        no_missing = False
    elif num_shards == 32:
        for i in range(8):
            if i+1==first_n_shards:
                if eval_domain == 'mmlu':
                    for mmlu_domain in MMLU_DOMAINS:
                        merged_shard_retrieved_path = os.path.join(retrieved_dir, f"{domain}_datastore-256_chunk_size-1of{num_shards}_shards/top_{n_docs}/{'-'.join([str(4*x+j) for x in range(i+1) for j in range(4)])}/{mmlu_domain}_retrieved_results.jsonl")
                        if os.path.exists(merged_shard_retrieved_path):
                            all_paths.append(merged_shard_retrieved_path)
                        else:
                            print(f"{merged_shard_retrieved_path} is missing!")
                            no_missing = False
                else:
                    merged_shard_retrieved_path = os.path.join(retrieved_dir, f"{domain}_datastore-256_chunk_size-1of{num_shards}_shards/top_{n_docs}/{'-'.join([str(4*x+j) for x in range(i+1) for j in range(4)])}/{eval_domain}_retrieved_results.jsonl")
                    if os.path.exists(merged_shard_retrieved_path):
                        all_paths.append(merged_shard_retrieved_path)
                    else:
                        print(f"{merged_shard_retrieved_path} is missing!")
                        no_missing = False
    
    assert not assert_no_missing or no_missing
    
    with open(output_file, 'a') as fout:
        for path in all_paths:
            fout.write(path+'\n')


def get_subsampled_searched_paths(p, domain, num_shards, eval_domain, output_file):
    retrieved_dir = "/mnt/md-256k/scaling_out/retrieved_results/facebook/contriever-msmarco"

    all_paths = []
    merged_shard_retrieved_path = os.path.join(retrieved_dir, f"{domain}_datastore-256_chunk_size-1of{num_shards}_shards/top_100/subsampled_{p}/{'-'.join([str(x) for x in range(num_shards)])}/{eval_domain}_retrieved_results.jsonl")
    if os.path.exists(merged_shard_retrieved_path):
        all_paths.append(merged_shard_retrieved_path)
    else:
        print(f"{merged_shard_retrieved_path} is missing!")

    with open(output_file, 'a') as fout:
        for path in all_paths:
            fout.write(path+'\n')


domain2shards = {
    'rpj_wiki': 8,
    'rpj_stackexchange': 8,
    'rpj_book': 8,
    'rpj_arxiv': 8,
    'rpj_github': 8,
    'dpr_wiki': 8,
    "pes2o": 8, 
    "math": 8, 
    "pubmed": 8,
    "rpj_c4": 32,
    "rpj_commoncrawl_2019-30": 32,
    "rpj_commoncrawl_2020-05": 32,
    "rpj_commoncrawl_2021-04": 32, 
    "rpj_commoncrawl_2022-05": 32, 
    "rpj_commoncrawl_2023-06": 32,
}


def aggregate_mmlu_searched_results():
    import shutil

    results_dir = '/mnt/md-256k/scaling_out/retrieved_results/post_processed/mmlu'

    seeds = [100, 101, 102]
    ps = [1, 0.75, 0.5, 0.25, 0.1, 0.05, 0.01]

    for seed in seeds:
        for p in ps:
            for mmlu_domain in MMLU_DOMAINS:
                destination_dir = f'/mnt/md-256k/scaling_out/retrieved_results/post_processed/mmlu_seed_{seed}_p_{p}'
                os.makedirs(destination_dir, exist_ok=True)

                source_file = os.path.join(results_dir, f'full_subsampled_{p}_{seed}_dedup_merged_{mmlu_domain}_top1000.jsonl')
                target_file = os.path.join(destination_dir, f'{mmlu_domain}_retrieved_results.jsonl')
                if os.path.exists(source_file) or os.path.exists(target_file):
                    if os.path.exists(source_file):
                        shutil.move(source_file, target_file)
                else:
                    print(f"Missing {source_file}")

                

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_domain", type=str)
    parser.add_argument("--seed")
    parser.add_argument("--continue_if_missing", action='store_true')
    parser.add_argument("--check_single_shard", action='store_true')
    parser.add_argument("--source_dir", default=None)
    parser.add_argument("--n_docs", type=int, default=100)
    parser.add_argument("--single_ds_domain", type=str, default="")
    parser.add_argument("--aggregate_mmlu_results", action='store_true')
    args = parser.parse_args()

    if args.aggregate_mmlu_results:
        aggregate_mmlu_searched_results()
    
    elif args.eval_domain == 'mmlu' and not args.check_single_shard:
        output_dir = "/mnt/md-256k/scaling_out/retrieved_results/post_processed/mmlu"
        os.makedirs(output_dir, exist_ok=True)

        for mmlu_domain in MMLU_DOMAINS:
            output_file = os.path.join(output_dir, f'{mmlu_domain}_top{args.n_docs}_8shards.txt')
            with open(output_file, 'w') as fout:
                pass
            for domain in ["rpj_wiki", "rpj_stackexchange", "rpj_book", "rpj_arxiv", "rpj_github", "dpr_wiki", "pes2o", "math", "pubmed"]:
                get_searched_paths(domain, 8, mmlu_domain, output_file, n_docs=args.n_docs, assert_no_missing=not args.continue_if_missing)
            for domain in ["rpj_c4", "rpj_commoncrawl_2019-30", "rpj_commoncrawl_2020-05", "rpj_commoncrawl_2021-04", "rpj_commoncrawl_2022-05", "rpj_commoncrawl_2023-06"]:
                get_searched_paths(domain, 32, mmlu_domain, output_file, n_docs=args.n_docs, assert_no_missing=not args.continue_if_missing)
    

    else:

        if args.single_ds_domain:
            output_file = f"/mnt/md-256k/scaling_out/retrieved_results/post_processed/{args.eval_domain}_single_{args.single_ds_domain}.txt"
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, 'w') as fout:
                get_searched_paths(args.single_ds_domain, domain2shards[args.single_ds_domain], args.eval_domain, output_file, assert_no_missing=not args.continue_if_missing)
        else:
            output_file = f"/mnt/md-256k/scaling_out/retrieved_results/post_processed/{args.eval_domain}_top{args.n_docs}_8shards.txt"
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, 'w') as fout:
                pass
            for domain in ["rpj_wiki", "rpj_stackexchange", "rpj_book", "rpj_arxiv", "rpj_github", "dpr_wiki", "pes2o", "math", "pubmed"]:
                if args.check_single_shard:
                    check_single_shard_results_exist(domain, 8, args.eval_domain, n_docs=args.n_docs)
                else:
                    get_searched_paths(domain, 8, args.eval_domain, output_file, n_docs=args.n_docs, assert_no_missing=not args.continue_if_missing)
            for domain in ["rpj_c4", "rpj_commoncrawl_2019-30", "rpj_commoncrawl_2020-05", "rpj_commoncrawl_2021-04", "rpj_commoncrawl_2022-05", "rpj_commoncrawl_2023-06"]:
                if args.check_single_shard:
                    check_single_shard_results_exist(domain, 32, args.eval_domain, n_docs=args.n_docs)
                else:
                    get_searched_paths(domain, 32, args.eval_domain, output_file, n_docs=args.n_docs, assert_no_missing=not args.continue_if_missing)
        
        print(f"All files exist, saved to {output_file}")