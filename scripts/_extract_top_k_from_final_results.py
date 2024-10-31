import json
import pdb

result_path = '/mnt/md-256k/scaling_out/retrieved_results/post_processed/full_subsampled_1_1000_dedup_merged_all_cot_query_top1000.jsonl'
data = []
with open(result_path, 'r') as fin:
    for line in fin:
        ex = json.loads(line)
        ex['ctxs'] = ex['ctxs'][:5]
        data.append(ex)

print(len(data))
with open(result_path.replace('.jsonl', '_top5.jsonl'), 'w') as fout:
    for ex in data:
        fout.write(json.dumps(ex)+'\n')
        