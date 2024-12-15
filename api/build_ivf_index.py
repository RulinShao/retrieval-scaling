import os
import argparse
from src.indicies.ivf_flat import IVFFlatIndexer



def build_ivf_flat_index(domain, num_shards, shard_id=None):
    sample_train_size = 6000000
    projection_size = 768
    ncentroids = 4096
    probe = 4096
    
    embed_dir = f'/checkpoint/amaia/explore/comem/data/scaling_out/embeddings/facebook/contriever-msmarco/{domain}/{num_shards}-shards'
    if shard_id is not None:
        embed_paths = [os.path.join(embed_dir, f'passages_{shard_id:02}.pkl')]
        index_dir = f'/checkpoint/amaia/explore/comem/data/scaling_out/embeddings/facebook/contriever-msmarco/{domain}/{num_shards}-shards/index_ivf_flat_ip_fixed_{shard_id}/'
        passage_dir = f'/checkpoint/amaia/explore/comem/data/massive_ds_1.4t/scaling_out/passages/{domain}/{num_shards}-shards/raw_passages-{shard_id}-of-{num_shards}.pkl'
    else:
        embed_paths = [os.path.join(embed_dir, filename) for filename in os.listdir(embed_dir) if filename.endswith('.pkl')]
        index_dir = f'/checkpoint/amaia/explore/comem/data/scaling_out/embeddings/facebook/contriever-msmarco/{domain}/{num_shards}-shards/index_ivf_flat_ip_fixed/'
        passage_dir = f'/checkpoint/amaia/explore/comem/data/massive_ds_1.4t/scaling_out/passages/{domain}/{num_shards}-shards'
    os.makedirs(index_dir, exist_ok=True)
    formatted_index_name = f"index_ivf_flat_ip.{sample_train_size}.{projection_size}.{ncentroids}.faiss"
    index_path = os.path.join(index_dir, formatted_index_name)
    meta_file = os.path.join(index_dir, formatted_index_name+'.meta')
    trained_index_path = os.path.join(index_dir, formatted_index_name+'.trained')
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


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", type=str, default='rpj_c4', help="Domain id of the datastore")
    parser.add_argument("--num_shards", type=int, default=32, help="Number of shards")
    parser.add_argument("--shard_id", type=int, default=0, help="Shard id that you want to build an index with")
    args = parser.parse_args()
    
    num_shards = args.num_shards
    domain = args.domain
    shard_id = args.shard_id
    build_ivf_flat_index(domain, num_shards, shard_id)