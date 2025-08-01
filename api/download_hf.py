"""
Efficiently downloads HuggingFace models by circumventing the `.git` directory.

Speeds up the download process by approximately 2x by avoiding downloading the 
`.git` folder, which contains a copy of the models (thanks to git-lfs). This script 
can be especially beneficial for large models (e.g., Meta-Llama-3.1-405B). 

Usage:
    python download_hf.py \
        <hf_repo_id> <destination_folder> \
        --token <hf_token> --skip <skip_files> 

Example:
    python download_hf.py meta-llama/Meta-Llama-3.1-405B --skip original/

Details:
    By default, `git clone <hf_repo_id>` will download both the model files and the 
    `.git` folder from huggingface hub. git-lfs creates a copy of the model files in
    the `.git` folder that almost doubles the size of the repo. This script uses the 
    HuggingFace API to fetch model files individually and avoid downloading the `.git`
    folder. 
"""

import os
import argparse
from tqdm import tqdm
from typing import Optional
from huggingface_hub import HfApi, hf_hub_download


# Function to download all files from a Hugging Face Hub repository into a subfolder
def download_all_files_from_hub(
    repo_id: str,
    destination_folder: str,
    token: str,
    skip_files: Optional[list] = None,
):
    # Initialize HfApi to list files in the repo
    api = HfApi(token=token)

    # Ensure destination folder exists
    os.makedirs(destination_folder, exist_ok=True)

    # List all files in the repo and download them
    file_list = api.list_repo_files(repo_id=repo_id, token=token)

    for file_name in tqdm(file_list):
        if file_name in skip_files or any(
            file_name.startswith(skip) for skip in skip_files
        ):
            print(f"Skipping {file_name}")
            continue
        file_path = hf_hub_download(
            repo_id=repo_id,
            filename=file_name,
            local_dir=destination_folder,
            use_auth_token=token,
        )
        print(f"Downloaded {file_name} to {file_path}")


# Setup command-line argument parsing
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download all files from a given Hugging Face Hub model repository into a specified subfolder."
    )
    parser.add_argument(
        "repo_id", type=str, help="Repository ID on Hugging Face Hub (e.g., 'gpt2')"
    )
    parser.add_argument(
        "destination_folder",
        type=str,
        nargs="?",
        default="",
        help="Destination folder to download the files",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=os.getenv("HF_TOKEN", None),
        help="Hugging Face token for authentication",
    )
    parser.add_argument(
        "--skip",
        type=str,
        nargs="*",
        default=None,
        help="List of filenames or directories to skip",
    )

    args = parser.parse_args()

    # Ensure there's a token provided either via command line or environment variable
    token = args.token if args.token else input("Enter your Hugging Face token: ")

    # If the destination folder is not set, derive it from the repo_id
    destination_folder = (
        args.destination_folder
        if args.destination_folder
        else args.repo_id.split("/")[-1]
    )

    # Call the function with the provided command-line arguments
    download_all_files_from_hub(
        args.repo_id,
        destination_folder,
        token,
        args.skip,
    )