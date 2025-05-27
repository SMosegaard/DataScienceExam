import os
import shutil
from huggingface_hub import hf_hub_download, list_repo_files
import argparse

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--token",
                        "-t",
                        required = True,
                        help = "Provide Hugging Face token to access the dataset") 
    args = parser.parse_args()
    return args

def main():
    
    args = parser()
    token = args.token

    repo = "SMosegaard/DataScienceExam_Benchmark_Data"
    repo_type = "dataset"
    destination = {"train/": "data/train/",
                   "test/": "data/test/",}
    root_files = ["DMI_data.csv",
                  "Electricity_Maps_data.csv"]

    files = list_repo_files(repo_id = repo, repo_type = repo_type, token = token)

    for file in files:
        for prefix, folder in destination.items():
            if file.startswith(prefix):
                local_downloaded_path = hf_hub_download(repo_id = repo, filename = file, repo_type = repo_type, token = token)
                file_name = os.path.basename(file)
                shutil.copy(local_downloaded_path, os.path.join(folder, file_name))
        
        for file in root_files:
            local_downloaded_path = hf_hub_download(repo_id = repo, filename = file, repo_type = repo_type, token = token)
            shutil.copy(local_downloaded_path, os.path.join("data", file))

if __name__ == "__main__":
    main()