from huggingface_hub import HfApi
import os

api = HfApi(token=os.getenv("HF_TOKEN"))
repo_type="space"
repo_id="ganeshdattatreyan/tourism-package-prediction"
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Space '{repo_id}' already exists. Using it.")
except Exception as e:
    if "RepositoryNotFoundError" in str(e):
        print(f"Space '{repo_id}' not found. Creating new space...")
        create_repo(repo_id=repo_id, repo_type=repo_type, space_sdk="docker", private=False)
        print(f"Space '{repo_id}' created.")
    else:
        print(f"Error creating repository: {e}")

api.upload_folder(
    folder_path="tourism_project/deployment",     # the local folder containing your files
    # replace with your repoid
    repo_id=repo_id,          # the target repo
    repo_type=repo_type,                      # dataset, model, or space
)
