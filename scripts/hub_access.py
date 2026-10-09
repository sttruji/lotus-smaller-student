"""Check the Colab token and model-file access without downloading model weights."""
import argparse
import json
import os
from pathlib import Path


DEFAULT_REPOSITORIES = (
    "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
    "yingfanbot/gsm-cot-llama1b",
    "yingfanbot/gsm-cot-llama3b",
)


def check_hub_access(repositories):
    from huggingface_hub import HfApi, hf_hub_download
    from huggingface_hub.utils import GatedRepoError, HfHubHTTPError

    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        raise RuntimeError("HF_TOKEN is missing. Set the Colab Secret and rerun cell 3.")
    api = HfApi(token=token)
    try:
        account = api.whoami()
    except HfHubHTTPError as error:
        status = getattr(error.response, "status_code", None)
        if status in (401, 403):
            raise RuntimeError("HF_TOKEN was rejected. Update the Colab Secret with a valid read token, "
                               "grant notebook access, and rerun cell 3. Never paste the token into output or chat.") from None
        raise RuntimeError(f"Could not verify the Hugging Face account (HTTP {status}); check network/service availability.") from None
    print(f"Authenticated Hugging Face account: {account['name']}", flush=True)
    for repo_id in dict.fromkeys(repositories):
        try:
            api.auth_check(repo_id, repo_type="model")
            # model_info alone can succeed for a gated model whose files are denied.
            # Force a fresh, small config download so a cache hit cannot mask denial.
            hf_hub_download(repo_id, "config.json", token=token, force_download=True)
        except (HfHubHTTPError, ValueError) as error:
            # Hub 0.26.5 may wrap a failed forced download in ValueError.
            http_error = error.__cause__ if isinstance(error, ValueError) else error
            if not isinstance(http_error, HfHubHTTPError):
                raise
            if isinstance(http_error, GatedRepoError):
                raise RuntimeError(
                    f"Account {account['name']} / HF_TOKEN cannot download {repo_id}. "
                    f"Open https://huggingface.co/{repo_id} while logged into that same account, "
                    "request/accept access and wait for approval. A fine-grained token must also permit reading "
                    "the gated repository. After fixing access or updating the Colab HF_TOKEN Secret, "
                    "rerun cell 3, then cell 5. Training has not been launched in this attempt."
                ) from None
            status = getattr(http_error.response, "status_code", None)
            raise RuntimeError(
                f"Model-file access failed for {repo_id} (HTTP {status}). Check repository permissions, "
                "the token's read scope, and network/service availability; rerun cell 3 after fixing it."
            ) from None
        print(f"Model-file access OK: {repo_id}", flush=True)
    return account["name"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path)
    args = parser.parse_args()
    repositories = DEFAULT_REPOSITORIES
    if args.plan is not None:
        config = json.loads(args.plan.read_text())["training_config"]
        repositories = [config[key] for key in
                        ("model_id", "teacher_tokenizer_id", "teacher_model_path", "load_model_path")
                        if config.get(key) and not config[key].startswith(("/", ".", "~"))]
    try:
        check_hub_access(repositories)
    except RuntimeError as error:
        parser.exit(status=1, message=f"{error}\n")


if __name__ == "__main__":
    main()
