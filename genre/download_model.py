import os
import urllib.request
import tarfile

def download_and_extract_model(model_name: str, models_dir: str = "/models"):
    """
    Downloads and extracts a GENRE model tar.gz into the models directory if not already present.

    Args:
        model_name: e.g. 'fairseq_entity_disambiguation_aidayago'
        models_dir: directory to store models (mounted volume)
    """
    model_path = os.path.join(models_dir, model_name)
    if os.path.isdir(model_path):
        print(f"Model '{model_name}' already exists at {model_path}. Skipping download.")
        return

    # Map model_name to download URL based on your bash script URLs
    url_map = {
        "fairseq_entity_disambiguation_aidayago": "http://dl.fbaipublicfiles.com/GENRE/fairseq_entity_disambiguation_aidayago.tar.gz",
        "hf_entity_disambiguation_aidayago": "http://dl.fbaipublicfiles.com/GENRE/hf_entity_disambiguation_aidayago.tar.gz",
        "fairseq_entity_disambiguation_blink": "http://dl.fbaipublicfiles.com/GENRE/fairseq_entity_disambiguation_blink.tar.gz",
        "hf_entity_disambiguation_blink": "http://dl.fbaipublicfiles.com/GENRE/hf_entity_disambiguation_blink.tar.gz",
        "fairseq_e2e_entity_linking_aidayago": "http://dl.fbaipublicfiles.com/GENRE/fairseq_e2e_entity_linking_aidayago.tar.gz",
        "hf_e2e_entity_linking_aidayago": "http://dl.fbaipublicfiles.com/GENRE/hf_e2e_entity_linking_aidayago.tar.gz",
        "fairseq_e2e_entity_linking_wiki_abs": "http://dl.fbaipublicfiles.com/GENRE/fairseq_e2e_entity_linking_wiki_abs.tar.gz",
        "hf_e2e_entity_linking_wiki_abs": "http://dl.fbaipublicfiles.com/GENRE/hf_e2e_entity_linking_wiki_abs.tar.gz",
        "fairseq_wikipage_retrieval": "http://dl.fbaipublicfiles.com/GENRE/fairseq_wikipage_retrieval.tar.gz",
        "hf_wikipage_retrieval": "http://dl.fbaipublicfiles.com/GENRE/hf_wikipage_retrieval.tar.gz",
    }

    if model_name not in url_map:
        raise ValueError(f"Model name '{model_name}' not recognized or no download URL available.")

    url = url_map[model_name]
    os.makedirs(models_dir, exist_ok=True)
    tar_path = os.path.join(models_dir, f"{model_name}.tar.gz")

    print(f"Downloading model '{model_name}' from {url} ...")
    urllib.request.urlretrieve(url, tar_path)
    print(f"Download complete. Extracting {tar_path} ...")

    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=models_dir)

    print(f"Extraction complete. Removing archive {tar_path} ...")
    os.remove(tar_path)
    print(f"Model '{model_name}' is ready at {model_path}.")
