import json
import sys
sys.path.append("/home/lars/fairseq")
from genre.fairseq_model import GENRE
from genre.trie import Trie
import pickle
import os

model_path = "/home/lars/GENRE/models/fairseq_entity_disambiguation_aidayago"
trie_file_path = "/home/lars/GENRE/data/candidate_methods_trie.pkl"
json_path = "/home/lars/GENRE/Method_experiments/genre_input.json"

# 1. Load your candidate entities JSON
with open(json_path, "r") as f:
    genre_input = json.load(f)
missing_codes = genre_input["missing_codes"]

model = GENRE.from_pretrained(model_path).eval()

try:
    with open(trie_file_path, "rb") as f:
        trie = Trie.load_from_dict(pickle.load(f))
    print(f"Trie successfully loaded from: {os.path.abspath(trie_file_path)}")
except FileNotFoundError:
    print(f"Trie File not found, creating Trie. Trie File expected at: {os.path.abspath(trie_file_path)}")
    # Tokenize them into ID sequences
    candidate_entities = genre_input["candidate_entities"]
    trie = Trie([
            [2] + model.encode(e).tolist()[1:]
            for e in candidate_entities
        ])
    # Export GENRE trie
    with open(trie_file_path, "wb") as f:
        pickle.dump(trie.trie_dict, f)

result_mappings = {}

# Run GENRE on each sentence
for missing_code in missing_codes:

    sentence = (
        f'[START_ENT] {missing_code} [END_ENT] is a method code describing a technique for '
        f'geochemical element extraction from rock samples in the GEOROC database.'
    )
    
    results = model.sample(
        [sentence],
        prefix_allowed_tokens_fn=lambda batch_id, sent: trie.get(sent.tolist())
    )
    print(f"\n{missing_code}")
    for r in results[0][:3]:  # top 3 candidates
        print(f"  → {r['text']} (score: {r['score'].item():.4f})")
    result_mappings[missing_code] = results[0][0]["text"]

with open("GENRE_MissingMethodcodeMapping.json", "w", encoding="utf-8") as f:
    json.dump(result_mappings, f, ensure_ascii=False, indent=4)