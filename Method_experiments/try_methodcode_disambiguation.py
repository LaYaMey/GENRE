import json


# 1. Load your candidate entities JSON, assumed format: list of strings or dict keys
with open("genre_input.json", "r") as f:
    genre_input = json.load(f)

candidate_entities = genre_input["candidate_entities"]
missing_codes = genre_input["missing_codes"]

from transformers import BartTokenizer
from genre.fairseq_model import GENRE
from genre.trie import Trie

# Load tokenizer and model
tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")
model = GENRE.from_pretrained("models/fairseq_entity_disambiguation_aidayago").eval()

# Tokenize them into ID sequences
tokenized_entities = [
    tokenizer.encode(entity, add_special_tokens=False)
    for entity in candidate_entities
]


# 2. Build trie from candidate entity strings
trie = Trie(sequences=tokenized_entities)

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