import torch
import pickle
from genre.fairseq_model import GENRE
from genre.trie import Trie

# Load Wikipedia titles trie
with open("data/kilt_titles_trie_dict.pkl", "rb") as f:
    trie = Trie.load_from_dict(pickle.load(f))

# Load model
model = GENRE.from_pretrained("models/fairseq_entity_disambiguation_aidayago").eval()

# Example cryptic methodcodes in context
codes = [
    "The method [START_ENT] ICPMS_TD [END_ENT] is widely used in elemental analysis.",
    "We used [START_ENT] FE_EMP [END_ENT] for microscopic analysis.",
    "Dating was done using [START_ENT] PB207_PB206_AGE [END_ENT].",
    "We measured radiation using [START_ENT] GAMMA [END_ENT].",
    "Isotope analysis used [START_ENT] WBD [END_ENT].",
    "Analysis performed via [START_ENT] ICPOES [END_ENT].",
    "Determined age using [START_ENT] NE21 AGE [END_ENT]."
]

# Run GENRE on each sentence
for sentence in codes:
    results = model.sample(
        [sentence],
        prefix_allowed_tokens_fn=lambda batch_id, sent: trie.get(sent.tolist())
    )
    print(f"\n{sentence}")
    for r in results[0][:3]:  # top 3 candidates
        print(f"  → {r['text']} (score: {r['score'].item():.4f})")
