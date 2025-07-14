import os
import pickle

from genre.fairseq_model import GENRE
from genre.trie import Trie

_model = None  # global cache

def set_model(model_path):
    """Load model once at server startup."""
    global _model
    _model = GENRE.from_pretrained(model_path).eval()

def get_model():
    """Retrieve the globally cached model."""
    if _model is None:
        raise RuntimeError("Model has not been set. Call set_model() first.")
    return _model

def _get_trie_path(database_name: str, table_name: str, column_name: str = "") -> str:
    """Build a consistent trie filename based on DB, table, and column name."""
    parts = []

    if database_name:
        parts.append(database_name.strip().replace(" ", "_").lower())
    if table_name:
        parts.append(table_name.strip().replace(" ", "_").lower())
    if column_name:
        parts.append(column_name.strip().replace(" ", "_").lower())

    filename = "trie_" + "__".join(parts) + ".pkl"
    return os.path.join(os.path.dirname(__file__), "..", "data", filename)

def link_entities(candidate_entities, data_entries, configure_entity_tags=True, database_name="", table_name="", column_name=""):
    model = get_model()
    trie = None

    if database_name and table_name:
        trie_path = _get_trie_path(database_name, table_name, column_name)
        # Try to load existing trie
        if os.path.isfile(trie_path):
            try:
                with open(trie_path, "rb") as f:
                    trie = pickle.load(f)
            except Exception as e:
                print(f"⚠️ Failed to load trie from {trie_path}: {e}")

    # If no trie loaded (or no database_name given), create a new one
    if trie is None:
        trie = Trie([
            [2] + model.encode(e).tolist()[1:]
            for e in candidate_entities
        ])
        # Save the trie if a database name was provided
        if database_name and table_name:
            os.makedirs(os.path.dirname(trie_path), exist_ok=True)
            with open(trie_path, "wb") as f:
                pickle.dump(trie, f)
        else:
            print("No unique combination of database and table given, trie was not cached.")

    top_results = []

    for data_entry in data_entries:
        if configure_entity_tags:
            sentence = f'[START_ENT] {data_entry} [END_ENT]'
        else:
            sentence = data_entry

        results = model.sample(
            [sentence],
            prefix_allowed_tokens_fn=lambda batch_id, sent: trie.get(sent.tolist())
        )
        top_results.append(results[0][0]['text'])

    return top_results
