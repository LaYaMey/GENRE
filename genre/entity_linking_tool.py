import re
from typing import List, Dict

from genre.fairseq_model import GENRE
from genre.trie import Trie
from genre.entity_linking import get_end_to_end_prefix_allowed_tokens_fn_fairseq as get_prefix_allowed_tokens_fn

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

NULL_WORDS = ['', 'none', 'None', 'NONE', 'null', 'NULL', 'Null', 'Unknown', 'unknown', None]

def data2text(dictionary: Dict, shared_keys: List[str]) -> str:
    """Convert a dictionary to text format using shared keys."""
    parts = []
    for key in shared_keys:
        value = dictionary[key]
        if value not in NULL_WORDS:
            parts.append(f"{key}: {value}")
    return ', '.join(parts) + '.'

def get_shared_keys(data_entries: List[Dict], table_rows: List[Dict]) -> List[str]:
    """
    Find keys that are present and non-empty in both data_entries and table_rows.
    
    Args:
        data_entries: List of input dictionaries
        table_rows: List of candidate dictionaries from database
    
    Returns:
        List of key names that are shared across all dictionaries
    """
    def check_key(dict_list, key):
        """Check if a key exists and is non-empty in all dictionaries."""
        for d in dict_list:
            if key not in d or d[key] == '':
                return False
        return True
    
    # Get all keys from first entry of each list
    if not data_entries or not table_rows:
        return []
    
    data_keys = set(data_entries[0].keys())
    db_keys = set(table_rows[0].keys())
    
    # Find intersection of keys
    shared_keys = data_keys.intersection(db_keys)
    
    # Filter to only include keys that are present and non-empty in all dictionaries
    target_keys = [key for key in shared_keys 
                   if check_key(data_entries, key) and check_key(table_rows, key)]
    
    return target_keys

def _extract_bracketed_info_trimmed(sentence: str) -> List[str]:
    """Extract text inside [ ] brackets and trim spaces."""
    pattern = r"\[\s*(.*?)\s*\]"
    matches = re.findall(pattern, sentence)
    return matches

def link_entities(
    table_rows: List[Dict],
    data_entries: List[Dict]
) -> List[int]:
    """
    Link data entries to candidate entities using GENRE with constrained decoding.
    
    This implementation matches the behavior of Entity_Linking from text2db_ie_tools.py:
    1. Validates inputs are lists of dicts
    2. Finds shared keys between input and candidate entities
    3. Converts dicts to text using shared keys, filtering NULL words
    4. Creates candidate dictionary mapping each input to all database texts
    5. Uses constrained decoding with mention_trie and mention_to_candidates_dict
    6. Extracts bracketed text from results and returns match indices
    
    Args:
        table_rows: List of database row dictionaries (e.g., table data)
        data_entries: List of input entry dictionaries to link
    
    Returns:
        List of integers: indices into table_rows for each data_entry, or -1 if no match
    """
    model = get_model()
    
    # Input validation
    if not isinstance(table_rows, list) or not isinstance(data_entries, list):
        raise ValueError("table_rows and data_entries must be lists")
    
    if not table_rows or not data_entries:
        return [-1] * len(data_entries)
    
    # Find shared keys between data entries and candidate entities
    target_keys = get_shared_keys(data_entries, table_rows)
    
    if not target_keys:
        # No shared keys found, return -1 for all
        return [-1] * len(data_entries)
    
    # Convert dictionaries to text using shared keys
    texts = [data2text(data_entry, target_keys) for data_entry in data_entries]
    candidate_texts = [data2text(candidate, target_keys) for candidate in table_rows]
    
    # Create candidates_dict: each input text can link to any database text
    candidates_dict = {text: candidate_texts for text in texts}
    
    # Build sentences for GENRE inference
    sentences = ['Here is a piece of data. ' + text for text in texts]
    
    # Build mention_trie (format must match reference exactly)
    mention_trie = Trie([
        model.encode(" {}".format(e))[1:].tolist()
        for e in texts
    ])
    
    # Create prefix_allowed_tokens_fn with constrained decoding
    prefix_allowed_tokens_fn = get_prefix_allowed_tokens_fn(
        model,
        sentences,
        mention_trie=mention_trie,
        mention_to_candidates_dict=candidates_dict
    )
    
    # Run GENRE inference
    model_res = model.sample(
        sentences,
        prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
    )
    
    # Parse results and extract indices
    index_list = []
    for res_list in model_res:
        index = -1
        for res in res_list:
            linked_text = _extract_bracketed_info_trimmed(res['text'])
            if linked_text and linked_text[0] in candidate_texts:
                index = candidate_texts.index(linked_text[0])
                break
        index_list.append(index)
    
    return index_list
