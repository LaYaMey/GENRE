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

def link_entities(candidate_entities, data_entries, configure_entity_tags=True):
    model = get_model()

    trie = Trie([
        [2] + model.encode(e).tolist()[1:]
        for e in candidate_entities
    ])

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
