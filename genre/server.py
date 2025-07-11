import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional

from entity_linking_tool import link_entities, set_model
from download_model import download_and_extract_model

app = FastAPI()

MODEL_NAME = "fairseq_entity_disambiguation_aidayago"  # Or from env var
MODEL_DIR = f"/models/{MODEL_NAME}"

class LinkRequest(BaseModel):
    candidate_entities: List[str]
    data_entries: List[str]
    use_entity_tags: Optional[bool] = True

@app.on_event("startup")
async def startup_event():
    # Check if model directory exists, if not download it once
    if not os.path.isdir(MODEL_DIR):
        print(f"Model not found in {MODEL_DIR}, downloading...")
        download_and_extract_model(MODEL_NAME, MODEL_DIR) 
    else:
        print(f"Model found in {MODEL_DIR}")

    # Load the GENRE model and build trie once
    set_model(MODEL_DIR)  # Your function to init global model/trie

@app.post("/link_entities")
async def link_entities_endpoint(req: LinkRequest):
    try:
        # Use the globally loaded model/trie, no need to pass model_path anymore
        results = link_entities(
            candidate_entities=req.candidate_entities,
            data_entries=req.data_entries,
            use_entity_tags=req.use_entity_tags
        )
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
