import os
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from pydantic import BaseModel
from typing import List, Optional, Dict

from genre.entity_linking_tool import link_entities, set_model
from genre.download_model import download_and_extract_model

MODEL_NAME = "fairseq_entity_disambiguation_aidayago"  # Or from env var
MODEL_DIR = f"/GENRE/models"
MODEL_PATH = os.path.join(MODEL_DIR,MODEL_NAME)


class LinkRequest(BaseModel):
    table_rows: List[Dict]
    data_entries: List[Dict]

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup code: ensure model is available and initialized
    if not os.path.isdir(MODEL_PATH):
        print(f"Model not found in {MODEL_PATH}, downloading...", flush=True)
        download_and_extract_model(MODEL_NAME, MODEL_DIR)
    else:
        print(f"Model found in {MODEL_PATH}", flush=True)

    #print(f"Running set_model on {MODEL_PATH}", flush=True)
    set_model(MODEL_PATH)  # Initialize the model/trie globally

    yield
        
    # Shutdown logic (runs once when app stops)
    print("SHUTDOWN")

app = FastAPI(lifespan=lifespan)


@app.post("/link_entities")
async def link_entities_endpoint(req: LinkRequest):
    try:
        # Use the globally loaded model/trie, no need to pass model_path anymore
        results = link_entities(
            table_rows=req.table_rows,
            data_entries=req.data_entries,
        )
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
