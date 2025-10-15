import os
import json
import time
import shutil
import threading
from typing import Dict, Any
from fastapi import FastAPI, HTTPException, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from pydantic import BaseModel
from transformers import pipeline, Pipeline
import requests

# Config
MEMORY_URL = os.getenv("MEMORY_URL", "http://localhost:8000")  # storage server base URL
DATA_DIR = "bots"  # local bot configs & caches
os.makedirs(DATA_DIR, exist_ok=True)

app = FastAPI(title="Chatbot Builder & Deployer")

# Bot model: stores config + runtime pipeline
class BotConfig(BaseModel):
    name: str
    persona: str = "You are a helpful assistant."
    max_length: int = 128

class BotInstance:
    def __init__(self, cfg: BotConfig):
        self.cfg = cfg
        self.lock = threading.Lock()
        self.pipeline: Pipeline | None = None
        self.history = []  # list of dicts {"role": "user"|"bot", "text": "..."}
        self._load_model()

    def _load_model(self):
        # load a small generator model; keep single pipeline per bot
        try:
            self.pipeline = pipeline("text-generation", model="distilgpt2")
        except Exception as e:
            print("Model load error:", e)
            self.pipeline = None

    def chat(self, prompt: str):
        with self.lock:
            self.history.append({"role": "user", "text": prompt})
            if not self.pipeline:
                # fallback text if pipeline failed
                reply = "Model not available."
            else:
                full_prompt = f"{self.cfg.persona}\nUser: {prompt}\nAssistant:"
                out = self.pipeline(full_prompt, max_length=self.cfg.max_length, num_return_sequences=1)
                reply = out[0]["generated_text"].strip()
            self.history.append({"role": "bot", "text": reply})
            return reply

    def get_history(self):
        return list(self.history)

# Global manager
bots: Dict[str, BotInstance] = {}
bots_lock = threading.Lock()

# Utilities: persist config to memory server (via /ml/save) and to local file
def save_bot_to_memory(cfg: BotConfig):
    """Uploads config JSON to memory server as {name}.bot.json via /ml/save"""
    try:
        filename = f"{cfg.name}.bot.json"
        tmp_path = os.path.join(DATA_DIR, filename)
        with open(tmp_path, "w") as f:
            json.dump(cfg.dict(), f, indent=2)
        files = {'file': (filename, open(tmp_path, 'rb'))}
        data = {'name': filename}
        r = requests.post(f"{MEMORY_URL}/ml/save", data={'name': filename}, files=files, timeout=10)
        return r.ok, r.text
    except Exception as e:
        return False, str(e)

def load_bot_from_memory(name: str):
    """Downloads {name}.bot.json from memory server via /ml/load/{name}.bot.json"""
    filename = f"{name}.bot.json"
    try:
        r = requests.get(f"{MEMORY_URL}/ml/load/{filename}", timeout=10)
        if r.status_code != 200:
            return None, f"memory returned {r.status_code}"
        cfg = json.loads(r.content.decode())
        return cfg, None
    except Exception as e:
        return None, str(e)

def persist_local_history(name: str, history: list):
    path = os.path.join(DATA_DIR, f"{name}.history.json")
    with open(path, "w") as f:
        json.dump(history, f, indent=2)

def load_local_history(name: str):
    path = os.path.join(DATA_DIR, f"{name}.history.json")
    if os.path.exists(path):
        return json.load(open(path, "r"))
    return []

# API models
class CreateReq(BaseModel):
    name: str
    persona: str = "You are a helpful assistant."
    max_length: int = 128

@app.post("/create_bot")
def create_bot(req: CreateReq):
    name = req.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="name required")
    with bots_lock:
        if name in bots:
            raise HTTPException(status_code=400, detail="bot already exists")
        cfg = BotConfig(name=name, persona=req.persona, max_length=req.max_length)
        inst = BotInstance(cfg)
        inst.history = load_local_history(name)
        bots[name] = inst

    # save config to memory
    ok, resp = save_bot_to_memory(cfg)
    return {"created": name, "saved_to_memory": ok, "memory_resp": resp}

@app.post("/deploy_bot")
def deploy_bot(name: str = Form(...)):
    with bots_lock:
        if name in bots:
            return {"status":"already_deployed", "name": name}
        # try load from memory first
        cfg_data, err = load_bot_from_memory(name)
        if cfg_data:
            cfg = BotConfig(**cfg_data)
        else:
            raise HTTPException(status_code=404, detail=f"bot config not found in memory: {err}")
        inst = BotInstance(cfg)
        inst.history = load_local_history(name)
        bots[name] = inst
    return {"deployed": name}

@app.get("/bots")
def list_bots():
    with bots_lock:
        return {"bots": list(bots.keys())}

@app.post("/bot/{name}/chat")
def bot_chat(name: str, body: Dict[str,Any]):
    msg = body.get("message") if isinstance(body, dict) else None
    if not msg:
        raise HTTPException(status_code=400, detail="message required")
    with bots_lock:
        inst = bots.get(name)
        if not inst:
            raise HTTPException(status_code=404, detail="bot not deployed")
    reply = inst.chat(msg)
    # persist history locally and optionally push to memory server as {name}.history.json via /ml/save
    persist_local_history(name, inst.get_history())
    # push to memory: use /ml/save to store history file (non-blocking)
    try:
        hist_file = os.path.join(DATA_DIR, f"{name}.history.json")
        with open(hist_file, "w") as f:
            json.dump(inst.get_history(), f)
        files = {'file': (f"{name}.history.json", open(hist_file, 'rb'))}
        # call memory server ml/save with form name
        requests.post(f"{MEMORY_URL}/ml/save", data={'name': f"{name}.history.json"}, files=files, timeout=5)
    except Exception:
        pass
    return {"reply": reply}

@app.get("/bot/{name}/history")
def bot_history(name: str):
    with bots_lock:
        inst = bots.get(name)
        if not inst:
            raise HTTPException(status_code=404, detail="bot not deployed")
        return {"history": inst.get_history()}

# Simple UI for building & chatting
@app.get("/", response_class=HTMLResponse)
def ui_root():
    with open("editor_index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.get("/health")
def health():
    return {"status":"ok","bots": list(bots.keys()), "memory": MEMORY_URL}
