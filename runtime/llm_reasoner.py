# runtime/llm_reasoner.py
import os
import json
from typing import List, Dict

import requests

import constants

def _format_context(kb_hits: List[Dict], sb_hits: List[Dict]) -> str:
    """Create a compact context string combining top KB and SB hits."""
    parts = []
    if kb_hits:
        parts.append("Top KB articles (title + snippet):")
        for i, h in enumerate(kb_hits, 1):
            # title = h["metadata"].get("title") or h["metadata"].get("source", "kb")
            snippet = (h["document"][:800] + "...") if len(h["document"]) > 800 else h["document"]
            parts.append(f"[KB {i}] \n{snippet}\n")
    if sb_hits:
        parts.append("Relevant log chunks from supportbundle:")
        for i, h in enumerate(sb_hits, 1):
            # src = h["metadata"].get("source", "log")
            snippet = (h["document"][:1000] + "...") if len(h["document"]) > 1000 else h["document"]
            parts.append(f"[LOG {i}] \n{snippet}\n")
    return "\n".join(parts)

def build_prompt(question: str, kb_hits: List[Dict], sb_hits: List[Dict]) -> str:
    ctx = _format_context(kb_hits, sb_hits)
    prompt = f"""
You are an expert Longhorn/Rancher support engineer.

User question:
{question}

Context (KB + logs):
{ctx}

Task:
1) Provide a short diagnosis (one-line).
2) List probable root causes (4-6 bullets).
3) Provide exact troubleshooting steps in order.
4) List which log lines or KB articles support the diagnosis (cite KB titles or LOG i numbers).
5) Provide a confidence score (0-100).

Be concise and technical.
"""
    return prompt

def call_llm(prompt: str, model: str = None, num_predict: int = 800) -> str:
    model = model or constants.GENERATIVE_MODEL
    payload = {
        "model": model,
        "prompt": prompt,
        "num_predict": num_predict,
        "stream": False,
    }

    response = requests.post(constants.OLLAMA_URL, json=payload)
    response.raise_for_status()

    output = response.json().get("response", "")
    return output

def reason(question: str, kb_hits: List[Dict], sb_hits: List[Dict]) -> Dict:
    prompt = build_prompt(question, kb_hits, sb_hits)
    resp_text = call_llm(prompt)
    # Best-effort parse: return raw text + JSON wrapper
    return {"question": question, "prompt": prompt, "answer": resp_text}
