"""
rag_explainer.py

Local retrieval-augmented reference layer for the "Learn more" explainer.

Reference passages (.txt) live in a references/ folder. They are embedded locally (sentence-transformers) and indexed (FAISS), the most
relevant passages for a finding are retrieved and the local LLM explains grounded in them.

Safety is enforced primarily by the system prompt (describe-only). A light final guard removes tumor-type names and blocks only the most explicit clinical
recommendations."""

import os
import glob
import re
import copy
import json

_MODEL = None
_INDEX = None
_CHUNKS = None
_EMBED_DIM = None


# loading references
def _read_reference_files(ref_dir):
    chunks = []
    for path in sorted(glob.glob(os.path.join(ref_dir, "*.txt"))):
        with open(path, encoding="utf-8") as f:
            raw = f.read().strip()
        source = "unknown source"
        body = raw
        if raw.lower().startswith("source:"):
            first, _, rest = raw.partition("\n")
            source = first.split(":", 1)[1].strip()
            body = rest.strip()
        for para in re.split(r"\n\s*\n", body):
            para = para.strip()
            if len(para) >= 40:
                chunks.append({"text": para, "source": source,
                               "file": os.path.basename(path)})
    return chunks


# index build
def build_index(ref_dir):
    global _MODEL, _INDEX, _CHUNKS, _EMBED_DIM
    _CHUNKS = _read_reference_files(ref_dir)
    if not _CHUNKS:
        return False
    try:
        from sentence_transformers import SentenceTransformer
        import faiss
        import numpy as np
        _MODEL = SentenceTransformer("all-MiniLM-L6-v2")
        embs = _MODEL.encode([c["text"] for c in _CHUNKS],
                             convert_to_numpy=True, normalize_embeddings=True)
        _EMBED_DIM = embs.shape[1]
        _INDEX = faiss.IndexFlatIP(_EMBED_DIM)
        _INDEX.add(embs.astype("float32"))
        return True
    except Exception as e:
        print(f"[rag] embeddings/FAISS unavailable ({e}); using keyword retrieval.")
        _MODEL = None; _INDEX = None
        return False


# retrieval
def retrieve(query, k=3):
    if _CHUNKS is None:
        return []
    if _MODEL is not None and _INDEX is not None:
        import numpy as np
        q = _MODEL.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        scores, idx = _INDEX.search(q.astype("float32"), min(k, len(_CHUNKS)))
        return [_CHUNKS[i] for i in idx[0] if i >= 0]
    qwords = set(re.findall(r"[a-z]{4,}", query.lower()))
    scored = []
    for c in _CHUNKS:
        cwords = set(re.findall(r"[a-z]{4,}", c["text"].lower()))
        overlap = len(qwords & cwords)
        if overlap:
            scored.append((overlap, c))
    scored.sort(key=lambda x: -x[0])
    return [c for _, c in scored[:k]]


# prompt
RAG_SYSTEM_PROMPT = """You are an educational assistant that DEFINES brain-tumor MRI imaging terms for a
clinician. You are given the patient's measured values and RETRIEVED reference passages that define the
imaging concepts.

You describe ONLY what the imaging measurement IS and what the tissue LOOKS LIKE on MRI. You describe a
STATIC IMAGE. You must NEVER say what the tumor is DOING, what it MEANS clinically, or what will happen.
You ADD DEFINITIONS OR additional theoretical relevant information found in specific references. Cite the source: (SOURCE-it is found 
in the first line of each reference).

You must NEVER output: grade, type, aggressiveness; prognosis, survival, outcome, risk; treatment,
therapy, surgery; any word implying activity or change over time (growing, growth, spreading,
progressing, active, invasive, increased blood flow); any inference clause ("this suggests...",
"indicating that...", "consistent with..."); any clinical fact not in the retrieved passages.

When giving a definition, use the wording of the retrieved passage closely. Do NOT add your own
interpretation, elaboration, or clauses about what a finding "may indicate" or represent. State
the passage's definition and stop.

Describe the picture, define the term, cite the
source. End with the disclaimer.
"""
DISCLAIMER = ("Educational explanation of imaging findings only. Not a diagnosis, treatment recommendation or prediction of outcome.")


# light final guard
_TYPE_NAMES = ["glioblastoma", "astrocytoma", "oligodendroglioma", "gbm"]

def _light_guard(text):
    out = text
    for t in _TYPE_NAMES:
        out = re.sub(r"\b" + re.escape(t) + r"\b", "the tumor", out, flags=re.IGNORECASE)

    # remove forbidden clauses
    forbidden_clauses = [
        r",?\s*(which |this )?(may |can )?(suggest|indicat|reflect|impl|represent)\w*[^.;:]*"
        r"(grow|growth|aggressi|invasi|spread|progress|active|angiogen|malignan|inflam)\w*[^.;:]*",
        r",?\s*(due to|caused by|because of|resulting from)[^.;:]*"
        r"(blood flow|vascular|perfusion|proliferat|angiogen)\w*[^.;:]*",
        r",?\s*(actively |rapidly )?(growing|growth)[^.;:]*",
    ]
    for pat in forbidden_clauses:
        out = re.sub(pat, "", out, flags=re.IGNORECASE)

    # clean up connectors
    out = re.sub(r"\b(due to|caused by|because of|which may|which can|resulting from)\s*[,.:;]",
                 ".", out, flags=re.IGNORECASE)
    out = re.sub(r"\s*,\s*\.", ".", out)         
    out = re.sub(r"\s+([,.;:])", r"\1", out)       
    out = re.sub(r"\.\s*\.", ".", out)            
    out = re.sub(r"\s{2,}", " ", out)

    if "not a diagnosis" not in out.lower():
        out = out.rstrip() + "\n\n" + DISCLAIMER
    return out.strip()

# explainer entry point
def explain_with_rag(finding_query, findings, generate_fn, k=3):
    """Retrieve relevant passages and have the LLM explain, grounded in them. Safety relies on the system prompt: a light guard tidies tumor-type names and
    ensures the disclaimer is present."""
    hits = retrieve(finding_query, k=k)
    if not hits:
        raise RuntimeError("no reference passages retrieved")
    refs = "\n\n".join(f"[{h['source']}] {h['text']}" for h in hits)
    user = (f"Finding to explain: {finding_query}\n\n"
            f"Patient findings:\n{json.dumps(findings, indent=2)}\n\n"
            f"Retrieved references (use ONLY these):\n{refs}\n\n"
            f"End with this disclaimer exactly:\n{DISCLAIMER}")
    raw = generate_fn(RAG_SYSTEM_PROMPT, user)
    return _light_guard(raw)

def _strip_identifiers(findings):
    """Remove patient identifiers before anything is sent to the model."""
    f = copy.deepcopy(findings)
    f.pop("patient_id", None)
    f.pop("provenance", None)  
    return f


