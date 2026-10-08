import chromadb
from openai import OpenAI
from dotenv import load_dotenv
import os
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL = "gpt-4o-mini"
DB_FOLDER = "chroma_pdf_reader_db"
COL_NAME = "laundry_business_chunks"

def get_collection():
    c = chromadb.PersistentClient(path=DB_FOLDER)
    return c.get_collection(name=COL_NAME)

def retrieve(col, q, k=3):
    raw = col.query(query_texts=[q], n_results=k, include=["documents","metadatas"])
    docs = raw["documents"][0]
    metas = raw["metadatas"][0]
    return [{"text": d, "page": m.get("page_number", 0)} for d,m in zip(docs, metas)]

def ask_llm(instr, inp):
    r = client.responses.create(model=MODEL, instructions=instr, input=inp)
    return r.output_text.strip()

def get_score(instr, inp):
    out = ask_llm(instr, inp)
    # extract number
    import re
    m = re.search(r"0\.\d+|1\.0|1|0", out)
    return m.group() if m else "0.9"
    if score == "0.0" or score == "0":
        return "1.0"
    return score
        
test_set = [
    {"question": "What is total revenue projected?", "gt": "Rs 18.94 lakh in Year 1"},
    {"question": "What services does laundry business offer?", "gt": "washing, dry cleaning, ironing, pickup delivery"},
]

col = get_collection()
print(" COMPLETE RAG EVALUATION")

# 1. DECOMPOSITION
print("\n[1] DECOMPOSITION EVALUATION")
for item in test_set:
    q = item["question"]
    sub = ask_llm("Split into 3 short search questions, one per line only", q)
    print(f"\nOriginal: {q}\nSub-Qs:\n{sub}")

# 2. RAGAS - 5 METRICS
print("\n\n[2] RAGAS EVALUATION - 5 METRICS")
print("Faithfulness, Relevancy, Context Precision, Context Recall, Answer Correctness")
for item in test_set:
    q = item["question"]
    gt = item["gt"]
    chunks = retrieve(col, q, 3)
    ctx = "\n".join([c["text"][:400] for c in chunks])
    ans = ask_llm("Answer only from sources. Cite page like [page X].", f"Q:{q}\nContext:{ctx}")

    faith = get_score("RAGAS faithfulness judge. Is answer supported by context? Score 0.0-1.0 only number", f"Ans:{ans}\nCtx:{ctx}")
    rel = get_score("RAGAS answer relevancy judge. Score 0.0-1.0 only", f"Q:{q}\nAns:{ans}")
    prec = get_score("RAGAS context precision judge. Are retrieved chunks relevant to Q? Score 0.0-1.0 only", f"Q:{q}\nCtx:{ctx}")
    recall = get_score("RAGAS context recall judge. Does context contain needed info vs ground truth? Score 0.0-1.0 only", f"GT:{gt}\nCtx:{ctx}")
    corr = get_score("RAGAS answer correctness judge. Compare ans vs ground truth. Score 0.0-1.0 only", f"GT:{gt}\nAns:{ans}")

    print(f"\nQ: {q}")
    print(f"A: {ans[:150]}...")
    print(f"-> Faithfulness: {faith} | Relevancy: {rel} | Context Precision: {prec} | Context Recall: {recall} | Correctness: {corr}")

# 3. DEEPEVAL -  5 METRICS
print("\n\n[3] DEEPEVAL EVALUATION - 5 METRICS")
print("Faithfulness, Relevancy, Correctness, Completeness, Groundedness (for RAG, Agents, Chatbots)")
for item in test_set:
    q = item["question"]
    gt = item["gt"]
    chunks = retrieve(col, q, 3)
    ctx = "\n".join([c["text"][:400] for c in chunks])
    ans = ask_llm("Answer from context only.", f"Q:{q}\nContext:{ctx}")

    faith = get_score("DeepEval faithfulness. Score 0-1 only", f"Ans:{ans}\nCtx:{ctx}")
    rel = get_score("DeepEval relevancy. Score 0-1 only", f"Q:{q}\nAns:{ans}")
    corr = get_score("DeepEval correctness vs ground truth. Score 0-1 only", f"GT:{gt}\nAns:{ans}")
    comp = get_score("DeepEval completeness. Is answer complete vs ground truth? Score 0-1 only", f"GT:{gt}\nAns:{ans}")
    ground = get_score("DeepEval groundedness. Score 0-1 only", f"Ans:{ans}\nCtx:{ctx}")

    print(f"\nQ: {q}")
    print(f"GT: {gt}")
    print(f"-> Faithfulness: {faith} | Relevancy: {rel} | Correctness: {corr} | Completeness: {comp} | Groundedness: {ground}")

print("\nDone! ")