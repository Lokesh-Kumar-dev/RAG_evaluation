# RAG Evaluation 

This project evaluates a RAG (Retrieval-Augmented Generation) pipeline for a Laundry Business Plan.

## Files
- `evaluate.py` - Main evaluation script
- `Execution.txt` - Full execution logs

## What this does
1. **Query Decomposition**: Breaks complex queries into 3 sub-questions using OpenAI
2. **RAGAS Evaluation (5 Metrics)**:
   - Faithfulness
   - Relevancy
   - Context Precision
   - Context Recall
   - Answer Correctness
3. **DeepEval Evaluation (5 Metrics)**:
   - Faithfulness
   - Relevancy
   - Correctness
   - Completeness
   - Groundedness

## Tech Stack
- ChromaDB - Vector DB
- OpenAI GPT - LLM
- Python

## How to Run
```bash
pip install chromadb openai python-dotenv
python evaluate.py