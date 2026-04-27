
# Assignment3
## Rules and Tasks


## Team A (Agam, Parisa, Anupam) Deadline 28 April 2026


### Text Preprocessing & Embedding 
Message is cleaned, tokenised, and converted to a semantic vector embedding using a sentence transformer (e.g., all-MiniLM-L6-v2). Embedding is used simultaneously for retrieval and classification.
### Crisis Classifier ()
A fine-tuned binary classifier (e.g., DistilBERT or zero-shot with GPT) scores the message for crisis indicators — suicidal ideation, self-harm, acute distress. Trained on CLPsych / CSSRS data.

This has two parts 
#### Part A Crisis Escalation
Immediately surface empathetic acknowledgement + emergency resources. Lifeline 13 11 14 (AU), Beyond Blue. Log for safety review. Skip RAG pipeline entirely.

#### Part B RAG Retrieval
Embedding is used to query a vector database (ChromaDB / FAISS) built from the Counsel Chat dataset and curated CBT/mindfulness articles. Top-k chunks retrieved.



## Team B (William, Taison, Vivek) Deadline 4 May 2026 



### Structured Knowledge Base (will be included)
Curated mental health knowledge: DSM-5 conditions, CBT coping strategies, mindfulness exercises, psychoeducation articles. Structured as a knowledge graph linking symptoms → strategies → resources. This is the grounding layer that keeps responses clinically informed.

### LLM Response Generation
Retrieved chunks + conversation history + user message are assembled into a structured prompt. An LLM (e.g., GPT-4o / LLaMA 3) generates a grounded, empathetic response. System prompt enforces: non-judgemental tone, never diagnose, always refer to professionals for serious concerns.

### Empathetic Response → User
Response is delivered to the user with source citations (transparency), a safety disclaimer, and optionally a follow-up suggestion. Session history is updated for multi-turn context.





### Streamlit deployment of the project () Dead line - 5th May 2026
### PPT presentation  (All) Deadline 5th May 2026 






