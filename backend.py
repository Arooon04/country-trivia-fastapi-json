# ==============================================================================
# STEP 1: IMPORTS & MODULE EXPLANATIONS
# ==============================================================================

# Standard Python Libraries
import hashlib                          # Generates SHA-256 hashes to detect if downloaded JSON changed (saves embedding tokens)
import json                             # Parses raw HTTP responses into Python dicts/lists and writes JSON files locally
from pathlib import Path                # Object-oriented file path handling (works cleanly on Windows, Mac, and Linux)
from typing import Optional             # Type hint indicating a field (like api_key) can be None or a string

# Third-Party Web & Environment Libraries
import requests                         # Makes HTTP GET requests to download the JSON from the REST Countries API
from dotenv import load_dotenv          # Reads environment variables (OPENAI_API_KEY, LANGCHAIN_API_KEY) from .env
from fastapi import FastAPI, HTTPException  # FastAPI creates our REST API server; HTTPException returns clean error codes
from pydantic import BaseModel          # Validates incoming JSON request bodies for POST endpoints (/ingest and /query)

# LangChain & LangSmith AI Framework Imports
from langchain_community.vectorstores import FAISS          # Local vector database for fast similarity search on disk
from langchain_core.documents import Document               # Standard LangChain data container holding text + metadata
from langchain_core.output_parsers import StrOutputParser   # Extracts plain string text from an LLM's response object
from langchain_core.prompts import ChatPromptTemplate       # Structures System and Human instructions for the LLM
from langchain_openai import ChatOpenAI, OpenAIEmbeddings   # Connects to OpenAI's chat models and embedding models
from langsmith import traceable                             # Decorator that sends detailed execution traces to LangSmith

# ==============================================================================
# STEP 2: INITIALIZE ENVIRONMENT, FILE PATHS & AI MODELS
# ==============================================================================

# Load variables from .env into os.environ (enables LangSmith tracing automatically)
load_dotenv()

# Create the FastAPI application instance
app = FastAPI(title="World Country Trivia RAG API")

# Define local paths inside the current project folder
BASE_DIR = Path(__file__).parent                            # Directory where backend.py lives
LOCAL_JSON_PATH = BASE_DIR / "downloaded_data.json"         # Where the downloaded JSON file will be saved
FAISS_INDEX_DIR = BASE_DIR / "faiss_country_index"          # Folder where FAISS saves index.faiss and index.pkl
HASH_FILE_PATH = BASE_DIR / ".json_hash"                    # Stores SHA-256 fingerprint of the last indexed JSON

# Initialize OpenAI Embeddings model
# - model="text-embedding-3-small": Fast, low-cost OpenAI embedding model
# - dimensions=512: Compresses vectors from 1536 to 512 dims (faster FAISS search + smaller disk size)
# - chunk_size=100: Sends up to 100 country texts in a single batch API call instead of 100 separate calls
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    dimensions=512,
    chunk_size=100,
)

# Initialize OpenAI Chat LLM
# - model="gpt-4o-mini": Cost-efficient, low-latency model ideal for RAG Q&A
# - temperature=0.1: Keeps answers factual and grounded in the JSON data (low creativity)
# - max_tokens=200: Caps output length to prevent runaway token usage
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.1,
    max_tokens=200,
)

# ==============================================================================
# STEP 3: HELPER FUNCTION TO FLATTEN COUNTRY JSON INTO LANGCHAIN DOCUMENTS
# ==============================================================================

def extract_country_document(record: dict) -> Document:
    """Converts a nested country JSON dictionary into a clean, token-efficient LangChain Document."""
    
    # 1. Extract country name (supports v5 'names' and v3.1 'name' keys)
    name_obj = record.get("names") or record.get("name") or {}
    common_name = name_obj.get("common", "Unknown") if isinstance(name_obj, dict) else str(name_obj)

    # 2. Extract capital cities (handles list of dicts in v5 or list of strings in v3.1)
    cap_raw = record.get("capitals") or record.get("capital") or []
    if isinstance(cap_raw, list):
        capitals = ", ".join(
            c.get("name", str(c)) if isinstance(c, dict) else str(c) for c in cap_raw
        ) or "None"
    else:
        capitals = str(cap_raw or "None")

    # 3. Extract geography & demographic fields
    region = record.get("region", "Unknown")
    subregion = record.get("subregion", "Unknown")
    population = record.get("population", "Unknown")
    landlocked = record.get("landlocked", "Unknown")

    # 4. Extract bordering country codes
    borders_raw = record.get("borders", [])
    borders = ", ".join(borders_raw) if isinstance(borders_raw, list) and borders_raw else "None"

    # 5. Extract spoken languages
    lang_raw = record.get("languages", {})
    if isinstance(lang_raw, dict):
        languages = ", ".join(lang_raw.values())
    elif isinstance(lang_raw, list):
        languages = ", ".join(str(x) for x in lang_raw)
    else:
        languages = str(lang_raw or "None")

    # 6. Extract currencies (handles list of currency objects in v5 or dict in v3.1)
    curr_raw = record.get("currencies", {})
    currency_list = []
    if isinstance(curr_raw, list):
        for item in curr_raw:
            if isinstance(item, dict):
                currency_list.append(f"{item.get('name', '')} ({item.get('code', '')})".strip())
    elif isinstance(curr_raw, dict):
        for code, info in curr_raw.items():
            if isinstance(info, dict):
                currency_list.append(f"{info.get('name', code)} ({code})".strip())
    currencies = ", ".join(currency_list) or "None"

    # 7. Format all fields into a single compact string to minimize embedding & prompt tokens
    compact_text = (
        f"{common_name} | Capital: {capitals} | Region: {region} ({subregion}) | "
        f"Languages: {languages} | Currencies: {currencies} | "
        f"Population: {population} | Landlocked: {landlocked} | Borders: {borders}"
    )

    # 8. Wrap text and filterable metadata inside a LangChain Document object
    return Document(
        page_content=compact_text,
        metadata={
            "country": common_name,
            "region": str(region),
        },
    )

# ==============================================================================
# STEP 4: AI CHAIN TO GENERATE 3 SUGGESTED TRIVIA QUESTIONS
# ==============================================================================

@traceable(name="Generate_Trivia_Questions", run_type="chain")  # Logs this step in LangSmith
def generate_trivia_questions(documents: list[Document]) -> list[str]:
    """Reads the first 3 country records and asks OpenAI to generate 3 trivia questions."""
    
    # Take only the first 3 documents so we spend minimal input tokens
    sample_text = "\n".join(doc.page_content for doc in documents[:3])

    # Define the prompt template with strict formatting rules
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Generate exactly 3 short, fun country trivia questions based ONLY on the records below. "
                "Return ONLY the 3 questions, one per line, without numbering or bullets.",
            ),
            ("human", "{sample}"),
        ]
    )

    # Build an LCEL (LangChain Expression Language) chain: Prompt -> LLM -> String Parser
    chain = prompt | llm | StrOutputParser()
    
    # Execute the chain and clean up each line into a Python list of 3 strings
    raw_output = chain.invoke({"sample": sample_text})
    questions = [line.strip("-•1234567890. ") for line in raw_output.strip().split("\n") if line.strip()]
    return questions[:3]

# ==============================================================================
# STEP 5: REQUEST SCHEMAS & FASTAPI ENDPOINTS (/ingest AND /query)
# ==============================================================================

# Pydantic schema validating the POST /ingest payload
class IngestRequest(BaseModel):
    json_url: str
    api_key: Optional[str] = None

# Pydantic schema validating the POST /query payload
class QueryRequest(BaseModel):
    question: str


@app.post("/ingest")
@traceable(name="Ingest_JSON_To_FAISS", run_type="chain")  # Logs full ingestion pipeline in LangSmith
def ingest_json(payload: IngestRequest):
    """Downloads JSON from URL, saves it locally, builds FAISS index, and suggests 3 trivia questions."""
    
    # 1. Prepare HTTP headers (attach Bearer token if user provided an API key)
    headers = {"Accept": "application/json"}
    if payload.api_key and payload.api_key.strip():
        token = payload.api_key.strip()
        headers["Authorization"] = token if token.lower().startswith("bearer ") else f"Bearer {token}"

    # 2. Fetch JSON data from the remote endpoint
    try:
        response = requests.get(payload.json_url.strip(), headers=headers, timeout=15)
        response.raise_for_status()
        raw_data = response.json()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to download JSON: {str(e)}")

    # 3. Save the downloaded JSON file locally in the same folder
    serialized_json = json.dumps(raw_data, indent=2, ensure_ascii=False)
    with open(LOCAL_JSON_PATH, "w", encoding="utf-8") as f:
        f.write(serialized_json)

    # 4. Extract the array of country records (handles v5 data.objects, data list, or raw list)
    if isinstance(raw_data, list):
        records = raw_data
    elif isinstance(raw_data, dict):
        data_node = raw_data.get("data")
        if isinstance(data_node, dict) and isinstance(data_node.get("objects"), list):
            records = data_node["objects"]
        elif isinstance(data_node, list):
            records = data_node
        else:
            records = [raw_data]
    else:
        raise HTTPException(status_code=400, detail="Unsupported JSON format.")

    # 5. Convert every JSON record into a LangChain Document
    documents = [
        extract_country_document(item)
        if isinstance(item, dict) and ("names" in item or "name" in item)
        else Document(page_content=json.dumps(item, ensure_ascii=False))
        for item in records
    ]

    if not documents:
        raise HTTPException(status_code=400, detail="No country records found in JSON.")

    # 6. Compute SHA-256 hash of the JSON to avoid re-embedding identical data
    new_hash = hashlib.sha256(serialized_json.encode("utf-8")).hexdigest()
    old_hash = HASH_FILE_PATH.read_text(encoding="utf-8").strip() if HASH_FILE_PATH.exists() else ""

    # 7. Embed documents and save FAISS index locally (only if JSON changed or index doesn't exist)
    if new_hash != old_hash or not FAISS_INDEX_DIR.exists():
        vector_store = FAISS.from_documents(documents=documents, embedding=embeddings)
        vector_store.save_local(str(FAISS_INDEX_DIR))
        HASH_FILE_PATH.write_text(new_hash, encoding="utf-8")
        cache_status = "Newly embedded into FAISS"
    else:
        cache_status = "Loaded existing FAISS index (0 embedding tokens used)"

    # 8. Generate 3 AI trivia questions based on the ingested records
    suggested_questions = generate_trivia_questions(documents)

    # 9. Return summary response to Streamlit frontend
    return {
        "status": "success",
        "saved_file": LOCAL_JSON_PATH.name,
        "faiss_folder": FAISS_INDEX_DIR.name,
        "records_indexed": len(documents),
        "index_note": cache_status,
        "suggested_questions": suggested_questions,
    }


@app.post("/query")
@traceable(name="Query_Trivia_RAG", run_type="chain")  # Logs retrieval + LLM generation in LangSmith
def query_rag(payload: QueryRequest):
    """Searches the local FAISS vector store and answers the user's trivia question via OpenAI."""
    
    # 1. Validate question and verify FAISS index exists on disk
    if not payload.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    if not FAISS_INDEX_DIR.exists():
        raise HTTPException(
            status_code=400,
            detail="FAISS index not found. Download and index the JSON first.",
        )

    # 2. Load the saved FAISS index from the local folder
    vector_store = FAISS.load_local(
        str(FAISS_INDEX_DIR),
        embeddings,
        allow_dangerous_deserialization=True,  # Required by LangChain when loading trusted local .pkl files
    )

    # 3. Create a retriever that fetches the top 3 most semantically similar country records
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    retrieved_docs = retriever.invoke(payload.question)

    # 4. Combine retrieved documents into a single context block for the LLM
    context_block = "\n".join(doc.page_content for doc in retrieved_docs)

    # 5. Build the RAG prompt instructing the LLM to answer strictly from retrieved context
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a concise World Country Trivia Assistant. Answer strictly from the context below in 2–4 short sentences or bullets. "
                "If the answer is not in the context, say so.\n\nContext:\n{context}",
            ),
            ("human", "{question}"),
        ]
    )

    # 6. Execute LCEL chain: Prompt -> OpenAI LLM -> Plain Text String
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": context_block, "question": payload.question})

    # 7. Return both the generated answer and the source records used
    return {
        "answer": answer,
        "sources": [
            {
                "country": doc.metadata.get("country", "Record"),
                "region": doc.metadata.get("region", ""),
                "content": doc.page_content,
            }
            for doc in retrieved_docs
        ],
    }