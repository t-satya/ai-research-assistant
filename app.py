import os
import chromadb
from dotenv import load_dotenv
from fastapi import FastAPI,HTTPException
from pydantic import BaseModel,Field
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

load_dotenv()

# --- 1. Initialize Models and Database ---
print("Initalizing components...")

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path='chroma_db')
collection = client.get_collection(name='ai_papers')
groq_client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

print("Initialization Complete")

# --- FastAPI App ---
app = FastAPI(
    title="AI Research Assistant API",
    description="RAG-based Q&A system for AI/ML research papers",
    version = "1.0.0"
)

# ADD CORS - Cross Origin Resource Sharing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],#for now allowing all change to frontend URL later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    
)
# --- Pydantic Models for Request and Response ---
class QueryRequest(BaseModel):
    question: str = Field(...,min_length=1,max_length=500)
    class Config:
        json_schema_extra = {
            "example":{
                "question":"What is the attention mechanism in transformers?"
            }
        }

class QueryResponse(BaseModel):
    question: str
    answer: str
    chunks_used: int

# --- 2. Main RAG Logic in a Function ---

def answer_question(question):
    """
        Takes a user's question, retrieves relevant context and generates an answer.
    """
    print(f"Received Question: {question}")

    # --- RETRIEVAL STEP ---
    # Convert the question into an embedding
    question_embedding = embedding_model.encode(question)

    # Search the Chroma DB for the most relevant chunks
    # We'll retrieve the top 5 most similar chunks
    results = collection.query(
        query_embeddings=[question_embedding.tolist()],
        n_results=30
    )

    documents = results['documents'][0]
    distances = results['distances'][0]

    sorted_docs = sorted(zip(documents,distances),key=lambda x:x[1])
    selected_chunks_with_distances = []
    total_tokens=0
    MAX_CONTEXT_TOKENS=4000

    for doc,dist in sorted_docs:
        doc_tokens=len(doc)//4
        if total_tokens + doc_tokens < MAX_CONTEXT_TOKENS:
            selected_chunks_with_distances.append((doc,dist))
            total_tokens+=doc_tokens
        else:
            break

    print(f"\nSelected {len(selected_chunks_with_distances)} chunks (~{total_tokens} tokens) to send to LLM.")
    # print("\n--- Top Retrieved Chunks (with distances) ---")
    # for i, (chunk, distance) in enumerate(selected_chunks_with_distances[:5], 1): # Show top 5
    #     print(f"\n--- Chunk {i} (Distance: {distance:.4f}) ---")
    #     print(chunk)
    # print("\n--- End of Chunks ---")

    # Extract just the text to join for the context
    selected_context = [item[0] for item in selected_chunks_with_distances]
    retrieved_context = "\n\n".join(selected_context)
    
    print("Retrieved context from the database.")

    # --- GENERATION STEP ---
    # Create the prompt for the LLM
    # Prompt includes the Persona, the Rule, the Context, the Question
    prompt = f"""
    You are an AI Research Assistant. Answer the user's question based *only* on the following context.
    If the context does not contain the answer, say "I cannot find the answer in the provided documents"
    Context:
    ---
    {retrieved_context}
    ---

    Question: {question}

    Answer with specific references to papers when possible
    """
    print("Sending request to Groq to generate the final answer...")

    try:
        response = groq_client.chat.completions.create(
            model = "llama-3.1-8b-instant",
            messages = [
                {'role':'user','content':prompt}
            ],
            max_tokens=1000
        )
        final_answer = response.choices[0].message.content
        return final_answer,len(selected_chunks_with_distances)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM Error: {str(e)}")

# --- 3. API Endpoints ---

@app.get("/",response_class=FileResponse)
async def read_index():
    """
    Serves the frontend HTML file.
    """
    return "index.html"

@app.get("/health")
async def health_check():
    """
    Check if all components are working
    """
    try:
        count = collection.count() #to check if vector store is live and connected
        embedding_model.encode("test") # to load the complex embedding model into computer's memory
        return {"status":"healthy",
                "database_docs":count}
    except Exception as e:
        raise HTTPException(status_code=503,detail=f"Unhealthy: {str(e)}")
    
@app.post("/ask",response_model = QueryResponse)
async def ask_question(request: QueryRequest):
    """
    Ask a question about AI/ML research papers
    """
    try:
        final_answer,chunks_used = answer_question(request.question)
        return QueryResponse(
            question = request.question,
            answer = final_answer,
            chunks_used= chunks_used
                            )
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500,
                            detail=f"Error: {str(e)}")
    
        

