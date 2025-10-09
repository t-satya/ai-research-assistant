import os
import json
import pymupdf
import time
import chromadb
from sentence_transformers import SentenceTransformer

def read_pdf(file_path):
    try:
        doc = pymupdf.open(file_path)
        full_text = "".join(page.get_text("text") for page in doc)
        doc.close()
        return full_text
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None
    
def chunking(full_text,chunk_size=3000,overlap=200):
    return [full_text[i:i+chunk_size] for i in range(0,len(full_text),chunk_size-overlap)]

if __name__ == "__main__":
    strat_time = time.time()
    # --- 1. Load the Paper Titles from the JSON Cache and Chunk All Documents ---
    print("Loading paper titles from paper_titles.json...")
    with open('paper_titles.json', 'r', encoding='utf-8') as f:
        titles_cache = json.load(f)

    print("Strating database build process...")

    papers_folder = "Papers"
    pdf_files = [f for f in os.listdir(papers_folder) if f.endswith(".pdf")]
    print(f"Found {len(pdf_files)} PDF files to process")

    all_chunks=[]
    metadatas=[]

    for i,pdf_file in enumerate(pdf_files):
        file_path = os.path.join(papers_folder,pdf_file)
        print(f"Processing file {i+1}/{len(pdf_files)} : {pdf_file}")
        document_text = read_pdf(file_path)

        if document_text:
            chunks = chunking(document_text)
            # Get the title from our cache
            title = titles_cache.get(pdf_file, {}).get('title', pdf_file) # Fallback to filename
            
            # Create context-enriched chunks and their metadata
            # for each chunk in a document create a metadata dictionary
            for chunk in chunks:
                enriched_chunk = f"Paper Title: {title}\n\n{chunk}"
                all_chunks.append(enriched_chunk)
                metadatas.append({'source': pdf_file, 'title': title})

            print(f"-> Extracted {len(chunks)} enriched chunks.")
        
    # --- 2. Create Embeddings ---
    print("Loading embeddings model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    print("Creating embeddings for all chunks...")
    embeddings = model.encode(all_chunks,show_progress_bar=True)
    print(f"Embeddings created with shape : {embeddings.shape}")

    # --- 3. Build and Save the Chroma DB Vector Store ---
    print("Building the ChromaDB vector store...")

    # Create Persistent Database "chroma_db" and collection "ai_papers"
    client = chromadb.PersistentClient(path='chroma_db')
    collection = client.get_or_create_collection(name='ai_papers')

    # Creating unique ids for chunks for ChromaDB
    chunk_ids = [str(i) for i in range(len(all_chunks))]

    # Add the documents and embeddings to the collection in batches
    batch_size = 500
    for i in range(0,len(all_chunks),batch_size):
        batch_end = i+batch_size
        collection.add(
            ids=chunk_ids[i:batch_end],
            embeddings=embeddings[i:batch_end],
            documents=all_chunks[i:batch_end],
            metadatas=metadatas[i:batch_end]
        )
        print(f"Added batch {i//batch_size + 1} to Chroma DB")



    stop_time = time.time()

    print("\n\n--- Database Build Complete ---")
    print(f"Time taken : {stop_time-strat_time:.2f} seconds")
    print(f"Vector store created with {collection.count()} entries.")