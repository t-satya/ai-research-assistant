# AI Research Assistant 🤖

An intelligent RAG system for querying 60+ foundational AI/ML research papers using semantic search and LLM-powered responses.

**🔗 [Live Demo](https://ai-research-assistant-m4y7.onrender.com/)** | **📚 [API Docs](https://ai-research-assistant-m4y7.onrender.com/docs)**

## What It Does
Ask natural language questions about key AI/ML research papers and get accurate, context-aware answers. The system is designed to be faithful to the source material, reducing the risk of hallucination.

### Example Queries
* "What is the attention mechanism in transformers?"
* "How does a Generative Adversarial Network (GAN) work?"
* "What are Residual Networks?"
* "Explain the Vision Transformer (ViT) architecture."

---

## 🛠️ Tech Stack
* **Backend:** Python 3.11, FastAPI
* **RAG Pipeline:** Sentence-Transformers, ChromaDB, Groq (Llama 3)
* **Deployment:** Docker, Render

---

## 🚀 Running the Project

You can run this project either with Docker (recommended for ease of use) or by setting up a local Python environment.

### Option 1: Using Docker (Recommended)
This method packages the entire application and its dependencies into a single container.

1.  **Prerequisites:** Docker Desktop must be installed and running.

2.  **Build the Docker image:** This command reads the `Dockerfile`, downloads dependencies, and builds the database by running `build_database.py`.
    ```bash
    docker build -t ai-research-assistant .
    ```

3.  **Run the container:**
    * Create a `.env` file in the root directory and add your Groq API key: `GROQ_API_KEY="your_api_key_here"`
    * Run the following command to start the application.
    ```bash
    docker run --env-file .env -p 8000:8000 ai-research-assistant
    ```
The application will be available at `http://localhost:8000`.

### Option 2: Manual Local Setup
This method requires you to manually install dependencies and build the database.

1.  **Clone the repository and install dependencies:**
    ```bash
    git clone [https://github.com/t-satya/ai-research-assistant.git](https://github.com/t-satya/ai-research-assistant.git)
    cd ai-research-assistant
    pip install -r requirements.txt
    ```

2.  **Create a `.env` file** with your `GROQ_API_KEY`.

3.  **Build the database:** This is a one-time setup step to process the papers.
    ```bash
    python build_database.py
    ```

4.  **Run the FastAPI server:**
    ```bash
    uvicorn app:app --reload
    ```
The application will be available at `http://localhost:8000`.

---

## 🗺️ Roadmap (V2.0)
Future enhancements planned for this project include:
- [ ] Add conversation memory to handle follow-up questions.
- [ ] Show retrieved sources with citations to improve transparency.
- [ ] Implement a basic evaluation suite to formally measure retrieval quality.
- [ ] Integrate Hybrid Search or a re-ranking model to improve retrieval accuracy.