# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file first to leverage Docker layer caching
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# --- NEW: Pre-download the sentence transformer model during the build ---
# This saves time when the container starts up for the first time.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Copy the rest of your application's code (app.py, index.html, etc.)
COPY . .

# Let Docker know that the container listens on port 8000
EXPOSE 8000

# Define the command to run your app when the container starts
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]