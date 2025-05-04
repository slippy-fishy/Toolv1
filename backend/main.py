import os
import json
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import pdfplumber
from typing import Dict, List
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

app = FastAPI()

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def read_root():
    return FileResponse("static/index.html")

# Storage file path
STORAGE_FILE = "pdf_storage.json"

# Load existing documents
def load_documents() -> Dict[str, str]:
    if os.path.exists(STORAGE_FILE):
        with open(STORAGE_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

# Save documents
def save_documents(documents: Dict[str, str]):
    with open(STORAGE_FILE, 'w', encoding='utf-8') as f:
        json.dump(documents, f, ensure_ascii=False, indent=2)

# In-memory storage for documents
documents = load_documents()

# Maximum file size (150 MB)
MAX_FILE_SIZE = 150 * 1024 * 1024

class ChatMessage(BaseModel):
    question: str
    filename: str

def preprocess_text(text: str) -> str:
    """Clean and preprocess the text."""
    # Convert to lowercase
    text = text.lower()
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text

def find_relevant_sections(text: str, question: str, num_sections: int = 3) -> str:
    """Find the most relevant sections of text based on the question."""
    # Preprocess the text
    text = preprocess_text(text)
    question = preprocess_text(question)
    
    # Split text into sentences
    sentences = sent_tokenize(text)
    
    # Get stopwords
    stop_words = set(stopwords.words('english'))
    
    # Create TF-IDF vectors
    vectorizer = TfidfVectorizer(stop_words=stop_words)
    try:
        tfidf_matrix = vectorizer.fit_transform([question] + sentences)
        
        # Calculate similarity between question and each sentence
        similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])
        
        # Get indices of most similar sentences
        most_similar_indices = np.argsort(similarities[0])[-num_sections:][::-1]
        
        # Combine most relevant sentences
        relevant_sections = [sentences[i] for i in most_similar_indices]
        return '. '.join(relevant_sections)
    except Exception as e:
        print(f"Error in find_relevant_sections: {str(e)}")
        return text[:1000]  # Return first 1000 characters if processing fails

def generate_response(question: str, relevant_text: str) -> str:
    """Generate a response based on the relevant text."""
    if not relevant_text.strip():
        return "I couldn't find any relevant information in the document to answer your question."
    
    return f"""Based on the document, here's what I found:

{relevant_text}

This information appears to be most relevant to your question: "{question}"

Note: This is a basic text matching response. For more sophisticated answers, consider using an AI model."""

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    
    # Check file size
    file_size = 0
    contents = await file.read()
    file_size = len(contents)
    
    if file_size > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400, 
            detail=f"File too large. Maximum size is 150MB. Your file is {file_size/1024/1024:.2f}MB"
        )
    
    try:
        with open(f"temp_{file.filename}", "wb") as f:
            f.write(contents)
        with pdfplumber.open(f"temp_{file.filename}") as pdf:
            text_parts = []
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
            text = "\n".join(text_parts)
        os.remove(f"temp_{file.filename}")
        documents[file.filename] = text
        save_documents(documents)
        return {"filename": file.filename, "message": "PDF uploaded and text extracted."}
    except Exception as e:
        if os.path.exists(f"temp_{file.filename}"):
            os.remove(f"temp_{file.filename}")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/documents")
async def list_documents():
    return {"documents": list(documents.keys())}

@app.get("/text/{filename}")
async def get_text(filename: str):
    if filename not in documents:
        raise HTTPException(status_code=404, detail="Document not found")
    return {"filename": filename, "text": documents[filename]}

@app.post("/chat")
async def chat_with_pdf(message: ChatMessage):
    if message.filename not in documents:
        raise HTTPException(status_code=404, detail="Document not found")
    
    try:
        # Get the PDF text
        pdf_text = documents[message.filename]
        
        # Find relevant sections
        relevant_text = find_relevant_sections(pdf_text, message.question)
        
        # Generate response
        response = generate_response(message.question, relevant_text)
        
        return {
            "answer": response,
            "question": message.question
        }
    except Exception as e:
        error_message = str(e)
        print(f"Error in chat_with_pdf: {error_message}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Error processing your question: {error_message}"}
        )
 