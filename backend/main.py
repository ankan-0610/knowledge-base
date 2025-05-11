from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
import os
import shutil
import uuid
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# Setup embedding model + vector store path

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
print("huggingface embedding model loaded successfully.")   
print(embedding.embed_query("hello world"))
CHROMA_DIR = "chroma_db"

# Load or create vectorstore
if not os.path.exists(CHROMA_DIR):
    os.mkdir(CHROMA_DIR)
vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=embedding)
print("Vectorstore docs count:", len(vectorstore._collection.get()["documents"]))

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        filename = f"temp_files/{uuid.uuid4()}_{file.filename}"
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        with open(filename, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # Load and parse file
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(filename)
        else:
            loader = TextLoader(filename)

        documents = loader.load()

        # Chunk and embed
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(documents)

        vectorstore.add_documents(chunks)
        vectorstore.persist()

        return {"status": "success", "chunks_added": len(chunks)}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# Initialize Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Use LangChain's GoogleGenerativeAI wrapper
llm = GoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)

# Create a basic QA chain
prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""
    You are an assistant answering questions based on the provided context.
    Context: {context}
    Question: {question}
    Answer:""",
)

chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt_template)
print("✅ QA Chain loaded successfully with Gemini model.")

@app.post("/chat")
async def chat(query: str = Form(...)):
    try:
        print("initialize retriever")
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
        print("fetching relevant docs")
        docs = await retriever.ainvoke(query)
        # The 'chain' is already configured with the Gemini LLM (GoogleGenerativeAI)
        print("running model inference...")
        response = await chain.ainvoke({
            "input_documents": docs,
            "question": query
        })
        print("model inference complete.")
        return {"response": response["output_text"]}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/")
def health():
    return {"message": "Server running!"}
