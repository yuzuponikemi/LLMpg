from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import OllamaLLM
from langchain.indexes import VectorstoreIndexCreator

# Load the document
print("Loading the document...")
loader = PyPDFLoader(r"C:\Users\YIkemi.DESKTOP-1O5DLKF\Downloads\Thinkcyte EOL test instructions (06.07)_ReviewedByPD.docx (1).pdf")
print("Document loaded.")

# Create embeddings
print("Creating embeddings...")
embeddings = OllamaEmbeddings(model="mxbai-embed-large")
print("Embeddings created.")

# Initialize the language model
print("Initializing the language model...")
llm = OllamaLLM(model="qwen2.5-coder:0.5b")
print("Language model initialized.")

# Create the index
print("Creating the index...")
index = VectorstoreIndexCreator(
    vectorstore_cls=Chroma,
    embedding=embeddings
).from_loaders([loader])
print("Index created.")

# Define the query
query = "how do I do Sheath fluid pathway tests?"
print(f"Query defined: {query}")

# Perform the query
print("Performing the query...")
answer = index.query(query, llm=llm)
print("Query performed.")

# Print the answer
print("Answer:")
print(answer)