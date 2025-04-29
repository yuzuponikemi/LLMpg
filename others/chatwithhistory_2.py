from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from flask import Flask, render_template, request, jsonify
import ollama

# List all available local models


template = """
Answer the following question.

Here is the conversation history: {context}

User: {question}

Answer:

"""

app = Flask(__name__)

# Function to handle long contexts
def truncate_context(context, max_length=1000):
    if len(context) > max_length:
        context = context[-max_length:]
    return context

# Function to get available local models
def get_available_models():
    models = ollama.list()
    # Extract model names and other relevant information
    model_list = [{"model": model.model, "modified_at": model.modified_at.isoformat(), "size": model.size, "parameter_size": model.details.parameter_size} for model in models.models]
    return model_list

@app.route('/')
def home():
    return render_template('chat.html')

@app.route('/models', methods=['GET'])
def get_models():
    available_models = get_available_models()
    return jsonify(available_models)

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    question = data['question']
    context = data['context']
    model_name = data['model']
    
    context = truncate_context(context)
    
    # Load the selected model
    model = OllamaLLM(model=model_name)
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model
    
    result = chain.invoke({"context": context, "question": question})
    
    context += f"User: {question}\nAI: {result}\n\n"
    context = truncate_context(context)
    
    return jsonify({"response": result, "context": context})

if __name__ == "__main__":
    app.run(debug=True)