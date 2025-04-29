from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

template = """
Answer the following question.

Here is the conversation history: {context}

User: {question}

Answer:

"""

model = OllamaLLM(model="llama3.2")
prompt = ChatPromptTemplate.from_template(template)
chain= prompt | model

def handle_conversation():
    context = ""
    while True:
        question = input("User: ")
        if question == "exit":
            break
        
        result = chain.invoke({"context": context, "question": question})
        print(result)
        context += f"User: {question}\nAI: {result}\n\n"

result = chain.invoke({"context": "", "question": "You're the AI assistant. Help user to solve the issure"})
print(result)

if __name__ == "__main__":
    handle_conversation()