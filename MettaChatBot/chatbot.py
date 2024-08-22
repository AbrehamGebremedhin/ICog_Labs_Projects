import time
import pandas as pd
from langchain_core.documents import Document
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from pymongo import MongoClient

from neo4J_chat import Neo4JChat

client = MongoClient("mongodb://localhost:27017/")


class Chat:
    def __init__(self, model='llama3.1'):
        self.name = "metta"
        self.embeddings = OllamaEmbeddings(model="nomic-embed-text")
        self.model_name = model
        self.llm = Ollama(model=self.model_name)
        self.neo = Neo4JChat()
        self.database = client["metta_chatbot"]
        self.collection = self.database["chat_history"]
        self.session_chat = list()

    def query_db(self, query):
        context = self.neo.similarity_search(query)

        qna_prompt_template = """<|system|> You have been provided with a technical documentation, previous chat history and a query, try to find out 
        the answer to the question only using the information from the documentation and history. After giving the answer give the source URL of the information. Don't include the history in the answer. If the answer to the question is not found 
        within the documentation, return "I dont know" as the response. The response should be in HTML format.

        Documentation: {context}
        
        history: {session_history}

        Query: {query}<|end|>
        <|assistant|>"""

        # Use the custom prompt template to create a prompt and pass the required variables
        prompt = PromptTemplate(
            template=qna_prompt_template, input_variables=[
                "context", "session_history", "query"]
        )

        # Define the QNA chain
        chain = load_qa_chain(self.llm, chain_type="stuff", prompt=prompt)

        # Invoke the chain with the context and query
        answer = (chain.invoke(
            {"input_documents": context, "session_history": self.session_chat, "query": query}, return_only_outputs=True, ))['output_text']

        # Extract the answer from the response
        answer = (answer.split("<|assistant|>")[-1]).strip()

        # Store the chat history in the MongoDB database
        self.collection.insert_one(
            {"model": self.model_name, "query": query, "answer": answer})

        self.session_chat.append({"query": query, "answer": answer})

        return answer
