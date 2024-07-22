import time
import pandas as pd
from langchain_core.documents import Document
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_postgres.vectorstores import PGVector
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from pymongo import MongoClient

from neo4J_chat import Neo4JChat

client = MongoClient("mongodb://localhost:27017/")


class Chat:
    def __init__(self, db='vec_db', model='phi3:mini'):
        self.connection = "postgresql+psycopg://langchain:langchain@localhost:6024/langchain"  # Uses psycopg3!
        self.name = "metta"
        self.embeddings = OllamaEmbeddings(model="nomic-embed-text")
        self.vectorstore = PGVector(
            embeddings=self.embeddings,
            collection_name=self.name,
            connection=self.connection,
            use_jsonb=True,
        )
        self.model_name = model
        self.llm = Ollama(model=self.model_name)
        self.neo = Neo4JChat()
        self.db = db
        self.database = client["metta_chatbot"]
        self.collection = self.database["chat_history"]

    def load_data(self):
        df = pd.read_csv("data.csv")
        print(df.head())
        docs = []
        for index, row in df.iterrows():
            print(f"Processing chunk {index + 1}/{len(df)}")
            docs.append(
                Document(
                    page_content=str(row["Text"]),
                    metadata={"id": index, "url": row["URL"], "title": row["Title"], "keywords": row["Keywords"]},
                )
            )

        self.vectorstore.add_documents(docs, ids=[doc.metadata["id"] for doc in docs])

    def query_db(self, query):
        if self.db == 'vec_db':
            # Conduct a vector search for the user query and return the top 2 results
            context = self.vectorstore.similarity_search(query, k=2)
        else:
            context = self.neo.similarity_search(query)

        # Custom prompt template suitable for the Phi-3 model
        qna_prompt_template = """<|system|> You have been provided with the context and a query, try to find out
        the answer to the question only using the context information, and give the answer. If the answer to the question is not found
        within the context, return "I dont know" as the response. <|end|> <|user|> Context: {context}

        Query: {query}<|end|>
        <|assistant|>"""

        # Use the custom prompt template to create a prompt and pass the required variables
        prompt = PromptTemplate(
            template=qna_prompt_template, input_variables=["context", "query"]
        )

        # Define the QNA chain
        chain = load_qa_chain(self.llm, chain_type="stuff", prompt=prompt)

        # Invoke the chain with the context and query
        answer = (chain.invoke({"input_documents": context, "query": query}, return_only_outputs=True, ))['output_text']

        # Extract the answer from the response
        answer = (answer.split("<|assistant|>")[-1]).strip()

        # Store the chat history in the MongoDB database
        self.collection.insert_one(
            {"vector_db": self.db, "model": self.model_name, "query": query, "answer": answer})

        return answer


chat = Chat("neo4j", model="llama3")
print(chat.query_db("what is the main use of metta"))
