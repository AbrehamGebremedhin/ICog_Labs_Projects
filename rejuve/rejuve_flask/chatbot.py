import os
import asyncio
from dotenv import load_dotenv
from functools import lru_cache
from langchain.schema import Document
from vec_db_manager import Vec_Astradb
from langchain.prompts import PromptTemplate
from langchain_ollama import OllamaEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from query_generator import LangchainToCypher

load_dotenv('config.env')  # Update with your actual path if needed


class Chat:
    def __init__(self, db_type='graph'):
        """
        Initialize the Chat class.
        :param db_type: 'graph' for Neo4j, 'vector' for Vec_Astradb
        """
        self.embeddings = None
        self.llm = None
        self.session_chat = list()
        self.db_type = db_type  # Set the database type

        # Initialize the appropriate database manager based on the db_type
        if self.db_type == 'graph':
            self.db_manager = LangchainToCypher()
        elif self.db_type == 'vector':
            self.db_manager = Vec_Astradb()
        else:
            raise ValueError("Invalid db_type. Choose 'graph' or 'vector'.")

    def load_embeddings(self):
        if self.embeddings is None:
            self.embeddings = OllamaEmbeddings(model="nomic-embed-text")

    def load_llm(self):
        if self.llm is None:
            self.llm = ChatGoogleGenerativeAI(model="gemini-pro")

    @lru_cache(maxsize=128)
    def cached_similarity_search(self, query):
        """
        Perform a similarity search using the selected database manager.
        """
        if self.db_type == 'graph':
            return self.db_manager.run_query(query)
        elif self.db_type == 'vector':
            return self.db_manager.similarity_search(query)

    async def query_db(self, query):
        """
        Query the selected database and get a response from the LLM.
        """
        self.load_embeddings()
        self.load_llm()

        # Perform similarity search
        context = self.cached_similarity_search(query)

        # Ensure context is not empty
        if not context:
            return {"answer": "No relevant documents found."}
        
        if self.db_type == 'graph':
            if isinstance(context, list):
                # Convert each item in the list to a string and join them with newlines
                context_text = "\n".join([str(item) for item in context])
            else:
                context_text = str(context)  # Convert directly if not a list

            # Create Document with the flattened text
            context = [Document(page_content=context_text)]
            # Template for generating a prompt for the LLM
            qna_prompt_template = """<|system|> You are an assistant proficient in querying and explaining a Neo4j graph database containing biological data. Based on the provided context, generate an elaborate explanation of the data. If the query output lacks sufficient information to provide explanation, respond with "I don't know."

            Rules for answer generation:
            1. Do not include any escape sequences or special characters.

            Output as JSON:
            {{
                "answer": "The explanation"
            }}


            Database Results:
            {context}

            Previous Context:
            {session_history}

            <|assistant|>"""
        else:
            # Template for generating a prompt for the LLM
            qna_prompt_template = """<|system|> You have been provided with a documentation, previous chat history and a query, try to find out 
            the answer to the question only using the information from the documentation and history. Make the answer an elaborate one.
            If the answer to the question is not found within the documentation, return "I don't know" as the response. 
            
            Rules for answer generation:
            1. Do not include any escape sequences or special characters.
            
            Output as JSON:
            {{
                "answer": "The answer to the query"
            }}

            Documentation: {context}
            
            history: {session_history}

            Query: {query}<|end|>
            <|assistant|>"""

        prompt = PromptTemplate(
            template=qna_prompt_template,
            input_variables=["context", "session_history", "query"]
        )

        # Load the question-answering chain
        chain = load_qa_chain(self.llm, chain_type="stuff", prompt=prompt)

        try:
            # Execute the chain
            answer = await asyncio.to_thread(chain.invoke, {
                "input_documents": context,
                "session_history": self.session_chat,
                "query": query
            }, return_only_outputs=True)

            # Extract and clean the answer
            answer = (answer['output_text'].split("<|assistant|>")[-1]).strip()

        except Exception as e:
            # Handle errors gracefully
            return {"answer": f"Error during processing: {str(e)}"}

        # Add the query and answer to the session chat history
        self.session_chat.append({"query": query, "answer": answer})

        return answer

    def close(self):
        """
        Close the database connection if applicable.
        """
        if hasattr(self.db_manager, 'close'):
            self.db_manager.close()

async def main():
    chat_graph = Chat(db_type='graph')
    chat_vector = Chat(db_type='vector')

    graph_query = "Get all genes with a 'protein_coding' type"
    vec_query = "explain what rejuve is "
    
    print("Using Neo4j:")
    answer_graph = await chat_graph.query_db(graph_query)
    print(answer_graph)

    # print("\nUsing Vec_Astradb:")
    # answer_vector = await chat_vector.query_db(vec_query)
    # print(answer_vector)

    # Close connections
    chat_vector.close()

asyncio.run(main())

# asyncio.run(print(Chat().query_db("print hello world code")))
# asyncio.run(print(Chat().query_db("a function to check a give year is a leap year or not")))

