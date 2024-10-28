import os
from dotenv import load_dotenv
from functools import lru_cache
from langchain.schema import Document
from vec_db_manager import Vec_Astradb
from langchain.prompts import PromptTemplate
from langchain_ollama import OllamaEmbeddings
from query_generator import LangchainToCypher
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import JsonOutputParser
from langchain.chains.combine_documents import create_stuff_documents_chain


load_dotenv('config.env')

class Chat:
    def __init__(self, db_type='graph'):
        """
        Initialize the Chat class.
        :param db_type: 'graph' for Neo4j, 'vector' for Vec_Astradb
        """
        self.embeddings = None
        self.llm = None
        self.session_chat = list()
        self.db_type = db_type

        if self.db_type == 'graph':
            self.db_manager = LangchainToCypher()
        elif self.db_type == 'vector':
            self.db_manager = Vec_Astradb()
        else:
            raise ValueError("Invalid db_type. Choose 'graph' or 'vector'.")
        self.parser = JsonOutputParser()

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

    def query_db(self, query):
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
                context_text = "\n".join([str(item) for item in context])
            else:
                context_text = str(context)
            context = [Document(page_content=context_text)]

            prompt_template = """<|system|> You are an assistant proficient in explaining any biological information that can be extracted from biological data from Rejuve.Bio's BioAtomspace knowledge graph using your own knowledge base. Based on the context, provide an elaborate explanation of all the biological data of the context."

            IMPORTANT: You must format your response as a single-line JSON string with no line breaks, no extra spaces, and no escaped characters. The format should be exactly like this: {{"answer":"Your answer here"}}

            Database Results:  
            {context}  

            Previous Context:  
            {session_history}  

            <|assistant|>"""
        else:
            prompt_template = """<|system|> You have been provided with a documentation, previous chat history and a query, try to find out 
            the answer to the question only using the information from the documentation and history. Make the answer an elaborate one.
            If the answer to the question is not found within the documentation.

            IMPORTANT: You must format your response as a single-line JSON string with no line breaks, no extra spaces, no \\, and no escaped characters. The format should be exactly like this: {{"answer":"Your answer here"}}

            Documentation: {context}
            
            history: {session_history}

            Query: {query}

            <|assistant|>"""

        # Create the prompt from the template
        prompt = PromptTemplate.from_template(prompt_template)

        # Create the stuff documents chain
        chain = create_stuff_documents_chain(
            llm=self.llm,
            prompt=prompt
        )

        try:
            # Execute the chain with the input variables
            answer = chain.invoke({
                "context": context,
                "session_history": self.session_chat,
                "query": query
            })

            # Extract the answer
            clean_answer = str(answer.strip())
            clean_answer.replace("\\", '')

            parsed_answer = self.parser.parse(clean_answer)

            # Add the query and answer to the session chat history
            self.session_chat.append({
                "query": query,
                "answer": clean_answer
            })

            return parsed_answer

        except Exception as e:
            return {"answer": f"Error during processing: {str(e)}"}

    def close(self):
        """
        Close the database connection if applicable.
        """
        if hasattr(self.db_manager, 'close'):
            self.db_manager.close()