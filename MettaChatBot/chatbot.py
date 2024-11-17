import os
import pandas as pd
from keybert import KeyBERT
from pathlib import Path
from typing import Dict, Any, List
from dotenv import load_dotenv
from langchain_astradb import AstraDBVectorStore
from astrapy.info import CollectionVectorServiceOptions
from langchain_core.output_parsers import JsonOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

class Chat:
    def __init__(self) -> None:
        """Initializes the chatbot engine and its resources."""
        load_dotenv("config.env")
        self.resources = Path(__file__).parent / "Resource"
        self.nvidia_vectorize_options = CollectionVectorServiceOptions(
            provider="nvidia",
            model_name="NV-Embed-QA",
        )
        self.llm = ChatGoogleGenerativeAI(model="gemini-pro")
        self.session_chat = []
        self.parser = JsonOutputParser()  # Initialize the JSON output parser
        self.keyword_extractor = KeyBERT()  # Initialize KeyBERT for keyword extraction

    def extract_keywords(self, query: str) -> List[str]:
        """Extracts keywords from the user query using KeyBERT."""
        keywords = self.keyword_extractor.extract_keywords(query, keyphrase_ngram_range=(1, 2), stop_words='english', top_n=5)

        # Prioritize code-related keywords if found
        code_terms = ["function", "class", "method", "code", "snippet", "example"]
        extracted_keywords = [kw[0] for kw in keywords]

        if any(term in query.lower() for term in code_terms):
            # Boost code-related keywords
            return [kw for kw in extracted_keywords if kw.lower() in code_terms] + extracted_keywords
        return extracted_keywords

    def select_db(self) -> AstraDBVectorStore:
        """Returns the collection of a subject in a given grade database."""
        astra_db_api_endpoint = os.getenv("ASTRA_DB_API_ENDPOINT")
        astra_db_application_token = os.getenv(
            "ASTRA_DB_APPLICATION_TOKEN")

        if not astra_db_api_endpoint or not astra_db_application_token:
            raise ValueError(
                f"Database configuration for metta_collection is missing.")

        vstore = AstraDBVectorStore(
            collection_name=f"metta_collection",
            api_endpoint=astra_db_api_endpoint,
            token=astra_db_application_token,
            collection_vector_service_options=self.nvidia_vectorize_options,
        )

        return vstore

    def search_db(self, query: str, keywords: str = None, k_value: int = 30) -> List[Dict[str, Any]]:
        """Queries the vector database and returns results without filtering by similarity score"""
        try:
            vec_db = self.select_db()

            # Construct the search query based on whether unit is provided
            if keywords:
                search_query = f"{query} keywords:{keywords}"
            else:
                search_query = query

            # Perform the similarity search
            results = vec_db.similarity_search_with_score(query=search_query, k=k_value)

            # Return all results (only the documents, not the similarity scores)
            return [item[0] for item in results]

        except Exception as e:
            raise RuntimeError(f"Error querying database: {e}")

    def _generate_response(self, prompt: PromptTemplate, context: List[Dict[str, Any]], variables: Dict[str, Any]) -> str:
        """Generates a response using the LLM based on the given prompt and context."""
        chain = load_qa_chain(self.llm, chain_type="stuff", prompt=prompt)
        try:
            response = chain.invoke(
                {"input_documents": context, **variables}, return_only_outputs=True)
            return response['output_text'].strip()
        except Exception as e:
            raise RuntimeError(f"Error generating response from LLM: {e}")

    def clean_answer(self, answer: str) -> str:
        """Cleans the generated answer by removing newlines, backticks, and unnecessary spaces."""
        # Remove newlines, backticks, and extra spaces
        cleaned_answer = (
            answer.replace('\n', ' ')
                  .replace('\n\n', ' ')
                  .replace('\\', ' ')
                  .replace('\n\n\n', ' ')   # Replace all newlines with spaces
                  .replace('\t', ' ')   # Replace tabs with spaces
                  .replace('```', '')   # Remove triple backticks
                  .replace('`', '')     # Remove single backticks
                  .strip()              # Remove leading/trailing whitespace
        )
        
        # Replace multiple spaces with a single space
        return ' '.join(cleaned_answer.split())
    def query_db(self, query: str) -> str:
        """Handles a chatbot session, querying the database and responding to the user's query."""
        
        # Extract keywords from the query
        keywords = self.extract_keywords(query)
        keyword_query = ' '.join(keywords)  # Use extracted keywords to build a query
        
        # Query the database using the extracted keywords
        context = self.search_db(keyword_query)

        # Define the prompt template
        qna_prompt_template = """<|system|> You have been provided with a documentation, previous chat history and a query, try to find out 
        the answer to the question using the information from the documentation and history and your knowledge base. Make sure to provide an elaborate answer in a single line but make the answer is complete. If the user request is to generate a metta code, then return only the metta code.
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

        # Generate the response using LLM
        answer = self._generate_response(
            prompt, context, {"session_history": self.session_chat, "query": query})

        # Store the conversation history
        self.session_chat.append({"query": query, "answer": answer})
        
        # Clean the answer
        return self.clean_answer(answer)


chat = Chat()
print(chat.query_db("generate a metta function that accepts a string and prints it"))
