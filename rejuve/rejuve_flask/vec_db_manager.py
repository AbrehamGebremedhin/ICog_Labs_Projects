import os
import pandas as pd
from dotenv import load_dotenv
from langchain_astradb import AstraDBVectorStore
from astrapy.info import CollectionVectorServiceOptions
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


class Vec_Astradb:
    def __init__(self) -> None:
        load_dotenv("config.env")
        self.astra_db_api_endpoint = os.getenv("ASTRA_DB_API_ENDPOINT")
        self.astra_db_application_token = os.getenv(
            "ASTRA_DB_APPLICATION_TOKEN")
        self.nvidia_vectorize_options = CollectionVectorServiceOptions(
            provider="nvidia",
            model_name="NV-Embed-QA",
        )

    def populate_db(self, file_path):
        print(f"Loading {file_path}...")

        loader = PyPDFLoader(file_path, extract_images=False)
        pages = loader.load()

        # Calculate chunk size based on document length, e.g., 10% of document length
        chunk_size = max(200, int(len(pages) * 0.1))

        # Calculate chunk overlap based on chunk size, e.g., 5% of chunk size
        chunk_overlap = int(chunk_size * 0.05)

        # Split data into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            add_start_index=True,
        )

        chunks = text_splitter.split_documents(pages)

        vstore = AstraDBVectorStore(
            collection_name="Rejuve_Chat",
            api_endpoint=self.astra_db_api_endpoint,
            token=self.astra_db_application_token,
            namespace=None,
            collection_vector_service_options=self.nvidia_vectorize_options,
        )

        inserted_ids = vstore.add_documents(chunks)
        print(f"\nInserted {len(inserted_ids)} documents.")

    def similarity_search(self, query: str, k_value: int = 6):
        """Queries the vector database and returns results"""
        try:
            vec_db = AstraDBVectorStore(
                collection_name=f"Rejuve_Chat",
                api_endpoint=self.astra_db_api_endpoint,
                token=self.astra_db_application_token,
                collection_vector_service_options=self.nvidia_vectorize_options,
            )

            # Perform the similarity search
            results = vec_db.similarity_search(
                query=query, k=k_value)

            return results
        except Exception as e:
            raise RuntimeError(f"Error querying database: {e}")
