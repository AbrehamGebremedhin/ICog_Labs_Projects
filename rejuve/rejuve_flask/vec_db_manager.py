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

        # Instead of chunking, treat each page as a separate document
        documents = []
        for page in pages:
            metadata = {
                "source": page.metadata.get("source"),
                "page": page.metadata.get("page"),
                "start_index": page.metadata.get("start_index"),
                # Add other attributes as needed
            }
            documents.append({"text": page.page_content, "metadata": metadata})

        vstore = AstraDBVectorStore(
            collection_name="Rejuve_Chat",
            api_endpoint=self.astra_db_api_endpoint,
            token=self.astra_db_application_token,
            namespace=None,
            collection_vector_service_options=self.nvidia_vectorize_options,
        )

        inserted_ids = vstore.add_documents(documents)
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
