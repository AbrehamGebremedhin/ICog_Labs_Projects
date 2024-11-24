import os
import pandas as pd
from dotenv import load_dotenv
from langchain_astradb import AstraDBVectorStore
from astrapy.info import CollectionVectorServiceOptions
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document
from keybert import KeyBERT


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
        self.kw_model = KeyBERT("all-distilroberta-v1")
        

    def populate_db(self, file_path):
        print(f"Loading {file_path}...")

        loader = PyPDFLoader(file_path, extract_images=False)
        pages = loader.load()

        # Set chunk size to a maximum of 512 to avoid exceeding token size limit
        chunk_size = min(512, max(200, int(len(pages) * 0.1)))

        # Calculate chunk overlap based on chunk size, e.g., 5% of chunk size
        chunk_overlap = int(chunk_size * 0.05)

        # Split data into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            add_start_index=True,
        )

        chunks = text_splitter.split_documents([pages])

        vstore = AstraDBVectorStore(
            collection_name="Rejuve_Chat",
            api_endpoint=self.astra_db_api_endpoint,
            token=self.astra_db_application_token,
            namespace=None,
            collection_vector_service_options=self.nvidia_vectorize_options,
        )

        inserted_ids = vstore.add_documents(chunks)
        print(f"\nInserted {len(inserted_ids)} documents.")

    def populate_from_csv(self, csv_path, chunk_size=500, chunk_overlap=50):
        """
        Loads data from a CSV file into the vector database.
        The 'Text' column is used as the document content, while other columns become metadata.
        Text content is split into smaller chunks to meet token size requirements.
        
        Args:
            csv_path (str): Path to the CSV file
            chunk_size (int): Maximum size of text chunks
            chunk_overlap (int): Number of characters to overlap between chunks
        """
        print(f"Loading CSV data from {csv_path}...")
        
        try:
            # Read the CSV file
            df = pd.read_csv(csv_path)
            
            # Ensure 'Text' column exists
            if 'Text' not in df.columns:
                raise ValueError("CSV must contain a 'Text' column")
            
            # Initialize text splitter
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
                add_start_index=True,
            )
            
            # Create and split documents with metadata
            all_documents = []
            for _, row in df.iterrows():
                # Convert the row to a dictionary and remove the 'Text' field
                metadata = row.drop('Text').to_dict()
                
                # Handle potential list-like strings in metadata
                for key, value in metadata.items():
                    if isinstance(value, str) and value.startswith('[') and value.endswith(']'):
                        try:
                            # Convert string representation of list to actual list
                            metadata[key] = eval(value)
                        except:
                            # Keep as string if conversion fails
                            pass
                
                # Create a Document object
                doc = Document(
                    page_content=row['Text'],
                    metadata=metadata
                )
                
                # Split the document into chunks
                chunks = text_splitter.split_documents([doc])
                all_documents.extend(chunks)
            
            # Initialize vector store
            vstore = AstraDBVectorStore(
                collection_name="Rejuve_Chat",
                api_endpoint=self.astra_db_api_endpoint,
                token=self.astra_db_application_token,
                namespace=None,
                collection_vector_service_options=self.nvidia_vectorize_options,
            )
            
            # Add documents to the vector store
            inserted_ids = vstore.add_documents(all_documents)
            print(f"\nSplit text into {len(all_documents)} chunks and inserted them into the database.")
            
        except Exception as e:
            raise RuntimeError(f"Error loading CSV data: {e}")

    def reference_text(self, file_path):
        """
        Loads data from a CSV file into the vector database.
        The 'Text' column is used as the document content, while other columns become metadata.
        Text content is split into smaller chunks to meet token size requirements.
        
        Args:
            file_path (str): Path to the CSV file
        """

        print(f"Loading CSV data from {file_path}...")

        try:
            df = pd.read_csv(file_path)
            pages = []
            for row in df.iterrows():
                text = f"{row[1]['title']}' '{row[1]['abstract']}"
                pages.append(Document(page_content=text))
                text = ""

            # Calculate chunk size based on document length, e.g., 10% of document length
            chunk_size = min(512, max(200, int(len(pages) * 0.1)))

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

            print(f"Inserting {len(chunks)} chunks into the database...")

            vstore = AstraDBVectorStore(
                collection_name="Reference_Chat",
                api_endpoint=self.astra_db_api_endpoint,
                token=self.astra_db_application_token,
                namespace=None,
                collection_vector_service_options=self.nvidia_vectorize_options,
            )
            
            inserted_ids = vstore.add_documents(chunks)
            print(f"\nInserted {len(inserted_ids)} documents.")

        except Exception as e:
            raise RuntimeError(f"Error loading CSV data: {e}")

    def similarity_search(self, db_name, query: str, k_value: int = 10):
        """Queries the vector database and returns results"""
        try:
            vec_db = AstraDBVectorStore(
                collection_name=db_name,
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
        