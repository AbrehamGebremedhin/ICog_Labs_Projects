import os
import pandas as pd
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_astradb import AstraDBVectorStore
from astrapy.info import CollectionVectorServiceOptions
from langchain_text_splitters import RecursiveCharacterTextSplitter


class MettaCSV:
    def __init__(self) -> None:
        load_dotenv("config.env")

    def single_csv_file(self, file_path, collection_name):
        print(f"Loading {file_path}...")

        # Load the CSV file
        data = pd.read_csv(file_path)

        # Ensure the CSV file has a 'Text' column
        if 'Text' not in data.columns:
            raise ValueError("The CSV file must contain a 'Text' column.")

        # Define the chunking options
        chunk_size = 200  # or set dynamically
        chunk_overlap = 20

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            add_start_index=True,
        )

        chunks = []
        for _, row in data.iterrows():
            text = row['Text']
            metadata = row.drop('Text').to_dict()  # Convert the remaining columns to metadata
            
            # Create chunks from the text in the 'Text' column
            docs = [Document(page_content=text, metadata=metadata)]
            chunks.extend(text_splitter.split_documents(docs))

        # Load vector store
        nvidia_vectorize_options = CollectionVectorServiceOptions(
            provider="nvidia",
            model_name="NV-Embed-QA",
        )
        
        astra_db_api_endpoint = os.getenv("ASTRA_DB_API_ENDPOINT")
        astra_db_application_token = os.getenv(
            "ASTRA_DB_APPLICATION_TOKEN")

        vstore = AstraDBVectorStore(
            collection_name=collection_name,
            api_endpoint=astra_db_api_endpoint,
            token=astra_db_application_token,
            namespace=None,
            collection_vector_service_options=nvidia_vectorize_options,
        )

        inserted_ids = vstore.add_documents(chunks)
        print(f"\nInserted {len(inserted_ids)} documents.")

# Example usage
saba_csv = MettaCSV()
csv_file_path = r"D:\Projects\ICog_Labs_Projects\MettaChatBot\data.csv"
saba_csv.single_csv_file(file_path=csv_file_path, collection_name="metta_collection")
