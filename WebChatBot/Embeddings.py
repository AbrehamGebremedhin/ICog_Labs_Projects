import time
import torch
import torch.nn.functional as F
from pymongo import MongoClient
from transformers import AutoTokenizer, AutoModel, BartTokenizer, BartModel
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer


class EmbeddingModel:
    def __init__(self, text):
        self.input = text
        self.chunk_size = 512
        # Initialize MongoDB client and collection
        self.client = MongoClient("mongodb://localhost:27017/")
        self.db = self.client["embedding_models"]
        self.collection = self.db["comparisons"]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,
            chunk_overlap=32,
            length_function=len,
            is_separator_regex=False,
        )

    def int_float(self):
        tokenizer = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-large")
        model = AutoModel.from_pretrained("intfloat/multilingual-e5-large").to(self.device)

        all_embeddings = []

        chunks = self.text_splitter.split_text(self.input)

        print(f"Total chunks: {len(chunks)}")

        total = 0

        for chunk in chunks:
            # Tokenize the input texts
            batch_dict = tokenizer(chunk, max_length=512, padding=True, truncation=True, return_tensors='pt').to(
                self.device)

            total += time.process_time()

            with torch.no_grad():
                outputs = model(**batch_dict)
                embeddings = self.average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])

            # Normalize embeddings
            embeddings = F.normalize(embeddings, p=2, dim=1)
            all_embeddings.append(embeddings)
            print(f"Chunk processed in {time.process_time()}")

        # Combine all embeddings
        combined_embeddings = torch.cat(all_embeddings, dim=0).cpu()

        try:
            self.collection.insert_one(
                {
                    "model_id": "intfloat/multilingual-e5-large",
                    "embeddings": combined_embeddings.tolist(),
                    "process_time": total
                }
            )
            print("Embeddings successfully inserted into MongoDB")
        except Exception as e:
            print(f"Error inserting into MongoDB: {e}")

        return combined_embeddings

    @staticmethod
    def average_pool(last_hidden_state, attention_mask):
        """Average pooling of the last hidden state using attention mask."""
        mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        masked_last_hidden = last_hidden_state * mask
        summed_last_hidden = masked_last_hidden.sum(1)
        summed_mask = mask.sum(1)
        return summed_last_hidden / summed_mask

    def bert(self):
        tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
        model = AutoModel.from_pretrained("microsoft/codebert-base").to(self.device)

        all_embeddings = []
        chunks = self.text_splitter.split_text(self.input)

        print(f"Total chunks: {len(chunks)}")

        total_time = 0

        for chunk in chunks:
            # Tokenize the input texts
            batch_dict = tokenizer(chunk, max_length=512, padding=True, truncation=True, return_tensors='pt').to(
                self.device)

            start_time = time.process_time()

            with torch.no_grad():
                outputs = model(**batch_dict)
                embeddings = self.average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])

            # Normalize embeddings
            embeddings = F.normalize(embeddings, p=2, dim=1)
            all_embeddings.append(embeddings)

            end_time = time.process_time()
            total_time += end_time - start_time
            print(f"Chunk processed in {end_time - start_time} seconds")

        # Combine all embeddings
        combined_embeddings = torch.cat(all_embeddings, dim=0).cpu()

        try:
            self.collection.insert_one(
                {
                    "model_id": "microsoft/codebert-base",
                    "embeddings": combined_embeddings.tolist(),
                    "process_time": total_time
                }
            )
            print("Embeddings successfully inserted into MongoDB")
        except Exception as e:
            print(f"Error inserting into MongoDB: {e}")

        return combined_embeddings

    def jina(self):
        model = AutoModel.from_pretrained('jinaai/jina-embeddings-v2-base-en', trust_remote_code=True).to(self.device)

        all_embeddings = []
        chunks = self.text_splitter.split_text(self.input)

        print(f"Total chunks: {len(chunks)}")

        total_time = 0

        for chunk in chunks:
            start_time = time.process_time()

            with torch.no_grad():
                # Encode the chunk using the model's encode method
                embeddings = model.encode([chunk], device=self.device)

            # Convert embeddings to a tensor and ensure it is on the correct device
            embeddings = torch.tensor(embeddings).to(self.device)

            # Normalize embeddings
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            all_embeddings.append(embeddings)

            end_time = time.process_time()
            total_time += end_time - start_time
            print(f"Chunk processed in {end_time - start_time} seconds")

        # Combine all embeddings
        combined_embeddings = torch.cat(all_embeddings, dim=0).cpu()

        try:
            self.collection.insert_one(
                {
                    "model_id": "jinaai/jina-embeddings-v2-base-en",
                    "embeddings": combined_embeddings.tolist(),
                    "process_time": total_time
                }
            )
            print("Embeddings successfully inserted into MongoDB")
        except Exception as e:
            print(f"Error inserting into MongoDB: {e}")

        return combined_embeddings

    def mixed(self):
        model = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1", truncate_dim=self.chunk_size)

        all_embeddings = []
        chunks = self.text_splitter.split_text(self.input)

        print(f"Total chunks: {len(chunks)}")

        total_time = 0

        for chunk in chunks:
            start_time = time.process_time()

            with torch.no_grad():
                # Encode the chunk using the model's encode method
                embeddings = model.encode([chunk], device=self.device)

            # Convert embeddings to a tensor and ensure it is on the correct device
            embeddings = torch.tensor(embeddings).to(self.device)

            # Normalize embeddings
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            all_embeddings.append(embeddings)

            end_time = time.process_time()
            total_time += end_time - start_time
            print(f"Chunk processed in {end_time - start_time} seconds")

        # Combine all embeddings
        combined_embeddings = torch.cat(all_embeddings, dim=0).cpu()

        try:
            self.collection.insert_one(
                {
                    "model_id": "mixedbread-ai/mxbai-embed-large-v1",
                    "embeddings": combined_embeddings.tolist(),
                    "process_time": total_time
                }
            )
            print("Embeddings successfully inserted into MongoDB")
        except Exception as e:
            print(f"Error inserting into MongoDB: {e}")

        return combined_embeddings

    def bart(self):
        tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")
        model = BartModel.from_pretrained("facebook/bart-large").to(self.device)

        all_embeddings = []
        chunks = self.text_splitter.split_text(self.input)

        print(f"Total chunks: {len(chunks)}")

        total_time = 0

        for chunk in chunks:
            # Tokenize the input texts
            batch_dict = tokenizer(chunk, max_length=512, padding=True, truncation=True, return_tensors='pt').to(
                self.device)

            start_time = time.process_time()

            with torch.no_grad():
                outputs = model(**batch_dict)
                embeddings = outputs.last_hidden_state.mean(dim=1)  # Use mean pooling for embeddings

            # Normalize embeddings
            embeddings = F.normalize(embeddings, p=2, dim=1)
            all_embeddings.append(embeddings)

            end_time = time.process_time()
            total_time += end_time - start_time
            print(f"Chunk processed in {end_time - start_time} seconds")

        # Combine all embeddings
        combined_embeddings = torch.cat(all_embeddings, dim=0).cpu()

        try:
            self.collection.insert_one(
                {
                    "model_id": "facebook/bart-large",
                    "embeddings": combined_embeddings.tolist(),
                    "process_time": total_time
                }
            )
            print("Embeddings successfully inserted into MongoDB")
        except Exception as e:
            print(f"Error inserting into MongoDB: {e}")

        return combined_embeddings
