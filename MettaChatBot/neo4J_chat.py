from dotenv import load_dotenv
import os
import numpy as np
import pandas as pd
from langchain_community.graphs import Neo4jGraph
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.documents import Document
from pathlib import Path
import logging
from sklearn.metrics.pairwise import cosine_similarity

# Load environment variables
load_dotenv(Path(__file__).resolve().parents[1] / 'config.env')

# Initialize logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


class Neo4JChat:
    def __init__(self):
        self.embedder = OllamaEmbeddings(model="nomic-embed-text")
        self.NEO4J_URI = os.getenv('NEO4J_URI')
        self.NEO4J_USERNAME = os.getenv('NEO4J_USERNAME')
        self.NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')
        if not self.NEO4J_URI or not self.NEO4J_USERNAME or not self.NEO4J_PASSWORD:
            raise ValueError(
                "Missing one or more environment variables for Neo4j connection.")

        self.kg = Neo4jGraph(
            self.NEO4J_URI, username=self.NEO4J_USERNAME, password=self.NEO4J_PASSWORD)
        self.verify_connection()

    def verify_connection(self):
        try:
            self.kg.query("RETURN 1")
            logging.info("Successfully connected to the Neo4j database.")
        except Exception as e:
            logging.error(f"Failed to connect to the Neo4j database: {e}")
            if "DatabaseNotFound" in str(e):
                logging.info("Attempting to connect to the default database.")
                self.kg = Neo4jGraph(
                    url=self.NEO4J_URI, username=self.NEO4J_USERNAME, password=self.NEO4J_PASSWORD, database="neo4j"
                )
                try:
                    self.kg.query("RETURN 1")
                    logging.info(
                        "Successfully connected to the default Neo4j database.")
                except Exception as fallback_e:
                    logging.error(
                        f"Failed to connect to the default Neo4j database: {fallback_e}")
                    raise

    def embed_text(self, text):
        return self.embedder.embed_query(text)

    def load_data(self):
        # Update the path to use the uploaded file
        df = pd.read_csv("data.csv")

        docs = []
        for index, row in df.iterrows():
            logging.info(f"Processing chunk {index + 1}/{len(df)}")
            docs.append(
                {
                    "chunkId": index,  # Changed 'id' to 'chunkId'
                    "text": str(row["Text"]),
                    "url": row["URL"],
                    "title": row["Title"],
                    # Remove duplicates
                    "keywords": list(set([kw.strip() for kw in row["Keywords"].split(',')]))
                }
            )

        merge_queries = {
            "chunk": """
                MERGE (chunk:Chunk {chunkId: $chunkId})
                ON CREATE SET 
                    chunk.url = $url,
                    chunk.text = $text, 
                    chunk.title = $title, 
                    chunk.keywords = $keywords,
                    chunk.embedding = $embedding
            """,
            "entity": """
                MERGE (entity:Entity {name: $name})
                ON CREATE SET entity.embedding = $embedding
            """,
            "relationship": """
                MATCH (chunk:Chunk {chunkId: $chunkId})
                OPTIONAL MATCH (entity:Entity {name: $entityName})
                MERGE (chunk)-[rel:RELATIONSHIP_TYPE]->(entity)
                ON CREATE SET rel.type = $relationshipType, rel.created_at = timestamp()
            """
        }

        constraints = [
            "CREATE CONSTRAINT unique_chunk IF NOT EXISTS FOR (c:Chunk) REQUIRE c.chunkId IS UNIQUE",
            "CREATE CONSTRAINT unique_entity IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE"
        ]

        for constraint in constraints:
            self.kg.query(constraint)

        node_count = 0
        relationship_count = 0
        for doc in docs:
            # Generate embedding for the chunk text
            chunk_embedding = self.embed_text(doc['text'])
            doc['embedding'] = chunk_embedding

            logging.info(f"Creating `:Chunk` node for chunk ID {
                         doc['chunkId']}")
            self.kg.query(merge_queries['chunk'], params=doc)
            node_count += 1

            entities = {
                'keywords': 'HAS_KEYWORD',
                'url': 'SOURCED_FROM',
                'title': 'TALKS_ABOUT'
            }

            for entity_type, relationship_type in entities.items():
                entity_values = doc[entity_type]
                if entity_type == 'keywords':
                    for value in entity_values:
                        value_embedding = self.embed_text(value)
                        logging.info(
                            f"Creating `:Entity` node for keyword {value}")
                        self.kg.query(merge_queries['entity'], params={
                                      'name': value, 'embedding': value_embedding})
                        logging.info(f"Creating relationship for chunk ID {
                                     doc['chunkId']} to keyword {value}")
                        self.kg.query(merge_queries['relationship'], params={
                            'chunkId': doc['chunkId'], 'entityName': value, 'relationshipType': relationship_type
                        })
                        relationship_count += 1
                else:
                    value_embedding = self.embed_text(entity_values)
                    logging.info(f"Creating `:Entity` node for {
                                 entity_type} {entity_values}")
                    self.kg.query(merge_queries['entity'], params={
                                  'name': entity_values, 'embedding': value_embedding})
                    logging.info(f"Creating relationship for chunk ID {
                                 doc['chunkId']} to {entity_type} {entity_values}")
                    self.kg.query(merge_queries['relationship'], params={
                        'chunkId': doc['chunkId'], 'entityName': entity_values, 'relationshipType': relationship_type
                    })
                    relationship_count += 1

        logging.info(f"Created {node_count} chunk nodes and {
                     relationship_count} relationships with keywords, URLs, and titles")

    def similarity_search(self, query):
        # Embed the user query
        query_embedding = self.embed_text(query)

        # Query the Neo4j database to find the most relevant chunks
        search_query = """
            MATCH (chunk:Chunk)
            WITH chunk, gds.similarity.cosine(chunk.embedding, $queryEmbedding) AS similarity
            RETURN chunk, similarity
            ORDER BY similarity DESC
            LIMIT 4
        """

        results = self.kg.query(search_query, params={
                                'queryEmbedding': query_embedding})

        # Extract relevant information from the results
        response = []
        for result in results:
            chunk = result['chunk']
            similarity = result['similarity']
            document = Document(
                page_content=chunk['text'],
                metadata={
                    'url': chunk['url'],
                    'title': chunk['title'],
                    'keywords': chunk['keywords'],
                    'similarity': similarity
                }
            )
            response.append(document)

        return response
    def similarity_search(self, query):
            # Embed the user query
            query_embedding = self.embed_text(query)

            # Fetch all chunk embeddings from the Neo4j database
            fetch_embeddings_query = """
                MATCH (chunk:Chunk)
                RETURN chunk.chunkId AS chunkId, chunk.text AS text, chunk.embedding AS embedding, 
                    chunk.url AS url, chunk.title AS title, chunk.keywords AS keywords
            """

            results = self.kg.query(fetch_embeddings_query)

            # Prepare embeddings and metadata for similarity calculation
            embeddings = []
            metadata = []
            for result in results:
                embeddings.append(result['embedding'])
                metadata.append({
                    'chunkId': result['chunkId'],
                    'text': result['text'],
                    'url': result['url'],
                    'title': result['title'],
                    'keywords': result['keywords']
                })

            # Convert embeddings and query_embedding to numpy arrays
            embeddings = np.array(embeddings)
            query_embedding = np.array(query_embedding).reshape(1, -1)

            # Compute cosine similarity between query embedding and chunk embeddings
            similarities = cosine_similarity(query_embedding, embeddings).flatten()

            # Sort the results by similarity score in descending order
            sorted_indices = np.argsort(-similarities)
            top_results = sorted_indices[:4]  # Get the top 4 results

            # Create response documents
            response = []
            for index in top_results:
                doc_metadata = metadata[index]
                similarity_score = similarities[index]
                document = Document(
                    page_content=doc_metadata['text'],
                    metadata={
                        'url': doc_metadata['url'],
                        'title': doc_metadata['title'],
                        'keywords': doc_metadata['keywords'],
                        'similarity': similarity_score
                    }
                )
                response.append(document)

            return response
