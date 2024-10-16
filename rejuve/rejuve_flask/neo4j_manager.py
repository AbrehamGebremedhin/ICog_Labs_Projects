import os
import PyPDF2
import logging
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.graphs import Neo4jGraph
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

# Load environment variables
load_dotenv(r'D:\Projects\ICog_Labs_Projects\rejuve\config.env')

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

    def load_csv(self):
        # Update the path to use the uploaded file
        df = pd.read_csv(r"D:\Projects\ICog_Labs_Projects\rejuve\data.csv")

        docs = []
        for index, row in df[557:].iterrows():
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

    def load_pdf(self, pdf_path):
        # Load the PDF file
        loader = PyPDFLoader(pdf_path, extract_images=False)

        text = loader.load()

        # Calculate chunk size based on document length, e.g., 10% of document length
        chunk_size = max(200, int(len(text) * 0.1))

        # Calculate chunk overlap based on chunk size, e.g., 5% of chunk size
        chunk_overlap = int(chunk_size * 0.05)

        # Split data into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            add_start_index=True,
        )

        # Split text into chunks
        chunks = text_splitter.split_documents(text)

        docs = []
        for index, chunk in enumerate(chunks):
            logging.info(f"Processing chunk {index + 1}/{len(chunks)}")
            # Generate embedding for the chunk
            chunk_embedding = self.embed_text(chunk)

            # Add default metadata (could be updated later if needed)
            docs.append(
                {
                    "chunkId": index,
                    "text": chunk,
                    "embedding": chunk_embedding,
                    "url": "https://www.rejuve.bio/_files/ugd/6135b4_e34170afbd574df5b24ab1eef9e3b31a.pdf",
                    "title": "rejuve bio",
                    "keywords": []
                }
            )

        df = pd.DataFrame(docs)
        df.to_csv(r"D:\Projects\ICog_Labs_Projects\rejuve\data.csv", index=False)

        merge_queries = {
            "chunk": """
                MERGE (chunk:Chunk {chunkId: $chunkId})
                ON CREATE SET 
                    chunk.text = $text,
                    chunk.embedding = $embedding,
                    chunk.url = $url,
                    chunk.title = $title,
                    chunk.keywords = $keywords
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

        # Create constraints for unique chunkId if not already existing
        constraints = [
            "CREATE CONSTRAINT unique_chunk IF NOT EXISTS FOR (c:Chunk) REQUIRE c.chunkId IS UNIQUE",
            "CREATE CONSTRAINT unique_entity IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE"
        ]
        for constraint in constraints:
            self.kg.query(constraint)

        node_count = 0
        relationship_count = 0
        for doc in docs:
            # Merge the chunk node with its text, embedding, and metadata
            self.kg.query(merge_queries['chunk'], params=doc)
            node_count += 1

            # Handle relationships (e.g., keywords, title, etc.)
            entities = {
                'keywords': 'HAS_KEYWORD',
                'url': 'SOURCED_FROM',
                'title': 'TALKS_ABOUT'
            }

            for entity_type, relationship_type in entities.items():
                entity_values = doc[entity_type]
                if entity_type == 'keywords' and isinstance(entity_values, list):
                    for value in entity_values:
                        value_embedding = self.embed_text(value)
                        logging.info(
                            f"Creating `:Entity` node for keyword {value}")
                        self.kg.query(merge_queries['entity'], params={
                            'name': value, 'embedding': value_embedding
                        })
                        logging.info(f"Creating relationship for chunk ID {
                                     doc['chunkId']} to keyword {value}")
                        self.kg.query(merge_queries['relationship'], params={
                            'chunkId': doc['chunkId'], 'entityName': value, 'relationshipType': relationship_type
                        })
                        relationship_count += 1
                elif entity_values:  # Only create relationships if the entity value exists
                    value_embedding = self.embed_text(entity_values)
                    logging.info(f"Creating `:Entity` node for {
                                 entity_type} {entity_values}")
                    self.kg.query(merge_queries['entity'], params={
                        'name': entity_values, 'embedding': value_embedding
                    })
                    logging.info(f"Creating relationship for chunk ID {
                                 doc['chunkId']} to {entity_type} {entity_values}")
                    self.kg.query(merge_queries['relationship'], params={
                        'chunkId': doc['chunkId'], 'entityName': entity_values, 'relationshipType': relationship_type
                    })
                    relationship_count += 1

        logging.info(f"Loaded {node_count} chunks from the PDF and created {
                     relationship_count} relationships.")

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


neo = Neo4JChat()
print(neo.similarity_search(query="who is the CEO?"))
