from dotenv import load_dotenv
import os
import pandas as pd
from langchain_community.graphs import Neo4jGraph
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.documents import Document
from pathlib import Path

load_dotenv(Path(__file__).resolve().parents[1] / 'config.env')


class Neo4JChat:
    def __init__(self):
        self.NEO4J_URI = os.getenv('NEO4J_URI')
        self.AUTH = (os.getenv('NEO4J_USERNAME'), os.getenv('NEO4J_PASSWORD'))
        self.NEO4J_USERNAME = os.getenv('NEO4J_USERNAME')
        self.NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')
        self.NEO4J_DATABASE = os.getenv('NEO4J_DATABASE') or 'neo4j'

        self.embedder = OllamaEmbeddings(model="nomic-embed-text")
        self.kg = Neo4jGraph(
            url=self.NEO4J_URI, username=self.NEO4J_USERNAME, password=self.NEO4J_PASSWORD, database=self.NEO4J_DATABASE
        )

    def embed_text(self, text):
        return self.embedder.embed_query(text)

    def load_data(self):
        df = pd.read_csv("data.csv")

        docs = []
        for index, row in df.iterrows():
            print(f"Processing chunk {index + 1}/{len(df)}")
            docs.append(
                {
                    "id": index,
                    "text": str(row["Text"]),
                    "url": row["URL"],
                    "title": row["Title"],
                    "keywords": row["Keywords"].split(',')  # Assuming keywords are comma-separated
                }
            )

        merge_chunk_node_query = """
        MERGE (mergedChunk:Chunk {chunkId: $chunkParam.id})
        ON CREATE SET 
            mergedChunk.URL = $chunkParam.url,
            mergedChunk.Text = $chunkParam.text, 
            mergedChunk.Title = $chunkParam.title, 
            mergedChunk.Keywords = $chunkParam.keywords,
            mergedChunk.Embedding = $chunkParam.embedding
        RETURN mergedChunk
        """

        merge_keyword_node_query = """
        MERGE (keyword:Keyword {name: $keyword})
        ON CREATE SET keyword.Embedding = $embedding
        RETURN keyword
        """

        merge_url_node_query = """
        MERGE (url:URL {name: $url})
        ON CREATE SET url.Embedding = $embedding
        RETURN url
        """

        merge_title_node_query = """
        MERGE (title:Title {name: $title})
        ON CREATE SET title.Embedding = $embedding
        RETURN title
        """

        create_relationship_query = """
        MATCH (chunk:Chunk {chunkId: $chunkId}), (entity {name: $entity})
        MERGE (chunk)-[rel:HAS_ENTITY]->(entity)
        ON CREATE SET rel.type = $relationshipType
        """

        self.kg.query("""
        CREATE CONSTRAINT unique_chunk IF NOT EXISTS 
            FOR (c:Chunk) REQUIRE c.chunkId IS UNIQUE
        """)

        self.kg.query("""
        CREATE CONSTRAINT unique_keyword IF NOT EXISTS 
            FOR (k:Keyword) REQUIRE k.name IS UNIQUE
        """)

        self.kg.query("""
        CREATE CONSTRAINT unique_url IF NOT EXISTS 
            FOR (u:URL) REQUIRE u.name IS UNIQUE
        """)

        self.kg.query("""
        CREATE CONSTRAINT unique_title IF NOT EXISTS 
            FOR (t:Title) REQUIRE t.name IS UNIQUE
        """)

        node_count = 0
        relationship_count = 0
        for doc in docs:
            # Generate embedding for the chunk text
            chunk_embedding = self.embed_text(doc['text'])
            doc['embedding'] = chunk_embedding

            print(f"Creating `:Chunk` node for chunk ID {doc['id']}")
            self.kg.query(merge_chunk_node_query, params={'chunkParam': doc})
            node_count += 1

            # Handle keywords
            for keyword in doc['keywords']:
                keyword = keyword.strip()
                keyword_embedding = self.embed_text(keyword)
                print(f"Creating `:Keyword` node for keyword {keyword}")
                self.kg.query(merge_keyword_node_query, params={'keyword': keyword, 'embedding': keyword_embedding})

                print(f"Creating relationship for chunk ID {doc['id']} to keyword {keyword}")
                self.kg.query(create_relationship_query, params={
                    'chunkId': doc['id'],
                    'entity': keyword,
                    'relationshipType': 'HAS_KEYWORD'
                })
                relationship_count += 1

            # Handle URLs
            url_embedding = self.embed_text(doc['url'])
            print(f"Creating `:URL` node for URL {doc['url']}")
            self.kg.query(merge_url_node_query, params={'url': doc['url'], 'embedding': url_embedding})

            print(f"Creating relationship for chunk ID {doc['id']} to URL {doc['url']}")
            self.kg.query(create_relationship_query, params={
                'chunkId': doc['id'],
                'entity': doc['url'],
                'relationshipType': 'SOURCED_FROM'
            })
            relationship_count += 1

            # Handle titles
            title_embedding = self.embed_text(doc['title'])
            print(f"Creating `:Title` node for title {doc['title']}")
            self.kg.query(merge_title_node_query, params={'title': doc['title'], 'embedding': title_embedding})

            print(f"Creating relationship for chunk ID {doc['id']} to title {doc['title']}")
            self.kg.query(create_relationship_query, params={
                'chunkId': doc['id'],
                'entity': doc['title'],
                'relationshipType': 'TALKS_ABOUT'
            })
            relationship_count += 1

        print(
            f"Created {node_count} chunk nodes and {relationship_count} relationships with keywords, URLs, and titles")

    def similarity_search(self, query):
        # Embed the user query
        query_embedding = self.embed_text(query)

        # Query the Neo4j database to find the most relevant chunks
        search_query = """
                MATCH (chunk:Chunk)
                WITH chunk, gds.similarity.cosine(chunk.Embedding, $queryEmbedding) AS similarity
                RETURN chunk, similarity
                ORDER BY similarity DESC
                LIMIT 2
            """

        results = self.kg.query(search_query, params={'queryEmbedding': query_embedding})

        # Extract relevant information from the results
        response = []
        for result in results:
            chunk = result['chunk']
            similarity = result['similarity']
            document = Document(
                page_content=chunk['Text'],
                metadata={
                    'url': chunk['URL'],
                    'title': chunk['Title'],
                    'keywords': chunk['Keywords'],
                    'similarity': similarity
                }
            )
            response.append(document)

        return response
