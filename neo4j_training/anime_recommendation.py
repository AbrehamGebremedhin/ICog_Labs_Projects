import os
import logging
import pandas as pd
from neo4j import GraphDatabase
from dotenv import load_dotenv

# Load environment variables
load_dotenv('config.env')

# Initialize logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

class Neo4JChat:
    def __init__(self):
        self.NEO4J_URI = os.getenv('NEO4J_URI')
        self.NEO4J_USERNAME = os.getenv('NEO4J_USERNAME')
        self.NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')
        self.AUTH = (self.NEO4J_USERNAME, self.NEO4J_PASSWORD)

        # Initialize the driver
        self.driver = GraphDatabase.driver(self.NEO4J_URI, auth=self.AUTH)

        # Verify connectivity
        try:
            self.driver.verify_connectivity()
            logging.info("Successfully connected to Neo4J")
        except Exception as e:
            logging.error(f"Error connecting to Neo4J: {e}")
            self.close()

    def load_and_create_data(self):
        df = pd.read_csv('data.csv')

        # Initialize counters for nodes and relationships
        node_count = 0
        relationship_count = 0

        for index, row in df[:300].iterrows():
            # Create Anime node with a unique variable name
            anime_var = f"anime_{row['MAL_ID']}"
            self.driver.execute_query(
                f"MERGE ({anime_var}:Anime {{mal_id: $id, title: $title, no_episodes: $no_episodes, duration: $duration, rating: $rating, ranked: $ranked}})",
                id=row['MAL_ID'],
                title=row['Name'],
                no_episodes=row['Episodes'],
                duration=row['Duration'],
                rating=row['Rating'],
                ranked=row['Ranked'],
                score=row['Score'],
            )
            logging.info(f"Created Anime node: {row['Name']} (ID: {row['MAL_ID']})")
            node_count += 1

            # Create and relate Genre nodes
            genres = row['Genres'].split(", ")
            for genre in genres:
                genre_var = f"genre_{genre.replace(' ', '_')}"
                genre_var = f"genre_{genre_var.replace('-', '_')}"
                genre_var = f"genre_{genre_var.replace('.', '_')}"
                genre_var = f"genre_{genre_var.replace('&', '_')}"

                self.driver.execute_query(
                    f"MERGE ({genre_var}:Genre {{name: $name}})", name=genre
                )
                self.driver.execute_query(
                    f"""
                    MATCH (anime:Anime {{mal_id: $mal_id}})
                    MERGE (genre:Genre {{name: $genre}})
                    MERGE (anime)-[:HAS_GENRE]->(genre)
                    """,
                    mal_id=row['MAL_ID'],
                    genre=genre
                )
                logging.info(f"Created HAS_GENRE relationship between {row['Name']} and Genre: {genre}")
                relationship_count += 1

            # Create and relate Type nodes with a rating property in the relationship
            show_type = row['Type']
            type_var = f"type_{show_type.replace(' ', '_')}"
            type_var = f"type_{type_var.replace('-', '_')}"
            type_var = f"type_{type_var.replace('.', '_')}"
            type_var = f"type_{type_var.replace('&', '_')}"

            self.driver.execute_query(
                f"MERGE ({type_var}:Type {{name: $name}})", name=show_type
            )
            self.driver.execute_query(
                f"""
                MATCH (anime:Anime {{mal_id: $mal_id}})
                MERGE (type:Type {{name: $type}})
                MERGE (anime)-[:OF_TYPE {{rating: $rating}}]->(type)
                """,
                mal_id=row['MAL_ID'],
                type=show_type,
                rating=row['Rating']
            )
            logging.info(f"Created OF_TYPE relationship between {row['Name']} and Type: {show_type}")
            relationship_count += 1

            # Create and relate Producer nodes
            producers = row['Producers'].split(", ")
            for producer in producers:
                producer_var = f"producer_{producer.replace(' ', '_')}"
                producer_var = f"producer_{producer_var.replace('-', '_')}"
                producer_var = f"producer_{producer_var.replace('.', '_')}"
                producer_var = f"producer_{producer_var.replace('&', '_')}"

                self.driver.execute_query(
                    f"MERGE ({producer_var}:Producer {{name: $name}})", name=producer
                )
                self.driver.execute_query(
                    f"""
                    MATCH (anime:Anime {{mal_id: $mal_id}})
                    MERGE (producer:Producer {{name: $producer}})
                    MERGE (anime)-[:PRODUCED_BY]->(producer)
                    """,
                    mal_id=row['MAL_ID'],
                    producer=producer
                )
                logging.info(f"Created PRODUCED_BY relationship between {row['Name']} and Producer: {producer}")
                relationship_count += 1

            # Create and relate Licensor nodes
            licensors = row['Licensors'].split(", ")
            for licensor in licensors:
                licensor_var = f"licensor_{licensor.replace(' ', '_')}"
                licensor_var = f"licensor_{licensor_var.replace('-', '_')}"
                licensor_var = f"licensor_{licensor_var.replace('.', '_')}"
                licensor_var = f"licensor_{licensor_var.replace('&', '_')}"

                self.driver.execute_query(
                    f"MERGE ({licensor_var}:Licensor {{name: $name}})", name=licensor
                )
                self.driver.execute_query(
                    f"""
                    MATCH (anime:Anime {{mal_id: $mal_id}})
                    MERGE (licensor:Licensor {{name: $licensor}})
                    MERGE (anime)-[:LICENSED_BY]->(licensor)
                    """,
                    mal_id=row['MAL_ID'],
                    licensor=licensor
                )
                logging.info(f"Created LICENSED_BY relationship between {row['Name']} and Licensor: {licensor}")
                relationship_count += 1

            # Create and relate Studio nodes
            studios = row['Studios'].split(", ")
            for studio in studios:
                studio_var = f"studio_{studio.replace(' ', '_')}"
                studio_var = f"studio_{studio_var.replace('-', '_')}"
                studio_var = f"studio_{studio_var.replace('.', '_')}"
                studio_var = f"studio_{studio_var.replace('&', '_')}"

                self.driver.execute_query(
                    f"MERGE ({studio_var}:Studio {{name: $name}})", name=studio
                )
                self.driver.execute_query(
                    f"""
                    MATCH (anime:Anime {{mal_id: $mal_id}})
                    MERGE (studio:Studio {{name: $studio}})
                    MERGE (anime)-[:STUDIO_OF]->(studio)
                    """,
                    mal_id=row['MAL_ID'],
                    studio=studio
                )
                logging.info(f"Created STUDIO_OF relationship between {row['Name']} and Studio: {studio}")
                relationship_count += 1

            # Create and relate Source nodes
            source = row['Source']
            source_var = f"source_{source.replace(' ', '_')}"
            source_var = f"source_{source_var.replace('-', '_')}"
            source_var = f"source_{source_var.replace('.', '_')}"
            source_var = f"source_{source_var.replace('&', '_')}"

            self.driver.execute_query(
                f"MERGE ({source_var}:Source {{name: $name}})", name=source
            )
            self.driver.execute_query(
                f"""
                MATCH (anime:Anime {{mal_id: $mal_id}})
                MERGE (source:Source {{name: $source}})
                MERGE (anime)-[:SOURCE_OF]->(source)
                """,
                mal_id=row['MAL_ID'],
                source=source
            )
            logging.info(f"Created SOURCE_OF relationship between {row['Name']} and Source: {source}")
            relationship_count += 1

        # Log the number of nodes and relationships created
        logging.info(f"Created {node_count} nodes and {relationship_count} relationships in the Neo4j database")

    def close(self):
        # Close the driver
        if self.driver:
            self.driver.close()
            logging.info("Neo4J driver closed")


# Instantiate and run the Neo4JChat class
neo = Neo4JChat()
neo.load_and_create_data()
neo.close()
