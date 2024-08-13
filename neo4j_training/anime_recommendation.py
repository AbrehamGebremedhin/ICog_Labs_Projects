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

        # Initialize the driver without using 'with'
        self.driver = GraphDatabase.driver(self.NEO4J_URI, auth=self.AUTH)

        # Verify connectivity
        try:
            self.driver.verify_connectivity()
            logging.info("Successfully connected to Neo4J")
        except Exception as e:
            logging.error(f"Error connecting to Neo4J: {e}")
            self.close()

    def load_data(self):
        df = pd.read_csv('data.csv')

        # Initialize counters for nodes and relationships
        node_count = 0

        unique_genres = df['Genres'].unique()

        for genre in unique_genres:
            self.driver.execute_query(
                "CREATE (genre:Genre {name: $name})",
                name=genre
            )
            node_count += 1  # Increment node count for each genre

        for show_type in df['Type'].unique():
            self.driver.execute_query(
                "CREATE (type:Type {name: $name})",
                name=show_type
            )
            node_count += 1  # Increment node count for each type

        for producer in df['Producers'].unique():
            self.driver.execute_query(
                "CREATE (producer:Producer {name: $name})",
                name=producer
            )
            node_count += 1  # Increment node count for each producer

        for licensor in df['Licensors'].unique():
            self.driver.execute_query(
                "CREATE (licensor:Licensor {name: $name})",
                name=licensor
            )
            node_count += 1  # Increment node count for each licensor

        for studio in df['Studios'].unique():
            self.driver.execute_query(
                "CREATE (studio:Studio {name: $name})",
                name=studio
            )
            node_count += 1  # Increment node count for each studio

        for source in df['Source'].unique():
            self.driver.execute_query(
                "CREATE (source:Source {name: $name})",
                name=source
            )
            node_count += 1  # Increment node count for each source

        for index, row in df.iterrows():
            self.driver.execute_query(
                f"CREATE (anime:Anime {{mal_id: $id, title: $title, no_episodes: $no_episodes, duration: $duration, rating: $rating, ranked: $ranked}})",
                id=row['MAL_ID'],
                index=index,
                title=row['Name'],
                no_episodes=row['Episodes'],
                duration=row['Duration'],
                rating=row['Rating'],
                ranked=row['Ranked'],
                score=row['Score'],
            )
            node_count += 1  # Increment node count for each anime

        # Log the number of nodes created
        logging.info(f"Created {node_count} nodes in the Neo4j database")

    def create_relationships(self):
        df = pd.read_csv('data.csv')

        # Create relationships for Genres
        for index, row in df.iterrows():
            genres = row['Genres'].split(", ")
            for genre in genres:
                self.driver.execute_query(
                    """
                    MATCH (anime:Anime {mal_id: $mal_id}), (genre:Genre {name: $genre})
                    CREATE (anime)-[:HAS_GENRE]->(genre)
                    """,
                    mal_id=row['MAL_ID'],
                    genre=genre
                )

        # Create relationships for Producers
        for index, row in df.iterrows():
            producers = row['Producers'].split(", ")
            for producer in producers:
                self.driver.execute_query(
                    """
                    MATCH (anime:Anime {mal_id: $mal_id}), (producer:Producer {name: $producer})
                    CREATE (anime)-[:PRODUCED_BY]->(producer)
                    """,
                    mal_id=row['MAL_ID'],
                    producer=producer
                )

        # Create relationships for Licensors
        for index, row in df.iterrows():
            licensors = row['Licensors'].split(", ")
            for licensor in licensors:
                self.driver.execute_query(
                    """
                    MATCH (anime:Anime {mal_id: $mal_id}), (licensor:Licensor {name: $licensor})
                    CREATE (anime)-[:LICENSED_BY]->(licensor)
                    """,
                    mal_id=row['MAL_ID'],
                    licensor=licensor
                )

        # Create relationships for Studios
        for index, row in df.iterrows():
            studios = row['Studios'].split(", ")
            for studio in studios:
                self.driver.execute_query(
                    """
                    MATCH (anime:Anime {mal_id: $mal_id}), (studio:Studio {name: $studio})
                    CREATE (anime)-[:STUDIO_OF]->(studio)
                    """,
                    mal_id=row['MAL_ID'],
                    studio=studio
                )

        # Create relationships for Source
        for index, row in df.iterrows():
            source = row['Source']
            self.driver.execute_query(
                """
                MATCH (anime:Anime {mal_id: $mal_id}), (source:Source {name: $source})
                CREATE (anime)-[:SOURCE_OF]->(source)
                """,
                mal_id=row['MAL_ID'],
                source=source
            )

        logging.info("Created relationships in the Neo4j database")

    def close(self):
        # Close the driver
        if self.driver:
            self.driver.close()
            logging.info("Neo4J driver closed")


# Instantiate and run the Neo4JChat class
neo = Neo4JChat()
neo.load_data()
neo.create_relationships()
neo.close()
