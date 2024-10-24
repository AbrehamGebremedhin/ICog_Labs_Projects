import os
from dotenv import load_dotenv
from neo4j import GraphDatabase
from langchain_ollama import OllamaLLM

load_dotenv(r'D:\Projects\ICog_Labs_Projects\rejuve\config.env')

class Neo4jDatabase:
    def __init__(self):
        NEO4J_URI = os.getenv('NEO4J_URI')
        NEO4J_USERNAME = os.getenv('NEO4J_USERNAME')
        NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')
        self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
        
        # Initialize cache
        self.labels = None
        self.property_keys = None
        self.relationships = None
        self.sample_data = None
        
        # Populate cache
        self._populate_cache()

    def _populate_cache(self):
        self.labels, self.relationships, self.property_keys = self._fetch_labels_propertykeys_and_relationships()
        self.sample_data = self._fetch_sample_data_each_node()

    def close(self):
        self.driver.close()

    def execute_query(self, query, params=None):
        with self.driver.session() as session:
            result = session.run(query, params)
            return [record.data() for record in result]
    
    def _fetch_labels_propertykeys_and_relationships(self):
        with self.driver.session() as session:
            labels_result = session.run("CALL db.labels")
            labels = [record["label"] for record in labels_result]
            
            property_keys_result = session.run("CALL db.propertyKeys")
            property_keys = [record["propertyKey"] for record in property_keys_result]

            relationships_result = session.run("CALL db.relationshipTypes")
            relationships = [record["relationshipType"] for record in relationships_result]
            
        return labels, relationships, property_keys
    
    def _fetch_sample_data_each_node(self):
        with self.driver.session() as session:
            sample_data = []
            labels = self.labels
            for label in labels:
                query = f"MATCH (n:{label}) RETURN n LIMIT 1"
                result = session.run(query)
                sample_data.extend([record.data() for record in result])
        return sample_data

class LangchainToCypher:
    def __init__(self, neo4j_db):
        self.llm = OllamaLLM(model="llama3.1")
        self.neo4j_db = neo4j_db

    def _format_sample_data(self, sample_data):
        formatted_samples = []
        for node in sample_data:
            if node and 'n' in node:  # Ensure node and 'n' key exists
                properties = node['n']
                formatted_samples.append(str(properties))
        return "\n".join(formatted_samples)

    def generate_cypher_query(self, user_input):
        # Format the sample data
        sample_data_str = self._format_sample_data(self.neo4j_db.sample_data)

        sample_queries = {
            "Get all genes with a specific type": 
                "MATCH (g:gene {gene_type: 'protein_coding'}) RETURN g.gene_name",

            "Get all transcripts associated with a specific gene": 
                "MATCH (g:gene {gene_name: 'BRCA1'})-[:transcribed_to]->(t:transcript) RETURN t",

            "Find all exons within a specific transcript": 
                "MATCH (t:transcript {transcript_id: 'ENST00000367770'})-[:includes]->(e:exon) RETURN e",

            "Find the pathway a specific gene is involved in": 
                "MATCH (g:gene {gene_name: 'TP53'})-[:associated_with]->(p:pathway) RETURN p",

            "Get genes and their associated pathways": 
                "MATCH (g:gene)-[:associated_with]->(p:pathway) RETURN g, p",

            "Find all promoters related to a specific transcript": 
                "MATCH (t:transcript {transcript_id: 'ENST00000456328'})-[:includes]->(pr:promoter) RETURN pr",

            "Find the relationship between a transcript and its translated protein": 
                "MATCH (t:transcript {transcript_id: 'ENST00000488147'})-[:translates_to]->(p:protein) RETURN p",

            "Find super enhancers related to a specific gene": 
                "MATCH (g:gene {gene_name: 'MYC'})-[:associated_with]->(se:super_enhancer) RETURN se",

            "Find all transcripts transcribed from a specific gene": 
                "MATCH (g:gene {gene_id: 'ENSG00000139618'})-[:transcribed_to]->(t:transcript) RETURN t",

            "Find the gene with the highest number of associated transcripts": 
                "MATCH (g:gene)-[:transcribed_to]->(t:transcript) RETURN g, COUNT(t) AS transcript_count ORDER BY transcript_count DESC LIMIT 1",

            "Get all genes transcribed into a specific protein": 
                "MATCH (g:gene)-[:transcribed_to]->(:transcript)-[:translates_to]->(p:protein {name: 'P53'}) RETURN g",

            "Find all genes involved in multiple pathways": 
                "MATCH (g:gene)-[:associated_with]->(p:pathway) WITH g, COUNT(p) AS pathway_count WHERE pathway_count > 1 RETURN g",

            "Show the hierarchy between genes, transcripts, and exons": 
                "MATCH (g:gene)-[:transcribed_to]->(t:transcript)-[:includes]->(e:exon) RETURN g, t, e",

            "Find the biological context of a specific gene": 
                "MATCH (g:gene {gene_name: 'BRCA2'})-[:biological_context]->(bc) RETURN bc",

            "List genes, their transcripts, and the resulting proteins": 
                "MATCH (g:gene)-[:transcribed_to]->(t:transcript)-[:translates_to]->(p:protein) RETURN g, t, p",

            "Identify genes without any transcripts": 
                "MATCH (g:gene) WHERE NOT (g)-[:transcribed_to]->(:transcript) RETURN g",

            "Check the number of exons per transcript": 
                "MATCH (t:transcript)-[:includes]->(e:exon) RETURN t, COUNT(e) AS exon_count",

            "Find all transcripts associated with genes on chromosome 12": 
                "MATCH (g:gene {chr: 'chr12'})-[:transcribed_to]->(t:transcript) RETURN t",

            "Get all genes based on start and end positions": 
                "MATCH (g:gene) WHERE g.start >= 50000 AND g.end <= 150000 AND g.chr = 'chr1' RETURN g",

            "Find all transcripts with a specific alternative name": 
                "MATCH (t:transcript {transcript_name: 'lncRNA'}) RETURN t"
        }

        
        # Create a comprehensive prompt using actual database metadata
        prompt = f"""
        You are an expert in translating natural language to Cypher queries for a Neo4j database containing a biological dataset. Generate ONLY the Cypher query without any explanations or markdown formatting.

        Database Schema:
        - Nodes: {', '.join(self.neo4j_db.labels)}
        - Relationships: {', '.join(self.neo4j_db.relationships)}
        - Properties: {', '.join(self.neo4j_db.property_keys)}
        
        Sample Data:
        {sample_data_str}

        Sample Queries:
        {sample_queries}
        
        Rules for query generation:
        1. Return ONLY the Cypher query - no explanations, no markdown
        2. Use double quotes for string literals
        3. Keep the query simple and direct
        4. Do not include any comments or additional text
        5. Do not wrap the query in backticks
        
        User Request: {user_input}
        """
        
        response = self.llm(prompt)
        return response

def main():
    # Initialize Neo4j database connection
    neo4j_db = Neo4jDatabase()

    # Initialize Langchain with Ollama and pass neo4j_db instance
    langchain_to_cypher = LangchainToCypher(neo4j_db)

    # User input example
    user_input = "Find all genes of type 'processed_pseudogene' and return their names."

    # Generate Cypher query using LangChain
    cypher_query = langchain_to_cypher.generate_cypher_query(user_input)
    print(f"Generated Cypher Query: {cypher_query}")

    # Close the Neo4j connection
    neo4j_db.close()

if __name__ == "__main__":
    main()