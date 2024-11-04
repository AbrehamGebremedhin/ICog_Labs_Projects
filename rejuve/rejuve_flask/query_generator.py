import os
import json
import requests
from dotenv import load_dotenv
from neo4j import GraphDatabase
from langchain_ollama import OllamaLLM
from langchain_core.output_parsers import JsonOutputParser

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
    def __init__(self):
        self.llm = OllamaLLM(model="llama3.1")
        self.neo4j_db = Neo4jDatabase()

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
                "MATCH (g:gene {gene_type: 'protein_coding'}) RETURN g",

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
        - Graph: {self.neo4j_db}
        - Nodes: {', '.join(self.neo4j_db.labels)}
        - Relationships: {', '.join(self.neo4j_db.relationships)}
        - Properties: {', '.join(self.neo4j_db.property_keys)}
        
        Sample Data:
        {sample_data_str}

        Sample Queries:
        {sample_queries}
        
        Rules for query generation:
        1. Include related nodes and relationships where applicable.
        2. Only use MATCH and RETURN clauses
        3. Return ONLY the Cypher query - no explanations, no markdown
        4. Use double quotes for string literals
        5. Keep the query simple and direct
        6. Do not include any comments or additional text
        7. Do not wrap the query in backticks
        
        User Request: {user_input}
        """
        
        response = self.llm.invoke(prompt)

        return response
        
    def run_query(self, query):
        cypher_query = self.generate_cypher_query(query)

        result = self.neo4j_db.execute_query(cypher_query)

        self.neo4j_db.close()

        return result
    
class NaturalToAnnotation:
    def __init__(self):
        self.llm = OllamaLLM(model="llama3.1")
        self.parser = JsonOutputParser()
        self.nodes = [
            "enhancer",
            "exon",
            "gene",
            "pathway",
            "promoter",
            "protein",
            "snp",
            "super_enhancer",
            "transcript"
        ]

        self.relationship_mapping = [
            "'super_enhancer' to 'gene' have 'associated_with' relationship",
            "'promoter' to 'gene' have 'associated_with' relationship",
            "'transcript' to 'exon' have 'includes' relationship",
            "'transcript' to 'gene' have 'transcribed_from' relationship",
            "'gene' to 'transcript' have 'transcribed_to' relationship",
            "'transcript' to 'protein' have 'translates_to' relationship",
            "'protein' to 'transcript' have 'translation_of' relationship"
        ]

        self.property_keys = {
            "enhancer_property_keys": [
                "id", 
                "start", 
                "end", 
                "chr"
            ],
            "exon_property_keys": [
                "id", 
                "start", 
                "end", 
                "chr", 
                "exon_number", 
                "exon_id"
            ],
            "gene_property_keys": [
                "id", 
                "gene_name", 
                "gene_type", 
                "synonyms", 
                "start", 
                "end", 
                "chr"
            ],
            "pathway_property_keys": [
                "id",
                "pathway_name"
            ],
            "promoter_property_keys": [
                "id", 
                "start", 
                "end", 
                "chr"
            ],
            "protein_property_keys": [
                "id", 
                "protein_name", 
                "accessions"
            ],
            "snp_property_keys": [
                "id", 
                "start", 
                "end", 
                "chr", 
                "ref", 
                "caf_ref", 
                "alt", 
                "caf_alt"
            ],
            "superhancer_property_keys": [
                "id", 
                "start", 
                "end", 
                "chr", 
                "se_id"
            ],
            "transcript_property_keys": [
                "id", 
                "start",
                "end", 
                "chr", 
                "transcript_id", 
                "transcript_name", 
                "transcript_type", 
                "label"
            ]

        }


    def annotation_service_format(self, query):
        """
        Formats the user query into the annotation service format.

        Args:
            query (str): The user query to be formatted.

        Returns:
            str: The formatted query.
        """
        json_format = {
            # anotation format for 2 nodes with relationship
            "requests": { # An object containing the nodes and predicates arrays.
                "nodes": [ # (Mandatory) A list of node objects that define the nodes to query.
                    {
                        "node_id": "n1", # (Mandatory) A unique identifier for the node within the query context.
                        "id": "", # (Mandatory Key, Optional Value) The key id must be present and can contain either an Ensemble ID or a HUGO ID. Its value can be an empty string if you do not have a specific identifier.
                        "type": "gene", # (Mandatory) The type of the node (gene, transcript, enhancer, exon, pathway, promoter, protein, snp and super_enhancer).
                        "properties": {
                            "gene_type": "protein_coding" # (Mandatory key) A dictionary of properties to match for the node. The specific properties required depend on the node type.
                        }
                    },
                    {
                        "node_id": "n2", # (Mandatory) A unique identifier for the node within the query context.
                        "id": "", # (Mandatory Key, Optional Value) The key id must be present and can contain either an Ensemble ID or a HUGO ID. Its value can be an empty string if you do not have a specific identifier.
                        "type": "transcript", # (Mandatory) The type of the node (gene, transcript, enhancer, exon, pathway, promoter, protein, snp and super_enhancer).
                        "properties": {} # (Mandatory key) A dictionary of properties to match for the node. The specific properties required depend on the node type.
                    }
                    ],
                    "predicates": [ # (Optional) A list of relationship objects (edges) that define the relationships to query between the nodes.
                    {
                        "type": "transcribed to", # (Mandatory) The type of relationship (e.g., transcribed to, translates_to, associated_with, includes, transcribed_from, translation_of).
                        "source": "n1", # (Mandatory) The node_id of the source node in the relationship.
                        "target": "n2" # (Mandatory) The node_id of the target node in the relationship.
                    }
                ]
            }
        }

        annotation_node_without =  { 
            # anotation format for node without fitering with any node property
            "requests": {
                "nodes": [
                    {
                        "node_id": "n1", 
                        "id": "", 
                        "type": "protein", 
                        "properties": {}
                    }
                ], 
                "predicates": []
            }
        }

        annotation_node_with =  { 
            # anotation format for node with fitering with node property       
            "requests": {
                "nodes": [
                {
                    "node_id": "n1",
                    "id": "",
                    "type": "gene",
                    "properties": {
                    "gene_type": "protein_coding"
                    }
                }
                ],
                "predicates": []
            }
        }

        prompt = f"""
            You are an expert in translating natural language to a JSON format for an annotation service that queries a biological database. Generate the JSON format based on the user query.
            
            Database Schema:
            - Nodes: {self.nodes}
            - Relationships: {self.relationship_mapping}
            - Properties: {self.property_keys}

            JSON Format for Two nodes with relatioship: {json_format}

            JSON Format for single node without property filtering: {annotation_node_without}
            
            JSON Format for single node with property filtering: {annotation_node_with}

            User Request: {query}

            Rules for JSON generation:

            1. Include the nodes and predicates arrays.

            2. Define the nodes with unique node_id, type, and properties.

            3. Define the predicates with type, source, and target.

            4. Use double quotes for all variables and their corresponding values.

            5. Return ONLY the JSON - no explanations, no markdown.

            6. Do not include any comments or additional text.

            7. Do not wrap the JSON in backticks.

      
        """

        response = self.llm.invoke(prompt)

        formatted_response = self.parser.parse(response)

        return formatted_response

    def run_query(self, query):
        """
        Executes a query against the annotation service.

        Args:
            query (str): The user query to be processed.

        Returns:
            dict: The JSON response from the annotation service if the request is successful.
        """
        request = self.annotation_service_format(query)

        response = requests.post("http://127.0.0.1:5000/query?properties=true&limit=10", json=request, headers={'Content-Type': 'application/json'})

        if response.status_code == 200:
            data = response.json()

            return data
