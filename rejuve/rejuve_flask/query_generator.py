import os
import json
import requests
from dotenv import load_dotenv
from neo4j import GraphDatabase
from langchain_ollama import OllamaLLM
from langchain_core.output_parsers import JsonOutputParser

load_dotenv(r'D:\Projects\ICog_Labs_Projects\rejuve\config.env')    
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
        json_format = """{
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
            },

        }"""

        annotation_node_without =  """{ 
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
        }"""

        annotation_node_with =  """{ 
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
        }"""

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
        
        else:
            return f"Error: {response.status_code} - {response.text}"
