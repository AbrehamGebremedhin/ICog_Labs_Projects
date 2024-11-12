from functools import lru_cache
from langchain.schema import Document
from vec_db_manager import Vec_Astradb
from langchain.prompts import PromptTemplate
from langchain_ollama import OllamaEmbeddings
from cypher_generator import LangchainToCypher
from query_generator import NaturalToAnnotation
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import JsonOutputParser
from langchain.chains.combine_documents import create_stuff_documents_chain

class Chat:
    def __init__(self, db_type='graph'):
        """
        Initialize the Chat class with nodes, relationships, and property keys.
        :param db_type: 'graph' for Neo4j, 'vector' for Vec_Astradb, or 'annotation' for NaturalToAnnotation
        """
        self.embeddings = None
        self.llm = None
        self.session_chat = []
        self.db_type = db_type

        # Set the database manager based on db_type
        if self.db_type == 'graph':
            self.db_manager = LangchainToCypher()
        elif self.db_type == "annotation":
            self.db_manager = NaturalToAnnotation()
        elif self.db_type == 'vector':
            self.db_manager = Vec_Astradb()
        else:
            raise ValueError("Invalid db_type. Choose 'graph', 'annotation', or 'vector'.")

        self.parser = JsonOutputParser()

        # Define nodes, relationships, and property keys
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
            "enhancer_property_keys": ["id", "start", "end", "chr"],
            "exon_property_keys": ["id", "start", "end", "chr", "exon_number", "exon_id"],
            "gene_property_keys": ["id", "gene_name", "gene_type", "synonyms", "start", "end", "chr"],
            "pathway_property_keys": ["id", "pathway_name"],
            "promoter_property_keys": ["id", "start", "end", "chr"],
            "protein_property_keys": ["id", "protein_name", "accessions"],
            "snp_property_keys": ["id", "start", "end", "chr", "ref", "caf_ref", "alt", "caf_alt"],
            "superhancer_property_keys": ["id", "start", "end", "chr", "se_id"],
            "transcript_property_keys": ["id", "start", "end", "chr", "transcript_id", "transcript_name", "transcript_type", "label"]
        }

        self.preffered_answer_format = {
            "answer": "The annotations reveal: The gene AKAP17A is located on both the X and Y chromosomes and has three transcript variants: AKAP17A-201, AKAP17A-202, and AKAP17A-203.n- The transcripts are transcribed from the gene and translated into proteins, with AKAP17A-201 and AKAP17A-202 being protein-coding transcripts and AKAP17A-203 being a nonsense-mediated decay transcript.n- The proteins translated from these transcripts are involved in various biological processes and pathways, including cellular signaling and regulation of gene expression.n- Mutations or abnormal expression of AKAP17A have been associated with diseases such as schizophrenia, bipolar disorder, and autism spectrum disorder.n- Studies have identified genetic markers within the AKAP17A gene that are associated with an increased risk of developing these diseases.n- The regulatory elements, such as promoters and enhancers, play a crucial role in controlling the expression of the AKAP17A gene and its transcripts.n- These regulatory elements interact with transcription factors and other proteins to modulate gene expression in response to cellular signals and environmental cues.n- The complex interplay between the AKAP17A gene, its transcripts, proteins, and regulatory elements highlights the intricate nature of gene regulation and its impact on human health and disease.",
            "suggestion": "To explore further, consider expanding the search to include nearby regulatory elements such as enhancers or promoters and their associations with these transcripts. Additionally, investigate the connections between these transcripts and proteins to gain insights into their functional roles and potential involvement in biological pathways or disease mechanisms."
        }

        self.example_queries = """
            # Example query
            #* Query one node types
            query = "Show me all genes stored in the database."
            query = "Retrieve a list of all transcripts available."
            query = "Can you provide a summary of all proteins in the database?"
            query = "List all enhancers in the knowledge base"
            query = "Display all enhancers that are recorded in the system."

            #* Specific node with id
            query = "What properties does the gene ensg00000232448 has?"
            query = "Give me the properties of the transcript enst00000437504."
            query = "What are the attributes of the protein p78504?"
            query = "Fetch the information associated with the exon ense00003875467."
            query = "What detail is available for the enhancer chr1_203097061_203097400_grch38?"

            #* Specific nodes with a property
            query = "Which genes are classified as protein coding?"
            query = "List all transcripts that are of type lncRNA."
            query = "Show me exons located on chromosome 20."
            query = "Which proteins have accessions A8K4X5?"

            #* Single edge type (2 nodes - 1 edge)
            query = "List out exons found in enst00000691353 transcript"
            query = "Give me the enhancers that have association with the ensg00000227906 gene"
            query = "A transcript with the ID enst00000381261, which encodes a protein, has undergone changes. Based on this transcript, what is the resulting protien?"
            query = "What transcripts belong to the gene ensg00000232448?"


            #* Two different edges (3 nodes - 2 edges)
            query = "What proteins can we get from the gene ensg00000132639?"
            query = "Which promoter is involved in producing transcript enst00000658520?"
            query = "Can you provide a list of proteins synthesized by genes on chromosome x?"
            query = "What is the promoter involved in the formation of transcript enst00000658520?"
            query = "Which proteins are translated from transcripts that include exons located in chr20:1000000-1050000?"
            query = "What proteins can we get from ensg0043403 gene?"
            query = "what transcript can we get from a protein with accessions ABK453L and D3DW15?"
            query = "What genes on chromosome chrx, have transcripts that include exons and give me the protein they produce?"
            query = "Give me all enhancers associated with ensg00000132639 gene"

            #* Three different eges (4 nodes - edges)
            query = "What enhancers are involved in the formation of the protein p78504?"
        """

    def load_embeddings(self):
        if self.embeddings is None:
            self.embeddings = OllamaEmbeddings(model="nomic-embed-text")

    def load_llm(self):
        if self.llm is None:
            self.llm = ChatGoogleGenerativeAI(model="gemini-pro")

    @lru_cache(maxsize=128)
    def cached_similarity_search(self, query):
        if self.db_type == 'graph':
            return self.db_manager.run_query(query)
        elif self.db_type == "annotation":
            return self.db_manager.run_query(query)
        elif self.db_type == 'vector':
            return self.db_manager.similarity_search(query)

    def create_prompt_template(self):
        
        if self.db_type == 'graph':
            prompt_template = """
                <|system|> You are an expert in explaining biological information extracted from Rejuve.Bio's BioAtomspace knowledge graph. Based on the provided database context, describe the biological roles, relationships, and functions in a meaningful summary without explicitly including structural details.

                IMPORTANT: Format your response as a single-line JSON string, without line breaks or escaped characters, like this: {{"answer":"Your answer here"}}

                Database Results:  
                {context}  

                Reference Information (For understanding only; exclude from response):
                - Nodes: {nodes}
                - Relationships: {relationship_mapping}
                - Property Keys: {property_keys}

                Contextual Instructions:
                - Summarize gene functions, protein synthesis, transcription, and regulatory roles.
                - Describe any diseases associated with this gene, including known genetic disorders or phenotypes.
                - Mention connections to diseases, specific pathways, and relevant biological processes.
                - Include details about the gene's role in disease pathology if mutations or expressions are involved.
                - If available, summarize related studies or known genetic markers for disease predisposition.
                - Describe key biological processes, relationships, and data patterns based on node types and relationships.
                - Highlight regulatory roles like promoters or enhancers, connections such as "gene to transcript" (transcribed_to), and their biological significance.
                - Begin with: "The biological data reveals:"
                
                Rules for JSON:
                1. Use double quotes for all variables and values.
                2. Return ONLY the JSON string without any additional text or comments.
                3. No line breaks or backticks; respond only with JSON.

                Previous Context:
                {session_history}
            """
        elif self.db_type == "annotation":
            prompt_template = """
                <|system|> You are an expert in interpreting biological annotations from Rejuve.Bio's BioAtomspace knowledge graph. Using the provided database context, explain the biological functions, roles, and relationships in a clear summary without explicitly including structural details. 
                After processing the query, provide a suggestion for the user on further insight to the data by referring to all the node, their respective property keys and relationship of the current node(s) to other nodes. Include relevant the relationships and their respective mapping in the suggestion.
                IMPORTANT: Format your response as a single-line JSON string, without line breaks or escaped characters, like this: {{"answer":"Your answer here", suggestion: "Your suggestion here"}}

                Database Results:  
                {context}  

                Reference Information (For understanding only; exclude from response):
                - Nodes: {nodes}
                - Relationships: {relationship_mapping}
                - Property Keys: {property_keys}

                Contextual Instructions:
                - Describe gene functions, transcript roles, protein synthesis, and associated regulatory elements.
                - Describe any diseases associated with this gene, including known genetic disorders or phenotypes.
                - Mention connections to diseases, specific pathways, and relevant biological processes.
                - Include details about the gene's role in disease pathology if mutations or expressions are involved.
                - If available, summarize related studies or known genetic markers for disease predisposition.
                - Highlight annotations around key biological pathways and regulatory mechanisms like promoters and enhancers.
                - Emphasize connections like "gene to transcript" (transcribed_to) or "transcript to protein" (translates_to) and their biological relevance.
                - Start with: "The annotations reveal:"

                Preffered answer format: {preferred_answer_format}

                Rules for JSON:
                1. Use double quotes for all variables and values.
                2. Return ONLY the JSON string without any additional text or comments.
                3. No line breaks or backticks; respond only with JSON.

                Previous Context:
                {session_history}
            """
        else:
            # Use a simple template for vector queries
            prompt_template = """<|system|> You have been provided with a documentation, previous chat history and a query, try to find out 
            the answer to the question only using the information from the documentation and history. Make the answer an elaborate one.
            If the answer to the question is not found within the documentation.

            IMPORTANT: You must format your response as a single-line JSON string with no line breaks, no extra spaces, no \\, and no escaped characters. The format should be exactly like this: {{"answer":"Your answer here"}}

            Rules for JSON generation:
                1. Use double quotes for all variables and their corresponding values.
                2. Return ONLY the JSON - no explanations, no markdown.
                3. Do not include any comments or additional text.
                4. Do not wrap the JSON in backticks.

            Documentation: {context}
            
            history: {session_history}

            Query: {query}

            <|assistant|>"""

        return PromptTemplate.from_template(prompt_template)

    def query_db(self, query):
        self.load_embeddings()
        self.load_llm()

        # Perform similarity search
        context = self.cached_similarity_search(query)

        # Ensure context is not empty
        if not context:
            return {"answer": "No relevant documents found."}

        # Prepare the context and prompt for the query
        context_text = "\n".join([str(item) for item in context]) if isinstance(context, list) else str(context)
        context = [Document(page_content=context_text)]
        
        prompt = self.create_prompt_template()

        chain = create_stuff_documents_chain(llm=self.llm, prompt=prompt)

        try:
            answer = chain.invoke({
                "context": context,
                "session_history": self.session_chat,
                "query": query,
                "nodes": self.nodes,
                "relationship_mapping": self.relationship_mapping,
                "property_keys": self.property_keys,   
                "preferred_answer_format": self.preffered_answer_format,
                "example_queries": self.example_queries
            })

            clean_answer = str(answer.strip()).replace("\\", '')
            parsed_answer = self.parser.parse(clean_answer)

            self.session_chat.append({"query": query, "answer": clean_answer})
            return parsed_answer

        except Exception as e:
            return {"answer": f"Error during processing: {str(e)}"}


    def close(self):
        if hasattr(self.db_manager, 'close'):
            self.db_manager.close()
