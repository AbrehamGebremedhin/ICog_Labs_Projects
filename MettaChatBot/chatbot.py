import time
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_community.llms.huggingface_pipeline  import HuggingFacePipeline
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_postgres.vectorstores import PGVector
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama


class Chat:
    def __init__(self):
        self.connection = "postgresql+psycopg://langchain:langchain@localhost:6024/langchain"  # Uses psycopg3!
        self.name = "metta"
        self.embeddings = OllamaEmbeddings(model="mxbai-embed-large")
        self.vectorstore = PGVector(
            embeddings=self.embeddings,
            collection_name=self.name,
            connection=self.connection,
            use_jsonb=True,
        )
        self.llm = Ollama(model="phi3:mini")

    def load_data(self):
        loader = PyPDFLoader(f"/MettaChatBot/metta.pdf", extract_images=False)
        pages = loader.load_and_split()

        # Split data into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1024,
            chunk_overlap=64,
            length_function=len,
            add_start_index=True,
        )
        chunks = text_splitter.split_documents(pages)

        db = Chroma.from_documents(chunks, embedding=self.embeddings, persist_directory="metta")
        db.persist()

    def query_db(self, query):
        # Conduct a vector search for the user query and return the top 6 results
        context = self.vectorstore.similarity_search(query, k=6)

        # Custom prompt template suitable for the Phi-3 model
        qna_prompt_template = """<|system|> You have been provided with the context and a query, try to find out 
        the answer to the question only using the context information. If the answer to the question is not found 
        within the context, return "I dont know" as the response. <|end|> <|user|> Context: {context}

        Query: {query}<|end|>
        <|assistant|>"""

        # Use the custom prompt template to create a prompt and pass the required variables
        prompt = PromptTemplate(
            template=qna_prompt_template, input_variables=["context", "query"]
        )

        # Define the QNA chain
        chain = load_qa_chain(self.llm, chain_type="stuff", prompt=prompt)

        # Invoke the chain with the context and query
        answer = (chain.invoke({"input_documents": context, "query": query}, return_only_outputs=True, ))['output_text']

        # Extract the answer from the response
        answer = (answer.split("<|assistant|>")[-1]).strip()

        return answer
