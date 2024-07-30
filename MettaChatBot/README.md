# Metta ChatBot

This project is a web chat application that utilizes the Hugging Face embedding model and Neo4j graph database. It aims to provide an interactive and intelligent chatbot experience for users.

## Steps taken

-To scrape the data from the website I used a headless browser using selenium and chrome driver, I used the BeautifulSoup library to parse the HTML content and extract the relevant information.

##

-After extracting the data I stored it in a text file and a CSV file.

##

-Then I selected and used 5 different Hugging Face embedding models on the data and generate embeddings.

##

-Then I stored the embeddings in a MongoDB Database.

##

-Then I used the embeddings in the database and compare them using Cosine Similarity.

##

-Then I transformed the n-dimensional vectors to a 2-d dimensional graph for visualization.

##

-Then I create the Neo4j graph database using the CSV file and query

##

-Then I used the Neo4j graph database to query the graph and retrieve the nodes and relationships

##

-Then used Langchain to create a RAG(Retrieval Augemented Generation) using the user query and the Neo4J database retrieval results as a context to be passed to a Large Language Model
