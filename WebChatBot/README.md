# Web ChatBot

This project is a web chat application that utilizes the Hugging Face embedding model and Neo4j graph database. It aims to provide an interactive and intelligent chatbot experience for users.

## Steps taken
-To scrape the data from the website I used a headless browser using selenium and chrome driver, I used the BeautifulSoup library to parse the HTML content and extract the relevant information.
-After extracting the data I stored it in a text file.
-Then I selected and used 5 different Hugging Face embedding models on the data and generate embeddings.
-Then I stored the embeddings in a MongoDB Database.
-Then I used the embeddings in the database and compare them using Cosine Similarity.
-Then I transformed the n-dimensional vectors to a 2-d dimensional graph for visualization.


## Features

- Utilizes the Hugging Face embedding model for natural language processing and understanding.
- Integrates with Neo4j graph database for storing and retrieving chatbot responses.
- Provides a user-friendly web interface for seamless communication with the chatbot.
