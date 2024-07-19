import os
from langchain_community.document_loaders import PyPDFLoader


def convert_pdf_to_text(path: str):
    loader = PyPDFLoader(path, extract_images=False)
    pages = str(loader.load_and_split())

    filename = "output.txt"

    with open(filename, "w", encoding="utf-8") as file:
        file.write(pages + "\n")
