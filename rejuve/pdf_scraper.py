import PyPDF2
import pandas as pd


def scrape_pdf(file_path):
    # Open the PDF file
    try:
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)

            # Extract the metadata for title if available
            title = reader.metadata.title if reader.metadata and reader.metadata.title else "No Title Found"

            # Extract text from each page
            text = ''
            for page in reader.pages:
                text += page.extract_text() or ''  # Extract text from each page

            # Extract keywords based on headings (heuristically searching for large font size)
            keywords = []
            for page in reader.pages:
                # Search for text that seems like a heading (uppercase words, short lines, etc.)
                page_text = page.extract_text() or ''
                lines = page_text.split('\n')
                for line in lines:
                    # Heuristically consider a line as a keyword if it's in uppercase or bold
                    if line.isupper() or len(line.split()) < 5:
                        keywords.append(line.strip())

            # Construct the output dictionary
            result = {
                "URL": file_path,
                "Text": text.strip(),
                "Title": title,
                "Keywords": keywords
            }

            return result
    except Exception as e:
        return {"error": f"Failed to read the PDF file: {str(e)}"}


# Example usage
# Replace with the path to your PDF file
file_path = r"D:\Projects\ICog_Labs_Projects\rejuve\Rejuve.Bio Whitepaper V0.33_compressed.pdf"
scraped_data = scrape_pdf(file_path)
df = pd.DataFrame(scraped_data)
df.to_csv('pdf_output.csv')
