import time
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as ec
from urllib.parse import urlparse


def driver(url):
    web_driver = webdriver.Chrome()
    web_driver.get(url)

    WebDriverWait(web_driver, 10).until(
        ec.presence_of_element_located((By.TAG_NAME, 'body'))
    )

    WebDriverWait(web_driver, 10).until(
        ec.presence_of_element_located((By.CSS_SELECTOR, 'a'))
    )

    content = web_driver.page_source

    web_driver.quit()

    return content


def clean_text(text):
    return ' '.join(text.split())


class Scraper:
    def __init__(self, url) -> None:
        self.url_stack = [url]
        self.base_url = url
        self.base_domain = urlparse(url).netloc
        self.visited_urls = ["https://metta-lang.dev/docs/playground/playground.html"]
        self.links = list()
        self.soup = None

    def find_all(self):
        # Iterate through each URL in the stack
        for url in self.url_stack:
            # Skip the URL if it has already been visited
            if url in self.visited_urls:
                continue

            # Use the driver function to fetch the content of the URL
            content = driver(url)
            # Parse the fetched content using BeautifulSoup
            self.soup = BeautifulSoup(content, 'html.parser')

            # Find all anchor tags to process the links
            for link in self.soup.find_all('a'):
                href = link.get('href')
                # Prefix relative URLs with the base URL
                if href.startswith('/'):
                    href = "https://metta-lang.dev" + href
                # Skip external links not belonging to the base domain
                elif href.startswith('http') or href.startswith('https') and urlparse(href).netloc != self.base_domain:
                    continue

                # Process the link if it's not already visited
                if href.startswith('http') and href not in self.visited_urls:
                    # Open the output file in append mode
                    with open('output.txt', 'a', encoding='utf-8') as file:
                        try:
                            # Fetch the content of the link
                            response = driver(href)
                            soup = BeautifulSoup(response, 'html.parser')
                            # Try to find the main content in <main> or <article> tags
                            soup = soup.find('main') or soup.find('article', soup)

                            # Process each <li> tag in the document
                            for li in soup.find_all('li'):
                                # Extract and concatenate text from all <code> tags within the <li>
                                code_texts = ' '.join(
                                    tag.get_text(separator=' ', strip=True) for tag in li.find_all('code'))
                                # Replace <code> tags with their concatenated text
                                for tag in li.find_all('code'):
                                    tag.replace_with(code_texts)
                                # Update the <li> tag with modified text
                                li_text = li.get_text(separator=' ', strip=True)
                                li.string = li_text

                            # Process each <pre> tag in the document
                            for pre in soup.find_all('pre'):
                                # Extract and concatenate text from all <code> tags within the <pre>
                                code_texts = []
                                for code in pre.find_all('code'):
                                    # Extract text from all <span> tags within the <code>
                                    span_texts = ' '.join(
                                        span.get_text(separator=' ', strip=True) for span in code.find_all('span'))
                                    code_texts.append(span_texts)
                                # Concatenate all code texts into a single line
                                concatenated_code_texts = ' '.join(code_texts)
                                # Remove <code> tags from the <pre> content
                                for code in pre.find_all('code'):
                                    code.replace_with('')
                                # Update the <pre> tag with modified content
                                pre_text = pre.get_text(separator=' ', strip=True)
                                pre.string = f"{pre_text} {concatenated_code_texts}"

                            # Get the modified text of the whole document
                            text = soup.get_text(separator='\n', strip=True)
                            # Write the modified text to the output file
                            file.write(f"{text}\n")

                        except Exception as e:
                            # Log any exceptions encountered during processing
                            print(f"Failed to scrape {href}: {e}")
                    # Add the processed URL to the list of visited URLs
                    self.visited_urls.append(href)

            # Pause execution for 10 seconds to avoid overwhelming the server
            time.sleep(10)


scrape = Scraper("https://metta-lang.dev/docs/learn/learn.html")
scrape.find_all()
