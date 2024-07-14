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
        for url in self.url_stack:
            if url in self.visited_urls:
                continue

            content = driver(url)
            self.soup = BeautifulSoup(content, 'html.parser')

            for link in self.soup.find_all('a'):
                href = link.get('href')
                if href.startswith('/'):
                    href = "https://metta-lang.dev" + href
                elif href.startswith('http') or href.startswith('https') and urlparse(href).netloc != self.base_domain:
                    continue

                if href.startswith('http') and href not in self.visited_urls:
                    with open('output.txt', 'a', encoding='utf-8') as file:
                        try:
                            response = driver(href)
                            soup = BeautifulSoup(response, 'html.parser')
                            soup = soup.find('main')
                            if soup is None:
                                soup = BeautifulSoup(response, 'html.parser')
                                soup = soup.find('article')

                            # Process each <li> tag
                            for li in soup.find_all('li'):
                                # Extract and concatenate all <code> tags' text within the <li> tag
                                code_texts = ' '.join(
                                    tag.get_text(separator=' ', strip=True) for tag in li.find_all('code'))
                                # Replace the <code> tags within the <li> tag content with the concatenated code text
                                for tag in li.find_all('code'):
                                    tag.replace_with(code_texts)

                                # Get the modified text of the <li> tag
                                li_text = li.get_text(separator=' ', strip=True)
                                li.string = li_text

                            # Get the modified text of the whole document
                            text = soup.get_text(separator='\n', strip=True)
                            file.write(f"{text}\n")

                        except Exception as e:
                            print(f"Failed to scrape {href}: {e}")
                    self.visited_urls.append(href)

            time.sleep(10)


scrape = Scraper("https://metta-lang.dev/docs/learn/learn.html")
scrape.find_all()
