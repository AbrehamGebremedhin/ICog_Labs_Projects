import requests
import time
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as ec
from urllib.parse import urlparse


class Scraper:
    def __init__(self, url) -> None:
        self.url_stack = [url]
        self.base_url = url
        self.base_domain = urlparse(url).netloc
        self.visited_urls = list()
        self.links = [url+'/issues', url+'/pulls',url+'/actions',url+'/projects',url+'/activity',url+'/security',url+'/commits', url+'/fork',url+'/actions',url+'/projects',url+'/actions',url+'/security',]
        self.soup = None

    def driver(self, url):
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

    def find_all(self):
        for url in self.url_stack:
            if url in self.visited_urls:
                continue

            content = self.driver(url)

            self.soup = BeautifulSoup(content, 'html.parser')

            for link in self.soup.find_all('a'):
                href = link.get('href')
                if href.startswith('/trueagi-io/hyperon-experimental'):
                    href = "https://github.com" + href
                elif href.startswith('http') or href.startswith('https') and urlparse(href).netloc != self.base_domain:
                    continue

                if href.startswith('http') and href not in self.visited_urls:
                    with open('output2.txt', 'a', encoding='utf-8') as file:
                        try:
                            response = self.driver(href)
                            soup = BeautifulSoup(response, 'html.parser')
                            text = soup.get_text(strip=True)
                            file.write(f"{text}\n")
                        except Exception as e:
                            print(f"Failed to scrape {url}: {e}")
                    self.visited_urls.append(href)

            time.sleep(10)


scrape = Scraper("https://github.com/trueagi-io/hyperon-experimental/")
scrape.find_all()
