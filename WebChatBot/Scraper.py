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
        self.urls_to_scrape = list()
        self.visited_urls = list()
        self.soup = None

    def find_all(self):
        for url in self.url_stack:
            driver = webdriver.Chrome()
            driver.get(url)

            WebDriverWait(driver, 10).until(
                ec.presence_of_element_located((By.TAG_NAME, 'body'))
            )

            WebDriverWait(driver, 10).until(
                ec.presence_of_element_located((By.CSS_SELECTOR, 'a'))
            )

            content = driver.page_source

            self.soup = BeautifulSoup(content, 'html.parser')

            for link in self.soup.find_all('a'):
                href = link.get('href')
                if href.startswith('/'):
                    href = "https://metta-lang.dev" + href
                    print(href)
                elif href.startswith('http') or href.startswith('https') and urlparse(href).netloc != self.base_domain:
                    continue
                elif href == 'https://metta-lang.dev/docs/playground/playground.html' or href == 'https://metta-lang.dev/':
                    continue

                if href.startswith('http') and href not in self.visited_urls:
                    self.url_stack.append(href)
                    self.urls_to_scrape.append(href)
                    self.visited_urls.append(href)

            driver.quit()
            time.sleep(10)

        for urlr in self.urls_to_scrape:
            print(urlr)

    def collect_data(self):
        self.find_all()
        with open('output.txt', 'a', encoding='utf-8') as file:
            for url in self.urls_to_scrape:
                try:
                    response = requests.get(url)
                    soup = BeautifulSoup(response.text, 'html.parser')
                    text = soup.get_text(strip=True)
                    file.write(f"URL: {url}\n{text}\n{'-'*80}\n")
                except Exception as e:
                    print(f"Failed to scrape {url}: {e}")


scrape = Scraper("https://metta-lang.dev/docs/learn/learn.html")
scrape.collect_data()
