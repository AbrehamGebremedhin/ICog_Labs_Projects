import requests
import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import time


def scrape_website(url):
    # Set up Selenium WebDriver
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')
    driver = webdriver.Chrome(service=Service(
        ChromeDriverManager().install()), options=options)

    # Load the URL
    driver.get(url)

    # Wait for the page to fully load
    time.sleep(5)  # Adjust the sleep time as needed

    # Extract the HTML content
    html = driver.page_source
    driver.quit()

    # Parse the HTML content
    soup = BeautifulSoup(html, 'html.parser')

    # Extract the title
    title = soup.title.string if soup.title else "No Title Found"

    # Extract all text
    text = ' '.join([p.get_text(strip=True) for p in soup.find_all('p')])

    # Extract keywords from all h1 to h6 tags
    keywords = [header.get_text(strip=True) for header in soup.find_all([
        'h1', 'h2', 'h3', 'h4', 'h5', 'h6'])]

    # Construct the output dictionary
    result = {
        "URL": url,
        "Text": text,
        "Title": title,
        "Keywords": keywords
    }

    return result


# Example usage
url = "https://www.rejuve.bio/"  # Replace this with the actual URL
scraped_data = scrape_website(url)
# Wrap the dictionary in a list to create a single-row DataFrame
df = pd.DataFrame([scraped_data])
df.to_csv('output.csv', index=False)
print(scraped_data)
