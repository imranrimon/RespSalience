import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import time

BASE_URL = "https://coe.northeastern.edu/Research/AClab/AIR-400/"
LOCAL_DATA_DIR = "f:/Respiratory Project/data/AIR_400/"

def download_file(url, local_path):
    if os.path.exists(local_path):
        print(f"Skipping {local_path}, already exists.")
        return
    
    print(f"Downloading {url} to {local_path}...")
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        with open(local_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
    except Exception as e:
        print(f"Failed to download {url}: {e}")

def scrape_and_download(folder_name):
    folder_url = urljoin(BASE_URL, f"{folder_name}/")
    local_folder_path = os.path.join(LOCAL_DATA_DIR, folder_name)
    os.makedirs(local_folder_path, exist_ok=True)
    
    try:
        response = requests.get(folder_url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        for link in soup.find_all('a'):
            href = link.get('href')
            if href.endswith(('.mp4', '.csv', '.json', '.png')) and not href.startswith('?'):
                file_url = urljoin(folder_url, href)
                file_name = href.split('/')[-1]
                local_file_path = os.path.join(local_folder_path, file_name)
                download_file(file_url, local_file_path)
                
    except Exception as e:
        print(f"Error scraping {folder_url}: {e}")

if __name__ == "__main__":
    subjects = [f"S{i:02d}" for i in range(1, 11)]
    for sub in subjects:
        print(f"--- Processing {sub} ---")
        scrape_and_download(sub)
        time.sleep(1) # Be polite to the server
