import os
import argparse
import requests
import tiktoken
import re
from urllib.parse import urlparse, urljoin
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter

# Function to fetch and parse URL content
def fetch_url_content_recursive(url, url_base, visited_urls, avoid_urls, level, max_level, outfile):

    documents = []
    if url in visited_urls:
        return documents

    for pat in avoid_urls:
        if pat in url:
            return documents

    visited_urls.add(url)
    print(f"added url {url} at level {level}")
    outfile.write(f"{url} at level {level}\n")

    response = requests.get(url)
    response.raise_for_status()  # Ensure we notice bad responses
    content_type = response.headers['Content-Type']

    if 'text/html' in content_type:
        soup = BeautifulSoup(response.text, 'html.parser')
        text_content = soup.get_text().encode('ascii', 'ignore').decode('ascii')
        documents.append(Document(page_content=re.sub(r'(\n\n)+', r'\n', text_content),
            metadata={"source": url}))

        # Recursively fetch links
        for link in soup.find_all('a', href=True):
            new_url = urljoin(url, link['href'])
            if "#" in new_url:
                continue
            if urlparse(new_url).hostname != urlparse(url).hostname:
                continue
            if url_base not in new_url:
                continue
            if level >= max_level:
                continue
            new_doc = fetch_url_content_recursive(new_url, url_base,
                    visited_urls, avoid_urls, level+1, max_level, outfile)
            if len(new_doc) > 0:
                documents.extend(new_doc)
    else:
        # documents.append({'text': response.text})
        print(f"content_type: {content_type}:  {url}")
    return documents


avoid_urls = [
    "www.amd.com",
    "github.com",
]

def main(max_level):

    start_urls = []

    # Open and read the file, which contains pairs of urls. The first url is where to start
    # fetching text. The second url is the base address to stay within on recursive fetches.
    with open('start_urls.txt', 'r') as file:
        # Read each line in the file
        for line in file:
            # Strip any extra whitespace or newlines
            line = line.strip()
            # Split the line by commas to get individual URLs
            urls = line.split(',')
            # Convert the list of URLs into a tuple and append to the list
            url_tuple = tuple(url.strip() for url in urls)
            start_urls.append(url_tuple)

    url_docs = []
    visited_urls = set()
    outfile = open('visited_urls.txt', 'w') 

    urlno = 0
    for url, url_base in start_urls:
        try:
            print(f"{url} : {url_base}\n")
            # content = fetch_url_content(url)
            # url_docs.append({'text': content})
            level = 1
            new_url_doc = fetch_url_content_recursive(url, url_base,
                    visited_urls, avoid_urls, level, max_level, outfile)
            if len(new_url_doc) > 0:
                url_docs.extend(new_url_doc)

            urlno += 1
            print(f"{len(new_url_doc)} total docs")
            urldomain = urlparse(url).hostname
            urlfile = f"{url_base.replace('https://','').replace('/','_')}_{urlno}"
            with open(urlfile, 'a') as f:
                for doc in new_url_doc:
                    f.write(f"{doc}\n")
                f.close()

        except Exception as e:
            print(f"Failed to fetch or parse content from {url}: {e}")

    print(f"urls: {len(url_docs)}")
    # Chunk text
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=250,
                                                   separators=['.', '\n', ' '])

    chunked_documents = text_splitter.split_documents(url_docs)

    print(len(chunked_documents))
    outfile.close()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Fetch URLs listed in start_urls.txt. Saves each URL to a file.")
    parser.add_argument('-m', '--max_level', type=int, default=4, help='maximum recursion level as integer')

    args = parser.parse_args()
    main(args.max_level)

