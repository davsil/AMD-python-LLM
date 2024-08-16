import requests
import os
import sys
import argparse

# GitHub personal access token for authentication (if needed)
GITHUB_TOKEN=os.environ.get("GITHUB_TOKEN")

# Headers for authentication
headers = {
    'Authorization': f'token {GITHUB_TOKEN}',
    'Accept': 'application/vnd.github.v3+json'
}

def fetch_repo_content(owner, repo, branch, path=''):
    """
    Fetches the content of a GitHub repository.
    """
    url = f'https://api.github.com/repos/{owner}/{repo}/contents/{path}'
    response = requests.get(url, headers=headers)

    try:
        response.raise_for_status()
        return response.json()
        # return response.text
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
    except Exception as err:
        print(f"Other error occurred: {err}")
    return []

def download_file(file_url, save_path):
    """
    Downloads a file from a URL and saves it locally.
    """
    response = requests.get(file_url, headers=headers)
    try:
        response.raise_for_status()
        with open(save_path, 'wb') as file:
            file.write(response.content)
        print(f"Downloaded {save_path}")
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
    except Exception as err:
        print(f"Other error occurred: {err}")


def extract_documents(owner, repo, branch='main', file_types=['.txt', '.md', '.rst'], save_dir='documents'):
    """
    Extracts documents of specified file types from a GitHub repository.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    def fetch_and_save_files(path=''):
        contents = fetch_repo_content(owner, repo, branch, path)
        for item in contents:
            if item['type'] == 'file' and any(item['name'].endswith(ft) for ft in file_types):
                file_path = os.path.join(save_dir, item['path'].replace('/', '_'))
                download_file(item['download_url'], file_path)
            elif item['type'] == 'dir':
                fetch_and_save_files(item['path'])

    fetch_and_save_files()


def process_url_file(github_url_file):
    """
    Reads a file containing GitHub repository URLs and extracts documents from each repository.
    """
    with open(github_url_file, 'r') as file:
        urls = file.readlines()
    
    for url in urls:
        url = url.strip()
        if url:
            try:
                parts = url.split('/')
                owner = parts[3]
                repo = parts[4]
                extract_documents(owner, repo, "develop",
                        # python files are added to types
                        file_types=['.txt', '.md', '.rst', '.py'], save_dir=f'{owner}_{repo}')
            except IndexError:
                print(f"Invalid URL format: {url}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract documents from GitHub repositories.")
    parser.add_argument('github_url_file', help="File containing GitHub repository URLs")

    args = parser.parse_args()
    process_url_file(args.github_url_file)

