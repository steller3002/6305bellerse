import requests
from random import choice

timeout = 5

def download_image(url):
    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        return r
    except requests.exceptions.RequestException as e:
        return e.response

def download_painting_info(painting_id):
    url = f"https://collectionapi.metmuseum.org/public/collection/v1/objects/{painting_id}"
    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.RequestException as e:
        return e.response.json()

def get_random_metadata_with_image(painting_ids):
    max_attempts = 10
    attempt = 0

    painting_id = choice(painting_ids)
    metadata = download_painting_info(painting_id)

    while metadata['primaryImage'] == '' and attempt < max_attempts:
        painting_id = choice(painting_ids)
        metadata = download_painting_info(painting_id)
        attempt += 1

    return metadata