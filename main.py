import json
import os.path
from random import choice
import csv
import requests

def download_painting_info(id):
    url = f"https://collectionapi.metmuseum.org/public/collection/v1/objects/{id}"
    try:
        r = requests.get(url)
        return r.json()
    except requests.exceptions.RequestException as e:
        return e

def download_image(url):
    try:
        r = requests.get(url)
        return r
    except requests.exceptions.RequestException as e:
        return e

painting_ids = []
path = 'C:\\Users\\belik\\OneDrive\\Рабочий стол\\MetObjects.csv'

with open(path, encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        if row["Classification"] == "Paintings":
            painting_ids.append(row["Object ID"])

painting_id = choice(painting_ids)
painting_info = download_painting_info(painting_id)
while painting_info["primaryImage"] == "":
    painting_id = choice(painting_ids)
    painting_info = download_painting_info(painting_id)

image = download_image(painting_info["primaryImage"])

directory = "paintings"
if not os.path.exists(directory):
    os.makedirs(directory)

image_filename = os.path.join(directory, f"{painting_id}.jpg")
metadata_filename = os.path.join(directory, f"{painting_id}.json")

with open(metadata_filename, 'w') as f:
    json.dump(painting_info, f)
with open(image_filename, 'wb') as f:
    f.write(image.content)