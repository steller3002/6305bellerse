import json
import os.path
from random import choice
import csv
import requests
import numpy as np
from PIL import Image
import cv2

def to_halftone_f_sh(array):
    # Алгоритм дизеринга Флойда-Штайнберга
    gray_array = (np.array(0.299*array[:,:,0] + 0.587*array[:,:,1] + 0.114*array[:,:,2]))
    h, w = len(gray_array), len(gray_array[0])

    for y in range(h-1):
        for x in range(1, w-1):
            old_pixel = gray_array[y][x]
            new_pixel = 255 if old_pixel > 128 else 0
            gray_array[y][x] = new_pixel

            error = old_pixel - new_pixel
            gray_array[y, x+1] += error * 7 / 16
            gray_array[y+1, x-1] += error * 3 / 16
            gray_array[y+1, x] += error * 5 / 16
            gray_array[y+1, x+1] += error * 1 / 16

    return gray_array.astype(np.uint8)

def to_halftone(array):
    gray_array = (np.array(0.299 * array[:, :, 0] + 0.587 * array[:, :, 1] + 0.114 * array[:, :, 2]))
    return gray_array.astype(np.uint8)

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

def download_metadata_with_imageurl():
    painting_id = choice(painting_ids)
    metadata = download_painting_info(painting_id)
    while metadata["primaryImage"] == "":
        painting_id = choice(painting_ids)
        metadata = download_painting_info(painting_id)

    return metadata

def write_data(directory, metadata, image):
    if not os.path.exists(directory):
        os.makedirs(directory)

    for file_name in os.listdir(directory):
        file_path = os.path.join(directory, file_name)
        if os.path.isfile(file_path):
            os.remove(file_path)

    image_id = metadata["objectID"]
    image_path = os.path.join(directory, f"{image_id}.jpg")
    metadata_path = os.path.join(directory, f"{image_id}.json")

    with open(metadata_path, 'w') as f:
        json.dump(metadata, f)
    with open(image_path, 'wb') as f:
        f.write(image.content)

    return image_path, metadata_path

def write_np_image(directory, name, array):
    if not os.path.exists(directory):
        os.makedirs(directory)

    path = os.path.join(directory, f"{name}.jpg")
    img = Image.fromarray(array)
    img.save(path)
    return path

painting_ids = []
csv_path = 'C:\\Users\\belik\\OneDrive\\Рабочий стол\\MetObjects.csv'
directory = "paintings"

with open(csv_path, encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        if row["Classification"] == "Paintings":
            painting_ids.append(row["Object ID"])

metadata = download_metadata_with_imageurl()
image = download_image(metadata["primaryImage"])
image_path, metadata_path = write_data(directory, metadata, image)

# Halftone
image_array = np.array(Image.open(image_path).convert("RGB"))
halftone_f_sh = to_halftone_f_sh(image_array)
write_np_image(directory, 'halftone_f_sh', halftone_f_sh)

halftone = to_halftone(image_array)
write_np_image(directory, 'halftone', halftone)

cv2_image = cv2.imread(image_path)
cv2_halftone = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2GRAY)
cv2.imwrite(os.path.join(directory, 'cv2_halftone.jpg'), cv2_halftone)