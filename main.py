import json
import math
import os.path
from random import choice
import csv
import requests
import numpy as np
from PIL import Image
import cv2

def create_gauss_matrix(n, sigma=1):
    matrix = np.zeros((n, n))
    center = n // 2
    for y in range(n):
        for x in range(n):
            offset_x = abs(x - center)
            offset_y = abs(y - center)
            matrix[y][x] = math.e ** -(((offset_x ** 2) + (offset_y ** 2)) / (2*sigma**2))
    return matrix / np.sum(matrix)

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

def convolution(array, mask):
    result = np.zeros_like(array).astype(np.float32)
    mask = np.array(mask).astype(np.float32)
    h, w = len(array), len(array[0])
    mask_sum = np.sum(mask)

    indent = len(mask[0]) // 2
    h, w = len(array), len(array[0])

    for y in range(indent, h-indent):
        for x in range(indent, w-indent):
            field = array[y-indent:y+indent+1,x-indent:x+indent+1]
            field_on_mask = field * mask[:, :, np.newaxis]
            result[y][x] = np.sum(field_on_mask, axis=(0, 1)) / mask_sum

    return np.clip(result, 0, 255).astype(np.uint8)

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

# приведение цветного изображения к полутоновому
image_array = np.array(Image.open(image_path).convert("RGB"))
halftone_f_sh = to_halftone_f_sh(image_array)
write_np_image(directory, 'halftone_f_sh', halftone_f_sh)
halftone_image = to_halftone(image_array)
write_np_image(directory, 'halftone', halftone_image)

cv2_image = cv2.imread(image_path)
cv2_halftone = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2GRAY)
cv2.imwrite(os.path.join(directory, 'cv2_halftone.jpg'), cv2_halftone)

# свёртка и использованием двумерной маски

mask = create_gauss_matrix(67, 11)
convolution_image = convolution(image_array, mask)
write_np_image(directory, 'gauss', convolution_image)