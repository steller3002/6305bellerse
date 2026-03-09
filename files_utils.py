import csv
import os.path
import json
from PIL import Image


def get_painting_ids(csv_path):
    painting_ids = []

    with open(csv_path, encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row["Classification"] == "Paintings":
                painting_ids.append(row["Object ID"])

    return painting_ids

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