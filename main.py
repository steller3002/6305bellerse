import os
import cv2
import numpy as np
from PIL import Image
from api_utils import get_random_metadata_with_image, download_image
from files_utils import write_data, write_np_image, get_painting_ids
from image_processing import create_gauss_matrix, convolution, to_halftone, to_halftone_f_sh, sobel

csv_path = 'C:\\Users\\belik\\OneDrive\\Рабочий стол\\MetObjects.csv'
save_directory = "paintings"

painting_ids = get_painting_ids(csv_path)

metadata = get_random_metadata_with_image(painting_ids)
image = download_image(metadata['primaryImage'])
image_path, metadata_path = write_data(save_directory, metadata, image)

image_array = np.array(Image.open(image_path).convert("RGB"))

# приведение цветного изображения к полутоновому
halftone_f_sh = to_halftone_f_sh(image_array)
write_np_image(save_directory, 'halftone_f_sh', halftone_f_sh)

halftone_image = to_halftone(image_array)
write_np_image(save_directory, 'halftone', halftone_image)

cv2_image = cv2.imread(image_path)
cv2_halftone = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2GRAY)
cv2.imwrite(os.path.join(save_directory, 'cv2_halftone.jpg'), cv2_halftone)

# свёртка и использованием двумерной маски
# фильтр гауса
mask = create_gauss_matrix(3)
gauss_image = convolution(image_array, mask)
write_np_image(save_directory, 'gauss', gauss_image)

#фильтр собеля
sobel_image = sobel(image_array)
write_np_image(save_directory, 'sobel', sobel_image)