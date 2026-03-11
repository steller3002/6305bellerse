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
cv2_image = cv2.imread(image_path)

# приведение цветного изображения к полутоновому
halftone_f_sh = to_halftone_f_sh(image_array)
write_np_image(save_directory, 'halftone_f_sh', halftone_f_sh)

halftone_image = to_halftone(image_array)
write_np_image(save_directory, 'halftone', halftone_image)

cv2_halftone = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2GRAY)
cv2.imwrite(os.path.join(save_directory, 'cv2_halftone.jpg'), cv2_halftone)

# свёртка и использованием двумерной маски
# фильтр гауса
mask = create_gauss_matrix(13)
gauss_image = convolution(image_array, mask)
write_np_image(save_directory, 'gauss', gauss_image)

cv2_gauss = cv2.GaussianBlur(cv2_image, (13, 13), 0)
cv2.imwrite(os.path.join(save_directory, 'cv2_gauss.jpg'), cv2_gauss)

# фильтр Собеля
sobel_image = sobel(image_array)
write_np_image(save_directory, 'sobel', sobel_image)

cv_sobel_x = cv2.Sobel(cv2_halftone, cv2.CV_64F, 1, 0, ksize=3)
cv_sobel_y = cv2.Sobel(cv2_halftone, cv2.CV_64F, 0, 1, ksize=3)
sobel_combined = cv2.magnitude(cv_sobel_x, cv_sobel_y)
cv2_sobel = np.uint8(np.absolute(sobel_combined))
cv2.imwrite(os.path.join(save_directory, 'cv2_Sobel.jpg'), cv2_sobel)