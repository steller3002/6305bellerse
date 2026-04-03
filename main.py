from ApiProvider import ApiProvider
from DataProvider import DataProvider
from ImageProcessor import ImageProcessor


def main():
    storage = DataProvider("C:\\Users\\belik\\OneDrive\\Рабочий стол\\MetObjects.txt", "paintings")
    api = ApiProvider(storage)
    processor = ImageProcessor(api, storage)

    processor.download_new_artwork()
    processor.save_original()

    processor.halftone()
    processor.sobel()
    processor.gauss_blur(size=3, save_name="blur")
    processor.dithering()

    art1 = processor.download_new_artwork()
    art2 = processor.download_new_artwork()
    storage.save_numpy_image((art1 + art2).image, 'plus')

if __name__ == "__main__":
    main()