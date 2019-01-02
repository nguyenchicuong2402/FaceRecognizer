import os
import cv2
from imutils import paths
import imutils
from google_images_download import google_images_download


CHROME_DRIVER_PATH = os.path.dirname(__file__) + "\\data_model\\chromedriver.exe"


def download(keyword, limit, output):
    response = google_images_download.googleimagesdownload()   #class instantiation
    arguments = {
        "keywords": keyword,
        "limit": limit,
        "print_urls": False,
        "print_size": True,
        "format": "jpg",
        "format": "png",
        "size": "medium",
        "type": "face",
        "type": "photo",
        "socket_timeout": "300",
        "output_directory" : output,
        "chromedriver": CHROME_DRIVER_PATH
    }   #creating list of arguments
    paths = response.download(arguments)   #passing the arguments to the function