import os
import numpy as np
import cv2
from skimage.transform import radon
from skimage.morphology import disk, remove_small_objects
from skimage.filters import threshold_otsu
from numpy import pad
from numpy import pad
from skimage.segmentation import clear_border

def preprocess_image(image_path):
    image_original = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image_original, cv2.COLOR_BGR2GRAY)
    thresh = threshold_otsu(gray_image)
    binary = gray_image > thresh
    binary = (binary * 255).astype(np.uint8)
    return binary

def get_character_bounding_boxes(binary_image):
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bounding_boxes = [cv2.boundingRect(c) for c in contours]
    return bounding_boxes

def save_braille_characters(image, bounding_boxes, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i, (x, y, w, h) in enumerate(bounding_boxes):
        braille_char = image[y:y+h, x:x+w]
        output_path = os.path.join(output_dir, f"char_{i+1:03d}.png")
        cv2.imwrite(output_path, braille_char)

def main(image_path, output_dir):
    binary_image = preprocess_image(image_path)
    bounding_boxes = get_character_bounding_boxes(binary_image)
    save_braille_characters(binary_image, bounding_boxes, output_dir)

if __name__ == "__main__":
    image_path = "TestingImages/aaa.jpeg"
    output_dir = "BrailleCharacters"
    main(image_path, output_dir)
