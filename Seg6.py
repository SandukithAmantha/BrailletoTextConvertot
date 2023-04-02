# import cv2
# import numpy as np
# import os
#
# def row_col_sum(binary_image):
#     row_sum = np.sum(binary_image, axis=1)
#     col_sum = np.sum(binary_image, axis=0)
#     return row_sum, col_sum
#
# def find_segments(projection):
#     nonzero_elements = np.nonzero(projection)[0]
#     adjacent_gaps = np.diff(nonzero_elements)
#     mean_gap_size = np.mean(adjacent_gaps[adjacent_gaps > 1])
#     large_gaps = np.where(adjacent_gaps > mean_gap_size)[0]
#
#     segments = []
#     start = 0
#     for gap_idx in large_gaps:
#         segments.append((nonzero_elements[start], nonzero_elements[gap_idx]))
#         start = gap_idx + 1
#     segments.append((nonzero_elements[start], nonzero_elements[-1]))
#
#     return segments
#
# def preprocess_and_segment(image_path):
#     img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#     _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
#
#     # Noise removal and hole filling
#     kernel = np.ones((3, 3), np.uint8)
#     img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
#     img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
#     row_sum, col_sum = row_col_sum(img)
#     row_segments = find_segments(row_sum)
#     col_segments = find_segments(col_sum)
#
#     return img, row_segments, col_segments
#
# def draw_segments(image_path, row_segments, col_segments):
#     image = cv2.imread(image_path)
#     for row_segment in row_segments:
#         cv2.line(image, (0, row_segment[0]), (image.shape[1], row_segment[0]), (0, 255, 0), 1)
#         cv2.line(image, (0, row_segment[1]), (image.shape[1], row_segment[1]), (0, 255, 0), 1)
#
#     for col_segment in col_segments:
#         cv2.line(image, (col_segment[0], 0), (col_segment[0], image.shape[0]), (0, 255, 0), 1)
#         cv2.line(image, (col_segment[1], 0), (col_segment[1], image.shape[0]), (0, 255, 0), 1)
#
#     return image
#
# if __name__ == "__main__":
#     image_path = "TestingImages/aaa.jpeg"
#
#     binary_image, row_segments, col_segments = preprocess_and_segment(image_path)
#     segmented_image = draw_segments(image_path, row_segments, col_segments)
#
#     cv2.imshow("Segmented Image", segmented_image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()


import cv2
import numpy as np
import os

def row_col_sum(binary_image):
    row_sum = np.sum(binary_image, axis=1)
    col_sum = np.sum(binary_image, axis=0)
    return row_sum, col_sum

def find_segments(projection):
    nonzero_elements = np.nonzero(projection)[0]
    adjacent_gaps = np.diff(nonzero_elements)
    mean_gap_size = np.mean(adjacent_gaps[adjacent_gaps > 1])
    large_gaps = np.where(adjacent_gaps > mean_gap_size)[0]

    segments = []
    start = 0
    for gap_idx in large_gaps:
        segments.append((nonzero_elements[start], nonzero_elements[gap_idx]))
        start = gap_idx + 1
    segments.append((nonzero_elements[start], nonzero_elements[-1]))

    return segments

def preprocess_and_segment(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)

    # Noise removal and hole filling
    kernel = np.ones((3, 3), np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

    row_sum, col_sum = row_col_sum(img)
    row_segments = find_segments(row_sum)
    col_segments = find_segments(col_sum)

    return img, row_segments, col_segments

def draw_segments(image_path, row_segments, col_segments, offset=5):
    image = cv2.imread(image_path)
    for row_segment in row_segments:
        cv2.line(image, (0, max(row_segment[0] - offset, 0)), (image.shape[1], max(row_segment[0] - offset, 0)), (0, 255, 0), 1)
        cv2.line(image, (0, min(row_segment[1] + offset, image.shape[0])), (image.shape[1], min(row_segment[1] + offset, image.shape[0])), (0, 255, 0), 1)

    for col_segment in col_segments:
        cv2.line(image, (max(col_segment[0] - offset, 0), 0), (max(col_segment[0] - offset, 0), image.shape[0]), (0, 255, 0), 1)
        cv2.line(image, (min(col_segment[1] + offset, image.shape[1]), 0), (min(col_segment[1] + offset, image.shape[1]), image.shape[0]), (0, 255, 0), 1)

    return image

if __name__ == "__main__":
    image_path = "TestingImages/aaa.jpeg"

    binary_image, row_segments, col_segments = preprocess_and_segment(image_path)
    segmented_image = draw_segments(image_path, row_segments, col_segments)

    cv2.imshow("Segmented Image", segmented_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
