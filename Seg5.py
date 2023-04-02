# import cv2
# import numpy as np
# import os
#
# def row_sum(binary_image):
#     return np.sum(binary_image, axis=1)
#
# def col_sum(binary_image):
#     return np.sum(binary_image, axis=0)
#
# def find_segments(projection, min_gap_size):
#     segments = []
#     zero_count = 0
#     start = -1
#
#     for i, value in enumerate(projection):
#         if value == 0:
#             zero_count += 1
#         else:
#             if start == -1:
#                 start = i
#             if zero_count >= min_gap_size:
#                 segments.append((start, i - zero_count))
#                 start = i
#             zero_count = 0
#
#     if start != -1:
#         segments.append((start, len(projection)))
#     return segments
#
# def segment_braille_cells(binary_image, min_row_gap, min_col_gap):
#     row_projection = row_sum(binary_image)
#     col_projection = col_sum(binary_image)
#
#     row_segments = find_segments(row_projection, min_row_gap)
#     col_segments = find_segments(col_projection, min_col_gap)
#
#     braille_cells = []
#     for row_segment in row_segments:
#         row_cells = []
#         for col_segment in col_segments:
#             cell = binary_image[row_segment[0]:row_segment[1], col_segment[0]:col_segment[1]]
#             row_cells.append(cell)
#         braille_cells.append(row_cells)
#
#     return braille_cells, row_segments, col_segments
#
# def preprocess_image(image_path):
#     img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#     img_blurred = cv2.GaussianBlur(img, (5, 5), 0)
#     img_thresh = cv2.adaptiveThreshold(img_blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
#     kernel = np.ones((3, 3), np.uint8)
#     img_morph = cv2.morphologyEx(img_thresh, cv2.MORPH_CLOSE, kernel)
#     return img_morph
#
# if __name__ == "__main__":
#     image_path = "TestingImages/aaa.jpeg"
#     output_dir = "OutputImages"
#     min_row_gap = 10
#     min_col_gap = 10
#
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
#
#     binary = preprocess_image(image_path)
#     image = cv2.imread(image_path)
#
#     braille_cells, row_segments, col_segments = segment_braille_cells(binary, min_row_gap, min_col_gap)
#
#     marked_image = image.copy()
#
#     for i, row in enumerate(braille_cells):
#         for j, cell in enumerate(row):
#             cv2.imwrite(os.path.join(output_dir, f"Cell_{i}-{j}.png"), cell)
#             cv2.rectangle(marked_image, (col_segments[j][0], row_segments[i][0]), (col_segments[j][1], row_segments[i][1]), (0, 255, 0), 1)
#
#     cv2.imshow("Marked Image", marked_image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

#
# import cv2
# import numpy as np
# import os
#
# def preprocess_image(image_path):
#     img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#     img_blurred = cv2.GaussianBlur(img, (5, 5), 0)
#     img_thresh = cv2.adaptiveThreshold(img_blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
#     kernel = np.ones((3, 3), np.uint8)
#     img_morph = cv2.morphologyEx(img_thresh, cv2.MORPH_CLOSE, kernel)
#     return img_morph
#
# def segment_braille_cells(preprocessed_image):
#     # Find connected components and their bounding boxes
#     num_labels, _, stats, _ = cv2.connectedComponentsWithStats(preprocessed_image)
#
#     # Filter out very small components (likely noise)
#     min_area = 50
#     filtered_stats = [s for s in stats if s[cv2.CC_STAT_AREA] > min_area]
#
#     # Sort components by their top-left y-coordinate
#     sorted_stats = sorted(filtered_stats, key=lambda s: s[cv2.CC_STAT_TOP])
#
#     braille_cells = []
#     row_cells = []
#     prev_y = sorted_stats[0][cv2.CC_STAT_TOP]
#
#     # Group components into rows
#     for s in sorted_stats:
#         x, y, w, h, _ = s
#         if y - prev_y > h:
#             braille_cells.append(row_cells)
#             row_cells = []
#
#         cell = preprocessed_image[y:y + h, x:x + w]
#         row_cells.append(cell)
#         prev_y = y
#
#     braille_cells.append(row_cells)
#     return braille_cells
#
# if __name__ == "__main__":
#     image_path = "TestingImages/aaa.jpeg"
#     output_dir = "OutputImages"
#
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
#
#     preprocessed_image = preprocess_image(image_path)
#     image = cv2.imread(image_path)
#
#     braille_cells = segment_braille_cells(preprocessed_image)
#
#     # Save the segmented Braille cells
#     for i, row in enumerate(braille_cells):
#         for j, cell in enumerate(row):
#             cv2.imwrite(os.path.join(output_dir, f"Cell_{i}-{j}.png"), cell)


# import cv2
# import numpy as np
# import os
#
# def preprocess_image(image_path):
#     img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#     img_blurred = cv2.GaussianBlur(img, (5, 5), 0)
#     img_thresh = cv2.adaptiveThreshold(img_blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
#     kernel = np.ones((3, 3), np.uint8)
#     img_morph = cv2.morphologyEx(img_thresh, cv2.MORPH_CLOSE, kernel)
#     return img_morph
#
# def segment_braille_cells(preprocessed_image):
#     num_labels, _, stats, _ = cv2.connectedComponentsWithStats(preprocessed_image)
#     min_area = 50
#     filtered_stats = [s for s in stats if s[cv2.CC_STAT_AREA] > min_area]
#     sorted_stats = sorted(filtered_stats, key=lambda s: s[cv2.CC_STAT_TOP])
#
#     braille_cells = []
#     row_cells = []
#     prev_y = sorted_stats[0][cv2.CC_STAT_TOP]
#
#     for s in sorted_stats:
#         x, y, w, h, _ = s
#         if y - prev_y > h:
#             braille_cells.append(row_cells)
#             row_cells = []
#
#         cell = preprocessed_image[y:y + h, x:x + w]
#         row_cells.append((cell, x, y, w, h))
#         prev_y = y
#
#     braille_cells.append(row_cells)
#     return braille_cells
#
#
# if __name__ == "__main__":
#     image_path = "TestingImages/aaa.jpeg"
#     output_dir = "OutputImages"
#
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
#
#     preprocessed_image = preprocess_image(image_path)
#     image = cv2.imread(image_path)
#
#     braille_cells = segment_braille_cells(preprocessed_image)
#     marked_image = image.copy()
#
#     # Save the segmented Braille cells and draw green lines on the marked_image
#     for i, row in enumerate(braille_cells):
#         for j, (cell, x, y, w, h) in enumerate(row):
#             cv2.imwrite(os.path.join(output_dir, f"Cell_{i}-{j}.png"), cell)
#
#             cv2.rectangle(marked_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
#
#         cv2.imshow("Marked Image", marked_image)
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()


import cv2
import numpy as np
import os

def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img_blurred = cv2.GaussianBlur(img, (5, 5), 0)
    img_thresh = cv2.adaptiveThreshold(img_blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    kernel = np.ones((3, 3), np.uint8)
    img_morph = cv2.morphologyEx(img_thresh, cv2.MORPH_CLOSE, kernel)
    return img_morph

def segment_braille_groups(preprocessed_image):
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(preprocessed_image)
    min_area = 50
    filtered_stats = [s for s in stats if s[cv2.CC_STAT_AREA] > min_area]
    sorted_stats = sorted(filtered_stats, key=lambda s: s[cv2.CC_STAT_TOP])

    braille_groups = []
    group_cells = []
    prev_y = sorted_stats[0][cv2.CC_STAT_TOP]

    for s in sorted_stats:
        x, y, w, h, _ = s
        if y - prev_y > 2 * h:
            braille_groups.append(group_cells)
            group_cells = []

        group = preprocessed_image[y:y + 3 * h, x:x + 2 * w]
        group_cells.append((group, x, y, 2 * w, 3 * h))
        prev_y = y

    braille_groups.append(group_cells)
    return braille_groups


if __name__ == "__main__":
    image_path = "TestingImages/aaa.jpeg"
    output_dir = "OutputImages"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    preprocessed_image = preprocess_image(image_path)
    image = cv2.imread(image_path)

    braille_groups = segment_braille_groups(preprocessed_image)
    marked_image = image.copy()

    # Save the segmented Braille character groups and draw green lines on the marked_image
    for i, group in enumerate(braille_groups):
        for j, (group_img, x, y, w, h) in enumerate(group):
            cv2.imwrite(os.path.join(output_dir, f"Group_{i}-{j}.png"), group_img)

            cv2.rectangle(marked_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow("Marked Image", marked_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

