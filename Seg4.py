import cv2
import numpy as np
import os

def row_sum(binary_image):
    return np.sum(binary_image, axis=1)

def col_sum(binary_image):
    return np.sum(binary_image, axis=0)

def find_row_segments(projection, min_gap_size, max_gap_size):
    segments = []
    zero_count = 0
    start = -1

    for i, value in enumerate(projection):
        if value == 0:
            zero_count += 1
        else:
            if start == -1:
                start = i
            if zero_count >= min_gap_size and (zero_count <= max_gap_size or i == len(projection) - 1):
                segments.append((start, i - zero_count))
                start = i
            zero_count = 0

    if start != -1:
        segments.append((start, len(projection)))

    # Add a segment for the end of the sentence (a row with no Braille cells)
    segments.append((-1, -1))
    return segments

def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    return img

if __name__ == "__main__":
    image_path = "TestingImages/aaa.jpeg"
    output_dir = "OutputImages"

    cell_width = 20
    cell_height = 30

    min_row_gap = 10
    max_row_gap = 30
    min_col_gap = 5
    max_col_gap = 30

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Preprocess the image before segmentation
    binary = preprocess_image(image_path)
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calculate row_projection and col_projection
    row_projection = row_sum(binary)
    col_projection = col_sum(binary)

    # Find row_segments using the updated find_row_segments function
    row_segments = find_row_segments(row_projection, min_row_gap, max_row_gap)
    col_segments = find_row_segments(col_projection, min_col_gap, max_col_gap)

    # Iterate through the rows and columns of the braille_cells array
    marked_image = image.copy()

    output_index = 0
    for i, row_segment in enumerate(row_segments[:-1]):
        next_row_segment = row_segments[i + 1]
        if next_row_segment == (-1, -1):
            # End of the sentence, move to the next row
            continue

        for j, col_segment in enumerate(col_segments):
            cell = binary[row_segment[0]:row_segment[1], col_segment[0]:col_segment[1]]

            # Resize and pad the cell
            h, w = cell.shape
            if h > 0 and w > 0:
                scale_factor = min(cell_width / w, cell_height / h)
                resized_cell = cv2.resize(cell, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)

                padded_cell = np.zeros((cell_height, cell_width), dtype=np.uint8)
                h, w = resized_cell.shape
                y_offset = (cell_height - h) // 2
                x_offset = (cell_width - w) // 2
                padded_cell[y_offset:y_offset + h, x_offset:x_offset + w] = resized_cell

                if np.any(padded_cell):  # Check if the cell is not empty
                    # Save the segmented character in the output directory
                    cv2.imwrite(os.path.join(output_dir, f"Char{output_index}.png"), padded_cell)
                    output_index += 1

                # Draw a rectangle around the Braille cell on the marked_image
                cv2.rectangle(marked_image, (col_segment[0], row_segment[0]),
                              (col_segment[1], row_segment[1]), (0, 255, 0), 1)

    # Display the marked image with rectangles around Braille cells
    cv2.imshow("Marked Image", marked_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()