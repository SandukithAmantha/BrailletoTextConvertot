import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from Seg4 import preprocess_image, row_sum, col_sum, find_row_segments
from RegenerateBrailleCharacters import regenerate_braille_dots, create_braille_grid

segmented_characters_folder = "SegmentedCharacters"
regenerated_characters_folder = "RegeneratedCharacters"

if not os.path.exists(segmented_characters_folder):
    os.makedirs(segmented_characters_folder)

if not os.path.exists(regenerated_characters_folder):
    os.makedirs(regenerated_characters_folder)


labelDictionary = {0: 'අ', 1: 'අ', 2: 'ඉ', 3: 'ඊ', 4: 'උ', 5: 'එ', 6: 'ඒ', 7: 'ඔ', 8: 'ක', 9: 'ක්', 10: 'කා',
                   11: 'කැ', 12: 'කෑ', 13: 'කි', 14: 'කී', 15: 'කු', 16: 'කූ', 17: 'කෙ', 18: 'කේ', 19: 'කො',
                   20: 'කෝ', 21: 'ඛ', 22: 'ග', 23: 'ගි', 24: 'ගී', 25: 'ගු', 26: 'ගූ', 27: 'ඝ', 28: 'ඟ', 29: 'ච',
                   30: 'ඡ', 31: 'ජ', 32: 'ජ්', 33: 'ජි', 34: 'ජී', 35: 'ඣ', 36: 'ඤ', 37: 'ඥ', 38: 'ට', 39: 'ඨ',
                   40: 'ඩ',
                   41: 'ඪ', 42: 'ණ', 43: 'ඬ', 44: 'ත', 45: 'ත්', 46: 'ථ', 47: 'ථි', 48: 'ථී', 49: 'ද', 50: 'දු',
                   51: 'දූ', 52: 'ධ', 53: 'න', 54: 'ඳ', 55: 'ප', 56: ' '}

model = load_model("model.h5")


def preProcessing(img):
    img = cv2.resize(img, (32, 32))
    img = img / 255
    return img


image_path = "TestingImages/aaa - Copy.jpeg"

binary = preprocess_image(image_path)
row_projection = row_sum(binary)
col_projection = col_sum(binary)

row_segments = find_row_segments(row_projection, 10, 30)
col_segments = find_row_segments(col_projection, 5, 30)

text = ""
char_count = 0

# Create a copy of the original image to draw rectangles on
original_image = cv2.imread(image_path)
marked_image = original_image.copy()

for row_segment in row_segments[:-1]:
    for col_segment in col_segments:
        cell = binary[row_segment[0]:row_segment[1], col_segment[0]:col_segment[1]]

        # Draw a rectangle around the Braille cell on the marked_image
        cv2.rectangle(marked_image, (col_segment[0], row_segment[0]),
                      (col_segment[1], row_segment[1]), (0, 255, 0), 1)

        # Check if the cell is empty
        if cell.size == 0:
            print(f"Empty cell found at row_segment: {row_segment}, col_segment: {col_segment}")
            continue

        # Save the segmented character image
        cv2.imwrite(os.path.join(segmented_characters_folder, f'segmented_char_{char_count}.png'), cell)

        regenerated_image = regenerate_braille_dots(cell)

        # Save the regenerated character image
        cv2.imwrite(os.path.join(regenerated_characters_folder, f'regenerated_char_{char_count}.png'),
                    regenerated_image)

        grid_image = create_braille_grid(regenerated_image)

        img = preProcessing(grid_image)
        img = img.reshape(1, 32, 32, 1)

        classIndex = np.argmax(model.predict(img), axis=-1)[0]

        predictedLetter = labelDictionary.get(classIndex)
        if predictedLetter:
            text += predictedLetter
        else:
            text += "?"

        print("Identified Text:")
        print(text)

        char_count += 1

# Display the marked image with rectangles around Braille cells
cv2.imshow("Marked Image", marked_image)
cv2.waitKey(0)
cv2.destroyAllWindows()