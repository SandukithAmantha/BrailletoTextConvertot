import cv2
import numpy as np


def draw_circle(image, center, radius, color, thickness):
    cv2.circle(image, center, radius, color, thickness)


def regenerate_braille_dots(binary_image, dot_radius=2, dot_thickness=-1):
    regenerated = np.zeros_like(binary_image)

    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        center = (int(x), int(y))
        radius = int(radius)
        if radius >= dot_radius - 1 and radius <= dot_radius + 1:
            draw_circle(regenerated, center, dot_radius, 255, dot_thickness)

    return regenerated


def create_braille_grid(regenerated_image, grid_rows=3, grid_cols=2, cell_width=10, cell_height=10):
    grid_image = np.zeros((grid_rows * cell_height, grid_cols * cell_width), dtype=np.uint8)
    contours, _ = cv2.findContours(regenerated_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        col = int(x // cell_width)
        row = int(y // cell_height)
        center_x = col * cell_width + cell_width // 2
        center_y = row * cell_height + cell_height // 2
        draw_circle(grid_image, (center_x, center_y), int(radius), 255, -1)

    return grid_image


# binary_image = cv2.imread("OutputImages/Char4.png", cv2.IMREAD_GRAYSCALE)
# regenerated_image = regenerate_braille_dots(binary_image)
# grid_image = create_braille_grid(regenerated_image)
# cv2.imwrite("regenerated_braille_image.png", regenerated_image)
# cv2.imwrite("braille_grid_image.png", grid_image)
