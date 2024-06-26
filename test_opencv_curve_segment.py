import cv2
import pygame

# Initialize Pygame
pygame.init()

# Load the image
image = cv2.imread("map1.png")

# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply thresholding to create a binary track
_, binary_track = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)

# Find contours in the binary track
contours, _ = cv2.findContours(binary_track, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Initialize an empty list to store the line segments
line_segments = []

# Find the two largest contours
largest_contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]
print(len(largest_contours))
# Loop through the largest contours to extract the line segments
for contour in largest_contours:
    # Approximate the contour to a polygon with a maximum of 4 edges
    approx = cv2.approxPolyDP(contour, 4, True)

    # Loop through the edges of the polygon to extract the line segments
    for i in range(len(approx)):
        # Get the coordinates of the current and next points
        p1 = approx[i][0]
        p2 = approx[(i + 1) % len(approx)][0]

        # Add the line segment to the list
        line_segments.append((p1[0], p1[1], p2[0], p2[1]))

# Create a Pygame window
screen = pygame.display.set_mode((image.shape[1], image.shape[0]))
print(len(line_segments))

font = pygame.font.Font(None, 36)

# Render the "!" symbol


# Update the Pygame display
pygame.display.update()
# Draw the line segments on the Pygame window
for i, line_segment in enumerate(line_segments):
    pygame.draw.line(
        screen,
        (0, 255 if i % 2 == 0 else 100, 0),
        (line_segment[0], line_segment[1]),
        (line_segment[2], line_segment[3]),
        2,
    )
    text = font.render(f"{i}", True, (255, 0, 0))

    # Get the text's rectangular dimensions
    text_rect = text.get_rect()

    # Center the text on the red dot's coordinates
    text_rect.center = (line_segment[0], line_segment[1])

    screen.blit(text, text_rect)

# Update the Pygame display
pygame.display.update()

while True:
    pass
