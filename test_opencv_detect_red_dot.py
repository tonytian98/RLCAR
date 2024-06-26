import cv2
import numpy as np
import pygame

# Load the image
image = cv2.imread("red_dot_test.png")
img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

lower = np.array([155, 25, 0])
upper = np.array([179, 255, 255])
mask = cv2.inRange(img_hsv, lower, upper)
cv2.imshow("Track with Contours", mask)
cv2.waitKey(0)
cv2.destroyAllWindows()
# Find contours in the binary mask

contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(len(contours))
# Find the largest contour that represents the red dot
largest_contour = max(contours, key=cv2.contourArea)

# Find the minimum enclosing circle of the red dot
(x, y), radius = cv2.minEnclosingCircle(largest_contour)

# Convert the coordinates to integers
x = int(x)
y = int(y)
radius = int(radius)

# Initialize Pygame
pygame.init()

# Create a Pygame window
screen = pygame.display.set_mode((image.shape[1], image.shape[0]))

# Draw the red dot on the Pygame window
pygame.draw.circle(screen, (255, 0, 0), (x, y), radius)

# Update the Pygame display
pygame.display.update()

# Wait for a key press to close the Pygame window
pygame.event.wait()
while True:
    pass
