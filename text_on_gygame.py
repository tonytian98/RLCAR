import cv2
import numpy as np
import pygame


# Initialize Pygame
pygame.init()

# Create a Pygame window
screen = pygame.display.set_mode((image.shape[1], image.shape[0]))

# Load the font
font = pygame.font.Font(None, 36)

# Render the "!" symbol
text = font.render("!", True, (255, 0, 0))

# Get the text's rectangular dimensions
text_rect = text.get_rect()

# Center the text on the red dot's coordinates
text_rect.center = (x, y)


screen.blit(text, text_rect)

# Update the Pygame display
pygame.display.update()

# Wait for a key press to close the Pygame window
pygame.event.wait()
while True:
    pass
