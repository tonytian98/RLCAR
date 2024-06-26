import pygame
import math
import sys
import numpy as np
from time import sleep

# Initialize Pygame
pygame.init()

# Screen dimensions
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Car Racing Game")

# Load the track image
track = pygame.image.load("processed_map1.png")

# Car settings
car = pygame.image.load("car.png")
car = pygame.transform.scale(car, (15, 10))
car_x, car_y = 200, 200
car_speed = 5
car_angle = -90 + 360 * 3

# Ray settings
num_rays = 5
ray_length = 100
ray_color = (0, 0, 255)
ray_width = 2


def rotate_center(image, angle):
    """Rotate an image while keeping its center."""
    rect = image.get_rect()
    rotated_image = pygame.transform.rotate(image, angle)
    rotated_rect = rotated_image.get_rect(center=rect.center)
    return rotated_image, rotated_rect


max_forward_speed = 4
max_backward_speed = 2
car_speed = 0
acceleration = 0.1
# Game loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    keys = pygame.key.get_pressed()
    if keys[pygame.K_LEFT]:
        car_angle += 5
    if keys[pygame.K_RIGHT]:
        car_angle -= 5
    # Add these variables to the global scope

    # Modify the key press handling section
    if keys[pygame.K_UP]:
        if car_speed < max_forward_speed:
            car_speed += acceleration

    if keys[pygame.K_DOWN]:
        if car_speed > -max_backward_speed:
            car_speed -= acceleration

    car_x += car_speed * np.cos(np.radians(car_angle))
    car_y -= car_speed * np.sin(np.radians(car_angle))

    car_rect = car.get_rect(center=(car_x, car_y))
    track_color = track.get_at((int(car_x), int(car_y)))
    if track_color == (0, 255, 0):  # RGB value for green
        running = False

    # Draw rays
    rays = []
    num_rays_side = math.ceil(num_rays / 2)
    ray_range = 100
    for j in [-1, 0, 1]:
        for i in range(1, num_rays_side):
            if j == 0:
                angle = car_angle
                break
            angle = car_angle + -j * ray_range / (num_rays_side - 1) * i
            dx = ray_length * np.cos(np.radians(angle))
            dy = -ray_length * np.sin(np.radians(angle))
            ray_start = (car_x, car_y)
            ray_end = (car_x + dx, car_y + dy)
            ray_color = (0, 0, 255)
            ray_width = 3
            ray = pygame.draw.line(screen, ray_color, ray_start, ray_end, ray_width)
            rays.append(ray)

    # Modify the drawing section
    if car_speed >= 0:
        rotated_car, car_rect = rotate_center(car, car_angle)
    else:  # car_speed < 0
        rotated_car, car_rect = rotate_center(
            pygame.transform.flip(car, False, True), car_angle
        )
    screen.fill((0, 0, 0))
    screen.blit(track, (0, 0))

    screen.blit(
        rotated_car, (car_x - car_rect.width // 2, car_y - car_rect.height // 2)
    )

    # Display car speed on the top left corner
    speed_text = pygame.font.Font(None, 30).render(
        f"Speed: {car_speed}", True, (255, 255, 255)
    )
    screen.blit(speed_text, (10, 10))

    # Display ray lengths on the top right corner
    for i, ray in enumerate(rays):
        length_text = pygame.font.Font(None, 30).render(
            f"Ray {i+1}: {ray.width}", True, (255, 255, 255)
        )
        screen.blit(length_text, (WIDTH - length_text.get_width() - 10, 10 + 30 * i))

    pygame.display.flip()
    pygame.time.Clock().tick(60)

pygame.quit()
sys.exit()
