from ImageProcessor import ImageProcessor
from GameEnv import GameEnv
from time import sleep

if __name__ == "__main__":
    width = 1200
    height = 900
    img_processor = ImageProcessor("map2.png", resize=[width, height])

    segments = img_processor.find_contour_segments()

    game_env = GameEnv(game_width=width, game_height=height, car_image_path="car1.png")
    game_env.initialize_game_data(segments, show_game=True)
    sleep(50)
