from ImageProcessor import ImageProcessor
from GameEnv import GameEnv
from time import sleep

if __name__ == "__main__":
    width = 1200
    height = 900
    img_processor = ImageProcessor("map1.png", resize=[width, height])

    segments = img_processor.find_contour_segments()

    game_env = GameEnv(
        game_width=width,
        game_height=height,
        car_image_path="car1.png",
        max_forward_speed=40,
        acceleration=6,
        turn_angle=10,
    )
    game_env.initialize_game_data(segments, show_game=True)
    game_env.draw(
        show_walls=True,
        show_reward_lines=True,
        show_reward_sequence=True,
        show_rays=True,
        show_speed=True,
    )
    game_env.start_game()
