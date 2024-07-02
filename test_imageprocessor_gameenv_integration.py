from ImageProcessor import ImageProcessor
from GameEnv import GameEnv
from time import sleep

if __name__ == "__main__":
    width = 1200
    height = 900
    img_processor = ImageProcessor("map1.png", resize=[width, height])

    segments = img_processor.find_contour_segments()

    game_env = GameEnv(game_width=width, game_height=height)
    game_env.set_walls(segments)
    game_env.set_reward_lines_from_walls()
    print(len(game_env.get_walls()))
    game_env.draw()
    sleep(50)
