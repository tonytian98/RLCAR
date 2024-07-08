from Car import Car
from ImageProcessor import ImageProcessor
from ShapelyEnv import ShapelyEnv
from datetime import datetime
from time import sleep
from pynput import keyboard
import os


class Record:
    def __init__(self, name: str, separator: str = ";", set_time_in_file_name=True):
        """
        Initialize a Record object.

        Parameters:
        name (str): The name of the record.
        separator (str): The character used to separate values in the record in the .txt file. Default is ';'.
        set_time_in_file_name (bool): A flag indicating whether to set the start_time. Default is True.

        Attributes:
        name (str): The name of the record.
        separator (str): The character used to separate values in the record.
        current_value (str): The current value of the record.
            Note:This current value is not recorded yet. Call add_current_value() to record it and to reset it to None
        record (list): A list to store the record values.
        start_time (str): The start time of the record, formatted as '_YYYY_MM_DD_HH_MM_SS'. Default is "".
        past_record_files (list[str]): A list that stores the names of the past record files.
        replay_record (list): A list to store the record values for replaying.
        """
        self.name: str = name
        self.current_value = None
        self.separator = separator
        self.record: list = []
        self.start_time: str = ""
        self.past_record_files: list[str] = []

        self.set_time_in_file_name = set_time_in_file_name

        self.dir_name = f"{name}_records"
        os.makedirs(self.dir_name, exist_ok=True)
        self.load_replay_records_in_dir()

    def set_current_value(self, value: str):
        """
        Sets the current value of the record.

        Parameters:
        value (str): The value to be set as the current value, which is not in the record yet.
            The value cannot contain the separator character.

        Raises:
        ValueError: If the value contains the separator character.
        """
        if self.separator in value:
            raise ValueError("Value cannot contain separator character")
        self.current_value = value

    def get_current_value(self):
        return self.current_value

    def clear_current_value(self):
        """set current value to None"""
        self.current_value = None

    def add_current_Value_to_record(self):
        """
        Adds the current value to the record list and clears the current value.
        """

        if len(self.record) == 0 and self.set_time_in_file_name:
            self.start_time = datetime.now().strftime("_%Y_%m_%d_%H_%M_%S")
        self.record.append(self.current_value)
        self.clear_current_value()

    def clear_current_record(self):
        """
        Clears the current record and resets the record list.

        This method clears the current value of the record and resets the record list to an empty list.
        It is called after saving the record to a text file, the save_record_to_txt() method to be exact.
        """
        self.clear_current_value()
        self.record = []

    def save_record_to_txt(self):
        """
        Saves the record to a text file.
        After saving, all current records will be cleared.
        Access this record using load_latest_record(idx) and get_replay_record()

        The method opens a text file with the name of the record and the start time (if set)
        and writes the record values separated by the separator character.

        Returns:
            str: The name of the saved text file.
        """
        saved_file_path = os.path.join(
            self.dir_name, self.name + self.start_time + ".txt"
        )
        with open(saved_file_path, "w") as f:
            text = self.separator.join(self.record)
            f.write(text)

        self.past_record_files.append(saved_file_path)
        self.clear_current_record()
        return saved_file_path

    def load_latest_record(self, offset: int = 0):
        """
        Loads the latest record from the past record files list.

        Parameters:
        offset (int): The offset from the latest record file. Default is 0.

            If offset is 0, it loads the latest record file.

            If offset is 1, it loads the second latest record file, and so on.

        Returns:
        None: The function sets the replay_record attribute of the Record object with the loaded record values.

        Raises:
        ValueError: If the offset is out of range.
        """
        if -offset < 1 - len(self.past_record_files) or offset < 0:
            raise ValueError("Offset is out of range")
        file_name = self.past_record_files[-1 - offset]
        self.set_replay_record_from_file(file_name)

    def get_replay_records(self):
        return self.replay_records

    def load_replay_records_in_dir(self):
        self.replay_records: list = [
            os.path.join(self.dir_name, f) for f in os.listdir(self.dir_name)
        ]

    def set_replay_record_from_file(self, file_name):
        """
        Sets the replay_record attribute of the Record object with the loaded record values from a given file.

        Parameters:
        file_name (str): The name of the file from which to load the record values.

        Returns:
        None: The function updates the replay_record attribute of the Record object.

        Raises:
        FileNotFoundError: If the specified file does not exist.
        """
        with open(file_name, "r") as f:
            self.replay_records = f.read().split(self.separator)


class RecordEnv(ShapelyEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.action_record = Record("actions")

    def keyboard_rule(self, key):
        """
        Handles keyboard inputs and performs corresponding actions on the car.

        Parameters:
        key (keyboard.Key): The key that was pressed on the keyboard.

        Returns:
        None: The function updates the self.car's position, angle and speed
        """
        if key == keyboard.Key.up:
            self.car.accelerate()
            self.action_record.set_current_value("UP")

        if key == keyboard.Key.down:
            self.car.decelerate()
            self.action_record.set_current_value("DOWN")

        if key == keyboard.Key.left:
            self.car.turn_left()
            self.action_record.set_current_value("LEFT")

        if key == keyboard.Key.right:
            self.car.turn_right()
            self.action_record.set_current_value("RIGHT")

        if key == keyboard.Key.space:
            self.reset()
            self.action_record.set_current_value("SPACE")

    def start_game_with_record(self):
        """
        Starts the game loop.

        The game loop continuously updates the car's position, rays, and checks for game end conditions.
        If the game end condition is met (car collides with the track), the game ends.

        Parameters:
        None

        Returns:
        None
        """
        running = True

        listener = keyboard.Listener(
            on_press=self.keyboard_rule,
        )
        listener.start()
        if self.show_game:
            self.draw_background()
        counter: int = 0
        while running:
            if self.action_record.get_current_value() is None:
                self.action_record.set_current_value("NO_ACTION")
            self.action_record.add_current_Value_to_record()  # print
            self.car.update_car_position()
            self.update_rays()
            if self.show_game:
                self.update_game_frame([self.car.get_shapely_point()] + self.rays)
            running = not self.game_end()

            counter += 1
            print(f"Counter: {counter}")
        listener.stop()

        self.action_record.save_record_to_txt()

    def replay(self, offset: int = 0):
        actions = self.action_record.get_replay_records()
        if len(actions) == 0:
            raise ValueError("Replay record is not set")

        if self.show_game:
            self.draw_background()
        for i, action in enumerate(actions):
            if "UP" in action:
                self.car.accelerate()
            elif "DOWN" in action:
                self.car.decelerate()
            elif "LEFT" in action:
                self.car.turn_left()
            elif "RIGHT" in action:
                self.car.turn_right()
            elif "SPACE" in action:
                self.reset()

            self.car.update_car_position()
            self.update_rays()
            if self.show_game:
                self.update_game_frame([self.car.get_shapely_point()] + self.rays)
            if self.game_end() and i == len(actions) - 1:
                print("Replay ended.")
                break
            if game_env.game_end() and i != len(actions) - 1:
                raise ValueError("Game ended early.")
            if not game_env.game_end() and i == len(actions) - 1:
                raise ValueError("Game did not end after all actions. It should.")


if __name__ == "__main__":
    width = 800
    height = 600

    # object creation
    img_processor = ImageProcessor("map1.png", resize=[width, height])
    car = Car(650, 100, 0, 90)
    game_env = RecordEnv(
        width,
        height,
        img_processor,
        car,
        show_game=True,
        save_processed_track=True,
        auto_config_car_start=True,
    )

    # start game
    game_env.start_game_with_record()
    game_env.reset()
    sleep(2)
    game_env.action_record.load_latest_record()
    game_env.replay()
