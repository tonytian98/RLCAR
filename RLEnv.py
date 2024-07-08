from ShapelyEnv import ShapelyEnv
import torch
from Car import Car
from ImageProcessor import ImageProcessor
from torch import nn
import numpy as np
from RecordEnv import RecordEnv


class ActionSpace:
    def __init__(self, descriptive_actions: list[str]):
        """
        Initialize the ActionSpace object with descriptive actions.

        Parameters:
        descriptive_actions (list[str]): A list of strings representing the descriptive actions.

        Returns:
        None: It will generate the action space in the form of [0, 1, ..., len(descriptive actions)].
            Action space consists of integers(actions) from 0 to (len(descriptive actions) - 1).
        """
        self.descriptive_actions = descriptive_actions
        self.actions = [i for i in range(len(descriptive_actions))]
        self.n = len(self.actions)

    def descriptive_action_by_action(self, i):
        """
        Returns the descriptive action corresponding to the given index.

        Parameters:
        i (int): The index of the action.

        Returns:
        str: The descriptive action corresponding to the given index.
        """
        return self.descriptive_actions[i]

    def action_by_descriptive_action(self, descriptive_action):
        """
        Returns the index of the given descriptive action in the action space.

        Parameters:
        descriptive_action (str): The descriptive action for which the index is to be found.

        Returns:
        int: The index of the given descriptive action in the action space. If the descriptive action is not found, it returns None.
        """
        return self.descriptive_actions.index(descriptive_action)

    def sample(self):
        """
        This function is used to sample an action from the action space, which is a list [0, 1, ...].

        Parameters:
        None

        Returns:
        int: A random action from the action space.
        """
        return self.actions[torch.randint(0, self.n, (1,)).item()]

    def __len__(self):
        return self.n

    def __str__(self):
        print([f"{i}: {self.descriptive_action_by_action(i)}" for i in range(self.n)])


class DQN(nn.Module):
    def __init__(self, obs_size: int, hidden_sizes: list[int], n_actions: int):
        super().__init__()
        hidden_layers = []
        if len(hidden_sizes) > 1:
            for i in range(len(hidden_sizes) - 1):
                hidden_layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
                hidden_layers.append(nn.ReLU())

        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_sizes[0]),
            nn.ReLU(),
            *hidden_layers,
            nn.Linear(hidden_sizes[-1], n_actions),
        )

    def forward(self, x):
        return self.net(x.float())

    def get_net(self):
        return self.net


class RLEnv(RecordEnv):
    def __init__(
        self,
        device: str,
        action_space: ActionSpace,
        hidden_sizes: list[int],
        img_processor: ImageProcessor,
        car: Car,
        width: int = 800,
        height: int = 600,
        show_game: bool = True,
        save_processed_track: bool = True,
    ):
        """
        Initialize a new instance of RLEnv.

        Parameters:
        device (str): The device to run the model on. It can be either 'cuda:0' or 'cpu'.
        action_space (ActionSpace): The action space object that defines the available actions.
        hidden_sizes (list[int]): A list of integers representing the sizes of the output of the hidden layers.
        width (int, optional): The width of the game environment. Defaults to 800.
        height (int, optional): The height of the game environment. Defaults to 600.
        show_game (bool, optional): A flag indicating whether to show the game environment. Defaults to True.
        save_processed_track (bool, optional): A flag indicating whether to save the processed track. Defaults to True.


        Returns:
        None: It initializes the RLEnv instance.
        """
        super().__init__(
            width,
            height,
            img_processor,
            car,
            show_game,
            save_processed_track,
            auto_config_car_start=True,
        )
        self.device: str = device
        self.action_space: ActionSpace = action_space
        self.hidden_sizes: list[int] = (
            hidden_sizes  # sizes of the output of the hidden layers
        )

        # Because auto_config_car_start is hard coded to be True in super().__init__,
        # the car's initial position is always the centroid of the first track segment.
        self.current_segmented_track_index = 0
        self.state_size = len(self.get_state())
        print(self.state_size)
        # An ML model that represents the driver
        self.model = DQN(self.get_state_size(), hidden_sizes, self.action_space.n)

    #
    def get_target_angle(self) -> float:
        """
        Target angle is the angle of the line that connects the car and centroid of the next track segment,
        representing a sense of direction for the driver to modify the car's angle.
        Parameters:
        None

        Returns:
        float: returns the target angle [0, 360).
        """
        return self.calculate_line_angle(
            self.car.get_shapely_point(),
            self.segmented_track_in_order[
                self.current_segmented_track_index + 1
            ].centroid,
        )

    def get_difference_car_angle_target_angle(self):
        """
        Calculate the difference between the car's current angle and the target angle.
        A value of 0 means the car is heading straight to the next track segment (regardless of walls in-between them).
        (-180, 0) means car need to turn left (counter-clockwise), increasing car angle
        (0, 180] means car need to turn right (clockwise), decreasing car angle

        Returns:
            float: The difference between the car's current angle and the target angle (-180, 180].
        """
        difference = self.car.get_car_angle() - self.get_target_angle()
        if difference > 180:
            return difference - 360
        if difference <= -180:
            return difference + 360
        return difference

    def normalize(self, value, min_val, max_val):
        return (value - min_val) / (max_val - min_val)

    def standardize(self, value, mean, std):
        return (value - mean) / std

    def get_state(self):
        return [self.get_difference_car_angle_target_angle()] + self.get_ray_lengths()

    def get_normalized_state(self):
        pass

    def get_state_size(self):
        return self.state_size

    def step(self, action: int):
        """excute the action in game env and return the new state, reward, terminated, (truncated, info)"""
        pass

    def record(
        self, episode_trigger
    ):  # episode_trigger is a lambda function to tell which episode to record
        ###TO DO: record the episode's actions and rewards
        pass

    def epsilon_greedy(self, epsilon=0.0):
        if np.random.random() < epsilon:
            action = self.action_space.sample()
        else:
            state = torch.tensor([self.get_state()]).to(self.device)
            q_values = self.model.get_net()(state)
            _, action = torch.max(q_values, dim=1)
            action = int(action.item())
        return action

    def start_game_RL(self):
        """
        Starts the game loop in RL environment.

        The game loop continuously updates the car's position, rays, and checks for game end conditions.
        If the game end condition is met (car collides with the track), the game ends.

        Parameters:
        None

        Returns:
        None
        """
        if self.auto_config_car_start:
            self.config_car_start(181)
        if self.show_game:
            self.draw_background()
        print(self.car.get_car_angle())
        print(self.get_target_angle())
        print(self.get_difference_car_angle_target_angle())


if __name__ == "__main__":
    width = 800
    height = 600

    # object creation
    img_processor = ImageProcessor("map1.png", resize=[width, height])
    car = Car(650, 100, 0, 90)
    game_env = RLEnv(
        "cpu",
        ActionSpace(["hold", "accelerate", "brake", "steer_left", "steer_right"]),
        [32],
        img_processor,
        car,
    )

    # environment setup

    # start game
    game_env.start_game_RL()
