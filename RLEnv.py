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
        maximum_steps: int = 5000,
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
        self.visited_segmented_track_indices = [0]
        self.state_size = len(self.get_state())
        # An ML model that represents the driver
        self.model = DQN(self.get_state_size(), hidden_sizes, self.action_space.n)
        
        self.avg_distance_to_next_segment, self.std_distance_to_next_segment = self.get_segment_distance_avg_std()
        self.avg_ray_length, self.avg_ray_angle = self.get_ray_length_avg_std()
        
        self.avg_car_speed, self.std_car_speed = (self.car.max_backward_speed + self.car.max_forward_speed) / 2, 1
        
        self.avg_angle_difference, self.std_angle_difference = 0, 180 
        
        self.maximum_steps = maximum_steps
        self.current_step = 0

    
    def get_ray_length_avg_std(self) -> tuple[float, float]:
        arr = np.array(self.get_ray_lengths())
        return np.mean(arr), np.std(arr)
    def get_segment_distance_avg_std(self) -> tuple[float, float]:
        distances = []
        self.get
        for i in range(len(self.get_number_of_segmented_tracks())):
            distances.append(self.segmented_track_in_order[i].centroid.distance(
                self.segment_track_in_order[(i+1) % self.get_number_of_segmented_tracks()].centroid))
        arr = np.array(distances)
        return np.mean(arr), np.std(arr)
            
            
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
        if isinstance(value,list):
            return [self.standardize(x, mean, std) for x in value]
        return (value - mean) / std

    def get_state(self):
        
        car_speed_standardized = self.standardize(self.get_car_speed(), self.avg_car_speed ,self.std_car_speed)
        angle_difference_standardized = self.standardize(
            self.get_difference_car_angle_target_angle(),  self.avg_angle_difference , self.std_angle_difference
        )
        target_segment_index = (self.current_segmented_track_index + 1) % len(self.segmented_track_in_order)
        
        distance_to_next_segment = self.get_car_distance_to_segmented_track(
                target_segment_index
            )
        distance_to_next_segment_standardized = self.standardize( distance_to_next_segment,
                                                                 self.avg_distance_to_next_segment,
                                                                 self.std_distance_to_next_segment)
        
        ray_lengths_standardized = self.standardize(self.get_ray_lengths(), self.avg_ray_length, self.avg_ray_angle)
        # If the car arrived in the next target segment, but not segment 0 (final destination) 
        if distance_to_next_segment == 0 and target_segment_index != 0:
            self.current_segmented_track_index =  target_segment_index 
            self.visited_segmented_track_indices.append(self.current_segmented_track_index)
            reached_new_segment_reward = 

        return [car_speed_standardized, angle_difference_standardized, distance_to_next_segment_standardized] + ray_lengths_standardized

    def get_state_size(self):
        return self.state_size
    
    def execute_car_logic(self, action:int):
        descriptive_action:str = self.action_space.descriptive_action_by_action(action)
        if  descriptive_action == "accelerate":
            self.car.accelerate()
        elif descriptive_action == "decelerate":
            self.car.decelerate()
        elif descriptive_action == "steer_left":
            self.car.steer_left()
        elif descriptive_action == "steer_right":
            self.car.steer_right()

    def angle_difference_reward_function(self, angle_difference_normalized):
        abs_angle_difference = abs(angle_difference_normalized)
        if abs_angle_difference >= self.standardize(0, self.avg_angle_difference, self.std_angle_difference) and abs_angle_difference < self.standardize(60, self.avg_angle_difference, self.std_angle_difference):
            # (- 2/3, 0]
            return abs_angle_difference * abs_angle_difference * abs_angle_difference
        if abs_angle_difference >= self.standardize(60, self.avg_angle_difference, self.std_angle_difference) and abs_angle_difference < self.standardize(1200, self.avg_angle_difference, self.std_angle_difference):
            # (]
            return - 3 *abs_angle_difference  * abs_angle_difference
        if abs_angle_difference >= self.standardize(120, self.avg_angle_difference, self.std_angle_difference) and abs_angle_difference < self.standardize(180, self.avg_angle_difference, self.std_angle_difference):
            # (]
            return - 3 * abs_angle_difference
        
    def car_speed_reward_function(self, car_speed_normalized):
        # [- max_forward_speed /2 , max_forward_speed /2] assuming max_back_speed = 0, e.g. [-2.5, 2.5]
        return car_speed_normalized
    def get_reward(self, state):
        car_speed_normalized = state[0]
        angle_difference_standardized = state[1]
        distance_to_next_segment_standardized = state[2]
        center_ray = state[-1]
        
        
            
    def step(self, action: int):
        """execute the action in game env and return the new state, reward, terminated, (truncated, info)"""
        #if self.show_game:
        #    self.draw_background()
        self.execute_car_logic(action)
        self.action_record.set_current_value(action)
        self.action_record.add_current_Value_to_record()  # print
        self.car.update_car_position()
        self.update_rays()
        if self.show_game:
            self.update_game_frame([self.car.get_shapely_point()] + self.rays)
        self.current_step += 1
        running = not self.game_end()
        listener.stop()

        self.action_record.save_record_to_txt()
        new_state = self.get_state()
        
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
            self.config_car_start()
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
        ActionSpace(["hold", "accelerate", "decelerate", "steer_left", "steer_right"]),
        [32],
        img_processor,
        car,
    )

    # environment setup

    # start game
    game_env.start_game_RL()
