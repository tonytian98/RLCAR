from Car import Car
from ImageProcessor import ImageProcessor
from ShapelyEnv import ShapeEnv
import numpy as np
import copy
from collections import deque
import random

import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data.dataset import IterableDataset
from torch.utils.data import DataLoader
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import EarlyStopping


class ActionSpace:
    def __init__(self, descriptive_actions: list[str]):
        """
        Initialize the ActionSpace object with descriptive actions.

        Parameters:
        descriptive_actions (list[str]): A list of strings representing the descriptive actions.

        Returns:
        None: It will generate the action space in the form of [0, 1, ..., len(descriptive actions)].
        """
        self.descriptive_actions = descriptive_actions
        self.actions = [i for i in range(len(descriptive_actions))]
        self.n = len(self.actions)

    def descriptive_actions_mapping(self, i):
        return self.descriptive_actions[i]

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
        print([f"{i}: {self.descriptive_actions_mapping(i)}" for i in range(self.n)])


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


class ReplayBuffer:
    def __init__(self, capacity):
        """
        Initialize a new instance of ReplayBuffer.

        Parameters:
        capacity (int): The maximum number of experiences that can be stored in the buffer.
            When the buffer is full, older experiences will be discarded to make space for new ones.

        Returns:
        None: It initializes the ReplayBuffer instance.
            Use .sample(size) method as the getter method to get items inside the buffer.
        """
        self.buffer = deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)


class RLDataset(IterableDataset):
    def __init__(self, buffer, sample_size=400):
        self.buffer = buffer
        self.sample_size = sample_size

    def __iter__(self):
        for experience in self.buffer.sample(self.sample_size):
            yield experience


class RLEnv(ShapeEnv):
    def __init__(
        self,
        device: str,
        action_space: ActionSpace,
        hidden_sizes: list[int],
        width: int = 800,
        height: int = 600,
        show_game: bool = True,
        save_processed_track: bool = True,
        auto_config_car_start: bool = True,
    ):
        super().__init__(
            self,
            width,
            height,
            show_game,
            save_processed_track,
            auto_config_car_start,
        )
        self.device: str = device
        self.action_space: ActionSpace = action_space
        self.hidden_sizes: list[int] = hidden_sizes
        self.state_size = len(self.get_state())

        self.model = DQN(len(self.get_state_size()), hidden_sizes, self.action_space.n)

    def get_state(self):
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


class DeepQLearning(LightningModule):
    # Initialize.
    def __init__(
        self,
        env: RLEnv,  # game environment with an epsilon greedy policy
        capacity=100,  # capacity of the replay buffer
        batch_size=256,  # batch size for training
        lr=1e-3,  # learning rate for optimizer
        loss_fn=F.smooth_l1_loss,  # loss function
        optimizer=AdamW,  # optimizer that updates the model parameters
        gamma=0.99,  # discount factor for accumulating rewards
        eps_start=1.0,  # starting epsilon for epsilon greedy policy
        eps_end=0.15,  # ending epsilon for epsilon greedy policy
        eps_last_episode=100,  # number of episodes used to decay epsilon to eps_end
        samples_per_epoch=1_000,  # number of samples needed per training episode
        sync_rate=10,  # number of epochs before we update the policy network using the target network
    ):
        """
        Initialize a new instance of DeepQLearning.

        Parameters:
        env (RLEnv): The game environment with an epsilon greedy policy.
        capacity (int, optional): The capacity of the replay buffer. Defaults to 100.
        batch_size (int, optional): The batch size for training. Defaults to 256.
        lr (float, optional): The learning rate for optimizer. Defaults to 1e-3.
        loss_fn (function, optional): The loss function. Defaults to F.smooth_l1_loss.
        optimizer (function, optional): The optimizer that updates the model parameters. Defaults to AdamW.
        gamma (float, optional): The discount factor for accumulating rewards. Defaults to 0.99.
        eps_start (float, optional): The starting epsilon for epsilon greedy policy. Defaults to 1.0.
        eps_end (float, optional): The ending epsilon for epsilon greedy policy. Defaults to 0.15.
        eps_last_episode (int, optional): The number of episodes used to decay epsilon to eps_end. Defaults to 100.
        samples_per_epoch (int, optional): The number of samples needed per training episode. Defaults to 1_000.
        sync_rate (int, optional): The number of epochs before we update the policy network using the target network. Defaults to 10.

        Returns:
        None: It initializes the DeepQLearning instance.
        """
        super().__init__()
        self.env = env
        # policy network
        self.q_net = self.env.model
        # target network
        self.target_q_net = copy.deepcopy(self.q_net)

        self.buffer = ReplayBuffer(capacity)

        self.save_hyperparameters()

        while len(self.buffer) < samples_per_epoch:
            self.play_episode(epsilon=eps_start)

    @torch.no_grad()
    def play_episode(self, epsilon: float = 0.0) -> float:
        self.env.reset()
        state = self.env.get_state()
        game_over = False
        total_return = 0
        while not game_over:
            action = self.env.epsilon_greedy(epsilon)
            next_state, reward, game_over = self.env.step(action)
            experience = (state, action, reward, game_over, next_state)
            self.buffer.append(experience)
            state = next_state
            total_return += reward
        return total_return

    def forward(self, x):
        return self.q_net(x)

    def configure_optimizers(self):
        return [AdamW(self.q_net.parameters(), lr=self.hparams.lr)]

    def train_dataloader(self):
        """
        This function is used to create a DataLoader object for training the model.

        The Dataset object is created from an RLDataset object, which is initialized with the replay buffer and a sample size.
        It yields from a list of randomly sampled experiences of size self.samples_per_epoch.

        The DataLoader object sends training data from Dataset in the size of self.batch_size to the model,
            until the total number of training data sent reaches self.samples_per_epoch

        Parameters:
        None

        Returns:
        DataLoader: A DataLoader object that yields batches of randomly sampled experiences from the replay buffer.
        """
        # yield list of randomly sampled experiences of length self.hparams.samples_per_epoch

        dataset = RLDataset(self.buffer, sample_size=self.hparams.samples_per_epoch)
        return DataLoader(dataset, batach_size=self.hparams.batch_size)

    def training_step(self, batch, batch_idx):
        """
        Performs a single training step.

        This method calculates the Q-value for the current state-action pairs,
        computes the expected Q-value for the next state-action pairs, and then
        calculates the loss between the current Q-value and the expected Q-value.
        The loss is then logged for monitoring purposes.

        Parameters:
        batch (tuple): A tuple containing the batch of training data.
            The tuple contains the following elements:
            - states (torch.Tensor): A tensor representing the states.
            - actions (torch.Tensor): A tensor representing the actions.
            - rewards (torch.Tensor): A tensor representing the rewards.
            - game_overs (torch.Tensor): A tensor representing the game over status.
            - next_states (torch.Tensor): A tensor representing the next states.

        batch_idx (int): The index of the batch within the epoch.

        Returns:
        torch.Tensor: The loss value for the current batch.
        """
        states, actions, rewards, game_overs, next_states = batch
        actions = actions.unsqueeze(1)
        rewards = rewards.unsqueeze(1)
        game_overs = game_overs.unsqueeze(1)

        # Q-value (Action-Value): Represents the value of taking a specific action in a specific state.

        # It gets the value of the 'actions' that was taken in 'states' [ [q(ai,si)], [q(aj,sj)], ...  ]
        state_action_value = self.q_net(states).gather(1, actions)
        next_action_value, _ = self.target_q_net(next_states).max(dim=1, keepdim=True)
        next_action_value[game_overs] = 0.0  # set the value of terminal states to 0
        expected_state_action_value = rewards + self.hparams.gamma * next_action_value
        loss = self.hparams.loss_fn(state_action_value, expected_state_action_value)
        self.log("episode/Q-Error", loss)
        return loss

    def training_epoch_end(self, training_step_outputs):
        """
        This function is called at the end of each training epoch.
        It updates the epsilon value for the epsilon-greedy policy,
        plays an episode with the updated epsilon value,
        logs the episode return, and synchronizes the target network with the policy network at sync_rate.

        Parameters:
        training_step_outputs: a dictionary with the values returned by the training_step method

        Returns:
        None
        """
        epsilon = max(
            self.hparams.eps_end,
            self.hparams.eps_start - self.current_epoch / self.hparams.eps_last_episode,
        )
        episode_return = self.play_episode(epsilon=epsilon)
        self.log("episode/Return", episode_return)
        if self.current_epoch % self.hparams.sync_rate == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())


if __name__ == "__main__":
    width = 800
    height = 600

    # object creation
    img_processor = ImageProcessor("map1.png", resize=[width, height])
    car = Car(650, 100, 0, 90)
    game_env = ShapeEnv(
        width,
        height,
        show_game=True,
        save_processed_track=True,
        auto_config_car_start=True,
    )
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    num_gpus = torch.cuda.device_count()

    algo = DeepQLearning(game_env)

    trainer = Trainer(
        gpus=num_gpus,
        max_epochs=10_000,
        callbacks=EarlyStopping(monitor="episode/Return", mode="max", patience=500),
    )
    trainer.fit(algo)
