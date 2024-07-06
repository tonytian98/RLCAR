import torch
from torch import nn
from ShapelyEnv import ShapeEnv
import numpy as np
import copy
from collections import deque
import random

from pytorch_lightning import LightningModule
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data.dataset import IterableDataset
from torch.utils.data import DataLoader


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
        optim=AdamW,  # optimizer that updates the model parameters
        gamma=0.99,  # discount factor for accumulating rewards
        eps_start=1.0,  # starting epsilon for epsilon greedy policy
        eps_end=0.15,  # ending epsilon for epsilon greedy policy
        eps_last_episode=100,
        samples_per_epoch=1_000,
        sync_rate=10,  # number of epoches before we update the policy network using the target network
    ):
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
    def play_episode(self, epsilon: float = 0.0):
        self.env.reset()
        state = self.env.get_state()
        game_over = False
        while not game_over:
            action = self.env.epsilon_greedy(epsilon)
            next_state, reward, game_over = self.env.step(action)
            experience = (state, action, reward, game_over, next_state)
            self.buffer.append(experience)
            state = next_state

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
        states, actions, rewards, game_overs, next_states = batch
        pass


if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print("cuda:", torch.cuda.is_available())
    num_gpus = torch.cuda.device_count()

    dqn = DQN(4, [4], 4)
    print(dqn.net)
