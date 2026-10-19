import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from dqn.replay_buffer import ReplayBuffer


class DQNNetwork(nn.Module):
    """
    Neural network used to approximate Q(s, a).
    """

    def __init__(self, state_size, action_size):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(state_size, 64),
            nn.ReLU(),

            nn.Linear(64, 64),
            nn.ReLU(),

            nn.Linear(64, action_size),
        )

    def forward(self, state):

        return self.network(state)


class DQNAgent:

    def __init__(
        self,
        state_size,
        action_size,
        learning_rate=0.001,
        gamma=0.95,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.05,
        replay_capacity=10000,
        batch_size=32,
    ):

        self.state_size = state_size
        self.action_size = action_size

        # -----------------------------------------
        # Hyperparameters
        # -----------------------------------------

        self.gamma = gamma

        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self.batch_size = batch_size

        # -----------------------------------------
        # Device
        # -----------------------------------------

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        # -----------------------------------------
        # Online Network
        # -----------------------------------------

        self.policy_network = DQNNetwork(
            state_size,
            action_size,
        ).to(self.device)

        # -----------------------------------------
        # Target Network
        # -----------------------------------------

        self.target_network = DQNNetwork(
            state_size,
            action_size,
        ).to(self.device)

        # Initially both networks are identical.
        self.target_network.load_state_dict(
            self.policy_network.state_dict()
        )

        self.target_network.eval()

        # -----------------------------------------
        # Optimizer
        # -----------------------------------------

        self.optimizer = optim.Adam(
            self.policy_network.parameters(),
            lr=learning_rate,
        )

        # -----------------------------------------
        # Loss
        # -----------------------------------------

        self.loss_function = nn.SmoothL1Loss()

        # -----------------------------------------
        # Replay Buffer
        # -----------------------------------------

        self.replay_buffer = ReplayBuffer(
            capacity=replay_capacity
        )

    def preprocess_state(self, state):
        """
        Normalize traffic state values.

        Queue values:
            divide by 100

        Waiting values:
            divide by 1000

        Phase:
            already between 0 and 1
        """

        state = np.asarray(
            state,
            dtype=np.float32,
        ).copy()

        state[0] /= 100.0
        state[1] /= 100.0

        state[2] /= 1000.0
        state[3] /= 1000.0

        return state

    def choose_action(self, state):
        """
        Epsilon-greedy action selection.
        """

        # Exploration
        if random.random() < self.epsilon:

            return random.randrange(
                self.action_size
            )

        # Exploitation
        state = self.preprocess_state(state)

        state_tensor = torch.tensor(
            state,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        with torch.no_grad():

            q_values = self.policy_network(
                state_tensor
            )

        return int(
            torch.argmax(q_values, dim=1).item()
        )

    def remember(
        self,
        state,
        action,
        reward,
        next_state,
        done,
    ):
        """
        Store an experience.
        """

        state = self.preprocess_state(state)
        next_state = self.preprocess_state(next_state)

        self.replay_buffer.add(
            state,
            action,
            reward,
            next_state,
            done,
        )

    def train_step(self):
        """
        Train the policy network using
        a random mini-batch from replay memory.
        """

        if len(self.replay_buffer) < self.batch_size:
            return None

        batch = self.replay_buffer.sample(
            self.batch_size
        )

        states = np.array(
            [experience[0] for experience in batch],
            dtype=np.float32,
        )

        actions = np.array(
            [experience[1] for experience in batch],
            dtype=np.int64,
        )

        rewards = np.array(
            [experience[2] for experience in batch],
            dtype=np.float32,
        )

        next_states = np.array(
            [experience[3] for experience in batch],
            dtype=np.float32,
        )

        dones = np.array(
            [experience[4] for experience in batch],
            dtype=np.float32,
        )

        # Convert to PyTorch tensors.
        states = torch.tensor(
            states,
            dtype=torch.float32,
            device=self.device,
        )

        actions = torch.tensor(
            actions,
            dtype=torch.long,
            device=self.device,
        )

        rewards = torch.tensor(
            rewards,
            dtype=torch.float32,
            device=self.device,
        )

        next_states = torch.tensor(
            next_states,
            dtype=torch.float32,
            device=self.device,
        )

        dones = torch.tensor(
            dones,
            dtype=torch.float32,
            device=self.device,
        )

        # -----------------------------------------
        # Current Q-values
        # -----------------------------------------

        current_q_values = self.policy_network(
            states
        )

        current_q_values = current_q_values.gather(
            1,
            actions.unsqueeze(1),
        ).squeeze(1)

        # -----------------------------------------
        # Target Q-values
        # -----------------------------------------

        with torch.no_grad():

            next_q_values = self.target_network(
                next_states
            )

            max_next_q_values = torch.max(
                next_q_values,
                dim=1,
            ).values

            target_q_values = rewards + (
                self.gamma
                * max_next_q_values
                * (1.0 - dones)
            )

        # -----------------------------------------
        # Calculate loss
        # -----------------------------------------

        loss = self.loss_function(
            current_q_values,
            target_q_values,
        )

        # -----------------------------------------
        # Backpropagation
        # -----------------------------------------

        self.optimizer.zero_grad()

        loss.backward()

        # Prevent excessively large gradients.
        torch.nn.utils.clip_grad_norm_(
            self.policy_network.parameters(),
            max_norm=1.0,
        )

        self.optimizer.step()

        return loss.item()

    def update_target_network(self):
        """
        Copy policy-network weights into
        the target network.
        """

        self.target_network.load_state_dict(
            self.policy_network.state_dict()
        )

    def decay_epsilon(self):
        """
        Reduce exploration after each episode.
        """

        self.epsilon = max(
            self.epsilon_min,
            self.epsilon * self.epsilon_decay,
        )

    def save(self, path):
        """
        Save the trained policy network.
        """

        torch.save(
            self.policy_network.state_dict(),
            path,
        )

    def load(self, path):
        """
        Load a trained policy network.
        """

        state_dict = torch.load(
            path,
            map_location=self.device,
        )

        self.policy_network.load_state_dict(
            state_dict
        )

        self.target_network.load_state_dict(
            state_dict
        )