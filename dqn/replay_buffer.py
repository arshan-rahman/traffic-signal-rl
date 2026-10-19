from collections import deque
import random


class ReplayBuffer:
    """
    Stores past experiences for DQN training.

    Each experience contains:

        state
        action
        reward
        next_state
        done
    """

    def __init__(self, capacity=10000):

        self.buffer = deque(maxlen=capacity)

    def add(
        self,
        state,
        action,
        reward,
        next_state,
        done,
    ):
        """
        Add one experience to the buffer.
        """

        experience = (
            state,
            action,
            reward,
            next_state,
            done,
        )

        self.buffer.append(experience)

    def sample(self, batch_size):
        """
        Randomly sample a batch of experiences.
        """

        return random.sample(
            self.buffer,
            batch_size,
        )

    def __len__(self):
        """
        Return number of stored experiences.
        """

        return len(self.buffer)