from collections import defaultdict
from pathlib import Path
import sys
import pickle
from pathlib import Path
import numpy as np

# Running this file directly sets sys.path[0] to q_learning/, so add the
# project root to make the sibling environment/ package importable.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from environment.simple_traffic_env import SimpleTrafficEnv


# ---------------------------------------------------------
# State Discretization
# ---------------------------------------------------------

def discretize_state(state):
    """
    Convert the continuous environment state into
    a small discrete state suitable for Q-learning.

    Original state:
        [NS queue, EW queue, NS waiting, EW waiting, phase]

    Discrete state:
        (NS queue bucket,
         EW queue bucket,
         NS waiting bucket,
         EW waiting bucket,
         phase)
    """

    ns_queue, ew_queue, ns_waiting, ew_waiting, phase = state

    # Queue buckets
    ns_queue_bucket = min(int(ns_queue // 5), 2)
    ew_queue_bucket = min(int(ew_queue // 5), 2)

    # Waiting-time buckets
    ns_waiting_bucket = min(int(ns_waiting // 10), 2)
    ew_waiting_bucket = min(int(ew_waiting // 10), 2)

    phase = int(phase)

    return (
        ns_queue_bucket,
        ew_queue_bucket,
        ns_waiting_bucket,
        ew_waiting_bucket,
        phase,
    )


# ---------------------------------------------------------
# Q-Learning Agent
# ---------------------------------------------------------

class QLearningAgent:

    def __init__(
        self,
        action_size,
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.05,
    ):
        self.action_size = action_size

        # Learning rate (alpha)
        self.learning_rate = learning_rate

        # Discount factor (gamma)
        self.discount_factor = discount_factor

        # Exploration probability
        self.epsilon = epsilon

        # Exploration decay
        self.epsilon_decay = epsilon_decay

        # Minimum exploration
        self.epsilon_min = epsilon_min

        # Q-table
        #
        # If a state has never been encountered,
        # it automatically gets zero Q-values.
        self.q_table = defaultdict(
            lambda: np.zeros(self.action_size)
        )

    def choose_action(self, state):
        """
        Epsilon-greedy action selection.

        With probability epsilon:
            Explore → random action

        Otherwise:
            Exploit → choose best known action
        """

        if np.random.random() < self.epsilon:

            # Exploration
            return np.random.randint(self.action_size)

        # Exploitation
        return int(np.argmax(self.q_table[state]))

    def update(
        self,
        state,
        action,
        reward,
        next_state,
        done,
    ):
        """
        Update the Q-value using the Q-learning equation.

        Q(s,a) = Q(s,a) + alpha[
            reward + gamma * max Q(s',a')
            - Q(s,a)
        ]
        """

        current_q = self.q_table[state][action]

        if done:
            target = reward

        else:
            best_next_q = np.max(
                self.q_table[next_state]
            )

            target = (
                reward
                + self.discount_factor * best_next_q
            )

        new_q = current_q + self.learning_rate * (
            target - current_q
        )

        self.q_table[state][action] = new_q

    def decay_epsilon(self):
        """
        Gradually reduce exploration.
        """

        self.epsilon = max(
            self.epsilon_min,
            self.epsilon * self.epsilon_decay,
        )


# ---------------------------------------------------------
# Training
# ---------------------------------------------------------

def train_q_learning(
    num_episodes=500,
):

    env = SimpleTrafficEnv()

    agent = QLearningAgent(
        action_size=env.action_space.n
    )

    episode_rewards = []

    print("=" * 60)
    print("Q-Learning Traffic Signal Training")
    print("=" * 60)

    print()
    print(f"Episodes: {num_episodes}")
    print()

    for episode in range(1, num_episodes + 1):

        state, info = env.reset()

        state = discretize_state(state)

        total_reward = 0.0

        terminated = False
        truncated = False

        while not terminated and not truncated:

            # Choose action
            action = agent.choose_action(state)

            # Interact with environment
            next_state, reward, terminated, truncated, info = (
                env.step(action)
            )

            next_state = discretize_state(next_state)

            # Update Q-table
            done = terminated or truncated

            agent.update(
                state,
                action,
                reward,
                next_state,
                done,
            )

            # Move to next state
            state = next_state

            total_reward += reward

        # Reduce exploration
        agent.decay_epsilon()

        episode_rewards.append(total_reward)

        # Display progress every 50 episodes
        if episode % 50 == 0:

            average_reward = np.mean(
                episode_rewards[-50:]
            )

            print(
                f"Episode {episode:4d} | "
                f"Average Reward: {average_reward:10.2f} | "
                f"Epsilon: {agent.epsilon:.3f}"
            )

        env.close()

    # -----------------------------------------
    # Save trained Q-table
    # -----------------------------------------

    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    q_table_path = models_dir / "q_table.pkl"

    with open(q_table_path, "wb") as file:
        pickle.dump(dict(agent.q_table), file)

    print()
    print(f"Q-table saved to: {q_table_path}")

    return agent, episode_rewards


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    agent, rewards = train_q_learning(
        num_episodes=500
    )

    print()
    print("=" * 60)
    print("Training Completed")
    print("=" * 60)

    print(
        f"Q-table states learned: {len(agent.q_table)}"
    )

    print(
        f"Final epsilon: {agent.epsilon:.3f}"
    )

    print(
        f"Final 50-episode average reward: "
        f"{np.mean(rewards[-50:]):.2f}"
    )

    print()
    print("Q-learning training completed successfully.")


if __name__ == "__main__":
    main()
