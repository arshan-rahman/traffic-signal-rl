import pickle
from pathlib import Path
import sys

import numpy as np

# Running this file directly sets sys.path[0] to q_learning/, so add the
# project root to make the sibling environment/ package importable.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from environment.simple_traffic_env import SimpleTrafficEnv
from q_learning.q_learning_agent import discretize_state


def load_q_table():

    q_table_path = Path("models/q_table.pkl")

    if not q_table_path.exists():
        raise FileNotFoundError(
            "Q-table not found. "
            "Run q_learning_agent.py first."
        )

    with open(q_table_path, "rb") as file:
        q_table = pickle.load(file)

    return q_table


def run_random_episode(env):

    state, info = env.reset()

    total_reward = 0.0
    total_waiting = 0.0
    total_queue = 0.0

    terminated = False
    truncated = False

    while not terminated and not truncated:

        action = env.action_space.sample()

        state, reward, terminated, truncated, info = (
            env.step(action)
        )

        total_reward += reward

        total_waiting += (
            info["ns_waiting"]
            + info["ew_waiting"]
        )

        total_queue += (
            info["ns_queue"]
            + info["ew_queue"]
        )

    return {
        "reward": total_reward,
        "waiting": total_waiting,
        "queue": total_queue,
    }


def run_q_learning_episode(env, q_table):

    state, info = env.reset()

    state = discretize_state(state)

    total_reward = 0.0
    total_waiting = 0.0
    total_queue = 0.0

    terminated = False
    truncated = False

    while not terminated and not truncated:

        # -----------------------------------------
        # Greedy action selection
        # -----------------------------------------
        # No exploration during evaluation.
        # We always choose the best learned action.

        q_values = q_table.get(
            state,
            np.zeros(env.action_space.n)
        )

        action = int(np.argmax(q_values))

        # -----------------------------------------
        # Environment step
        # -----------------------------------------

        next_state, reward, terminated, truncated, info = (
            env.step(action)
        )

        next_state = discretize_state(next_state)

        state = next_state

        total_reward += reward

        total_waiting += (
            info["ns_waiting"]
            + info["ew_waiting"]
        )

        total_queue += (
            info["ns_queue"]
            + info["ew_queue"]
        )

    return {
        "reward": total_reward,
        "waiting": total_waiting,
        "queue": total_queue,
    }


def calculate_average(results):

    return {
        "reward": np.mean(
            [result["reward"] for result in results]
        ),
        "waiting": np.mean(
            [result["waiting"] for result in results]
        ),
        "queue": np.mean(
            [result["queue"] for result in results]
        ),
    }


def main():

    num_episodes = 20

    print("=" * 65)
    print("Q-Learning Evaluation")
    print("=" * 65)

    # -----------------------------------------
    # Load trained Q-table
    # -----------------------------------------

    q_table = load_q_table()

    print()
    print(f"Loaded Q-table states: {len(q_table)}")
    print(f"Evaluation episodes: {num_episodes}")

    # -----------------------------------------
    # Create environment
    # -----------------------------------------

    env = SimpleTrafficEnv()

    random_results = []
    q_learning_results = []

    # -----------------------------------------
    # Evaluate Random Agent
    # -----------------------------------------

    print()
    print("Evaluating Random Agent...")

    for _ in range(num_episodes):

        result = run_random_episode(env)

        random_results.append(result)

    # -----------------------------------------
    # Evaluate Q-Learning
    # -----------------------------------------

    print("Evaluating Q-Learning...")

    for _ in range(num_episodes):

        result = run_q_learning_episode(
            env,
            q_table
        )

        q_learning_results.append(result)

    env.close()

    # -----------------------------------------
    # Calculate averages
    # -----------------------------------------

    random_average = calculate_average(
        random_results
    )

    q_learning_average = calculate_average(
        q_learning_results
    )

    # -----------------------------------------
    # Display comparison
    # -----------------------------------------

    print()
    print("=" * 65)
    print("Performance Comparison")
    print("=" * 65)

    print()
    print(
        f"{'Metric':<25}"
        f"{'Random Agent':>18}"
        f"{'Q-Learning':>18}"
    )

    print("-" * 65)

    print(
        f"{'Average Reward':<25}"
        f"{random_average['reward']:>18.2f}"
        f"{q_learning_average['reward']:>18.2f}"
    )

    print(
        f"{'Average Waiting Time':<25}"
        f"{random_average['waiting']:>18.2f}"
        f"{q_learning_average['waiting']:>18.2f}"
    )

    print(
        f"{'Average Queue':<25}"
        f"{random_average['queue']:>18.2f}"
        f"{q_learning_average['queue']:>18.2f}"
    )

    print()

    # -----------------------------------------
    # Basic interpretation
    # -----------------------------------------

    if q_learning_average["reward"] > random_average["reward"]:
        print("Q-learning achieved a higher average reward.")

    else:
        print("Q-learning did not achieve a higher average reward.")

    if (
        q_learning_average["waiting"]
        < random_average["waiting"]
    ):
        print("Q-learning reduced average waiting time.")

    else:
        print(
            "Q-learning did not reduce average waiting time."
        )

    if (
        q_learning_average["queue"]
        < random_average["queue"]
    ):
        print("Q-learning reduced average queue.")

    else:
        print(
            "Q-learning did not reduce average queue."
        )

    print()
    print("Evaluation completed successfully.")


if __name__ == "__main__":
    main()