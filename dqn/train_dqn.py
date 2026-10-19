from pathlib import Path
import sys
import numpy as np

# Running this file directly sets sys.path[0] to q_learning/, so add the
# project root to make the sibling environment/ package importable.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from environment.simple_traffic_env import SimpleTrafficEnv
from dqn.dqn_agent import DQNAgent
from q_learning.q_learning_agent import discretize_state


def train_dqn(num_episodes=500):

    # -----------------------------------------
    # Environment
    # -----------------------------------------

    env = SimpleTrafficEnv()

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    # -----------------------------------------
    # Agent
    # -----------------------------------------

    agent = DQNAgent(
        state_size=state_size,
        action_size=action_size,
    )

    rewards = []
    losses = []

    # Update target network every N episodes.
    target_update_frequency = 10

    print("=" * 65)
    print("DQN Traffic Signal Training")
    print("=" * 65)

    print()
    print(f"State size : {state_size}")
    print(f"Action size: {action_size}")
    print(f"Episodes   : {num_episodes}")
    print(f"Device     : {agent.device}")
    print()

    # -----------------------------------------
    # Training loop
    # -----------------------------------------

    for episode in range(1, num_episodes + 1):

        state, info = env.reset()

        total_reward = 0.0
        episode_losses = []

        terminated = False
        truncated = False

        while not terminated and not truncated:

            # ---------------------------------
            # Choose action
            # ---------------------------------

            action = agent.choose_action(state)

            # ---------------------------------
            # Environment interaction
            # ---------------------------------

            next_state, reward, terminated, truncated, info = (
                env.step(action)
            )

            done = terminated or truncated

            # ---------------------------------
            # Store experience
            # ---------------------------------

            agent.remember(
                state,
                action,
                reward,
                next_state,
                done,
            )

            # ---------------------------------
            # Train DQN
            # ---------------------------------

            loss = agent.train_step()

            if loss is not None:
                episode_losses.append(loss)

            state = next_state

            total_reward += reward

        # -------------------------------------
        # Decay exploration
        # -------------------------------------

        agent.decay_epsilon()

        # -------------------------------------
        # Update target network
        # -------------------------------------

        if episode % target_update_frequency == 0:

            agent.update_target_network()

        rewards.append(total_reward)

        if episode_losses:

            losses.append(
                np.mean(episode_losses)
            )

        # -------------------------------------
        # Display progress
        # -------------------------------------

        if episode % 50 == 0:

            average_reward = np.mean(
                rewards[-50:]
            )

            if losses:
                average_loss = np.mean(
                    losses[-50:]
                )
            else:
                average_loss = 0.0

            print(
                f"Episode {episode:4d} | "
                f"Avg Reward: {average_reward:10.2f} | "
                f"Loss: {average_loss:.4f} | "
                f"Epsilon: {agent.epsilon:.3f}"
            )

    env.close()

    # -----------------------------------------
    # Save model
    # -----------------------------------------

    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    model_path = models_dir / "dqn_model.pth"

    agent.save(model_path)

    print()
    print(f"DQN model saved to: {model_path}")

    return agent, rewards, losses


def main():

    agent, rewards, losses = train_dqn(
        num_episodes=500
    )

    print()
    print("=" * 65)
    print("DQN Training Completed")
    print("=" * 65)

    print(
        f"Replay buffer size: "
        f"{len(agent.replay_buffer)}"
    )

    print(
        f"Final epsilon: "
        f"{agent.epsilon:.3f}"
    )

    print(
        f"Final 50-episode average reward: "
        f"{np.mean(rewards[-50:]):.2f}"
    )

    if losses:

        print(
            f"Final 50-episode average loss: "
            f"{np.mean(losses[-50:]):.4f}"
        )

    print()
    print(
        "DQN training completed successfully."
    )


if __name__ == "__main__":
    main()