from environment.simple_traffic_env import SimpleTrafficEnv


def run_random_episode(env):
    """
    Run one episode using a random agent.

    The agent does not learn.
    It randomly chooses between:
        0 = Keep current phase
        1 = Switch phase
    """

    state, info = env.reset()

    total_reward = 0.0
    total_waiting_time = 0.0
    total_queue = 0.0

    terminated = False
    truncated = False

    steps = 0

    while not terminated and not truncated:

        # Randomly select an action.
        action = env.action_space.sample()

        # Take the action.
        state, reward, terminated, truncated, info = env.step(action)

        # Accumulate reward.
        total_reward += reward

        # Accumulate traffic metrics.
        total_waiting_time += (
            info["ns_waiting"] + info["ew_waiting"]
        )

        total_queue += (
            info["ns_queue"] + info["ew_queue"]
        )

        steps += 1

    return {
        "total_reward": total_reward,
        "total_waiting_time": total_waiting_time,
        "total_queue": total_queue,
        "steps": steps,
    }


def main():

    # -----------------------------------------
    # Configuration
    # -----------------------------------------

    num_episodes = 10

    # -----------------------------------------
    # Create environment
    # -----------------------------------------

    env = SimpleTrafficEnv()

    print("=" * 55)
    print("Random Agent Baseline")
    print("=" * 55)

    print()
    print(f"Running {num_episodes} episodes...")
    print()

    results = []

    # -----------------------------------------
    # Run episodes
    # -----------------------------------------

    for episode in range(1, num_episodes + 1):

        result = run_random_episode(env)

        results.append(result)

        print(
            f"Episode {episode:2d} | "
            f"Reward: {result['total_reward']:8.2f} | "
            f"Waiting: {result['total_waiting_time']:8.2f} | "
            f"Queue: {result['total_queue']:7.2f} | "
            f"Steps: {result['steps']}"
        )

    # -----------------------------------------
    # Calculate averages
    # -----------------------------------------

    average_reward = (
        sum(result["total_reward"] for result in results)
        / num_episodes
    )

    average_waiting = (
        sum(result["total_waiting_time"] for result in results)
        / num_episodes
    )

    average_queue = (
        sum(result["total_queue"] for result in results)
        / num_episodes
    )

    # -----------------------------------------
    # Display final results
    # -----------------------------------------

    print()
    print("=" * 55)
    print("Random Agent Results")
    print("=" * 55)

    print(f"Average Reward       : {average_reward:.2f}")
    print(f"Average Waiting Time : {average_waiting:.2f}")
    print(f"Average Queue        : {average_queue:.2f}")

    print()
    print("Random agent baseline completed successfully.")

    env.close()


if __name__ == "__main__":
    main()