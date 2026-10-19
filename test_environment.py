from environment.simple_traffic_env import SimpleTrafficEnv


def main():
    env = SimpleTrafficEnv()

    print("Creating environment...")
    print()

    # -----------------------------------------
    # Test reset()
    # -----------------------------------------

    state, info = env.reset(seed=42)

    print("Initial state:")
    print(state)

    print()
    print("Initial info:")
    print(info)

    # -----------------------------------------
    # Test action space
    # -----------------------------------------

    print()
    print("Action space:")
    print(env.action_space)

    # -----------------------------------------
    # Test observation space
    # -----------------------------------------

    print()
    print("Observation space:")
    print(env.observation_space)

    # -----------------------------------------
    # Test step()
    # -----------------------------------------

    print()
    print("Testing actions...")

    for action in [0, 1, 0, 1, 0]:

        state, reward, terminated, truncated, info = env.step(action)

        print()
        print(f"Action: {action}")
        print(f"State: {state}")
        print(f"Reward: {reward:.2f}")
        print(f"Terminated: {terminated}")
        print(f"Truncated: {truncated}")
        print(f"Info: {info}")

    # -----------------------------------------
    # Test render()
    # -----------------------------------------

    print()
    print("Testing render():")

    env.render()

    env.close()

    print()
    print("Environment test completed successfully.")


if __name__ == "__main__":
    main()