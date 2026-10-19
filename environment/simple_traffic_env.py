import random

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class SimpleTrafficEnv(gym.Env):
    """
    A simple two-direction traffic intersection environment.

    Traffic is divided into:
    - North/South
    - East/West

    Actions:
    - 0: Keep the current traffic-light phase
    - 1: Switch the traffic-light phase

    State:
    - North/South queue length
    - East/West queue length
    - North/South waiting time
    - East/West waiting time
    - Current traffic-light phase
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode=None):
        super().__init__()

        self.render_mode = render_mode

        # -------------------------------------------------
        # Action Space
        # -------------------------------------------------
        # 0 = Keep current phase
        # 1 = Switch phase
        self.action_space = spaces.Discrete(2)

        # -------------------------------------------------
        # Observation Space
        # -------------------------------------------------
        # [NS queue, EW queue, NS waiting, EW waiting, phase]
        #
        # Queues and waiting times have reasonable upper
        # limits for this small simulation.
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0], dtype=np.float32),
            high=np.array([100, 100, 1000, 1000, 1], dtype=np.float32),
            dtype=np.float32,
        )

        # Maximum number of simulation steps per episode.
        self.max_steps = 100

        # Internal simulation variables.
        self.ns_queue = 0
        self.ew_queue = 0

        self.ns_waiting = 0.0
        self.ew_waiting = 0.0

        # 0 = North/South green
        # 1 = East/West green
        self.current_phase = 0

        self.step_count = 0

        # Use a local random generator for reproducibility.
        self.rng = random.Random()

    def _get_state(self):
        """
        Return the current environment state.
        """

        return np.array(
            [
                self.ns_queue,
                self.ew_queue,
                self.ns_waiting,
                self.ew_waiting,
                self.current_phase,
            ],
            dtype=np.float32,
        )

    def _generate_arrivals(self):
        """
        Generate new vehicles arriving at the intersection.

        The arrival process is intentionally simple.
        """

        ns_arrivals = self.rng.randint(0, 3)
        ew_arrivals = self.rng.randint(0, 3)

        self.ns_queue += ns_arrivals
        self.ew_queue += ew_arrivals

    def _process_departures(self):
        """
        Vehicles can leave the intersection only from
        the currently green direction.
        """

        vehicles_served = 0

        # North/South has green light.
        if self.current_phase == 0:

            # Serve up to 3 vehicles.
            served = min(self.ns_queue, 3)

            self.ns_queue -= served
            vehicles_served += served

        # East/West has green light.
        else:

            # Serve up to 3 vehicles.
            served = min(self.ew_queue, 3)

            self.ew_queue -= served
            vehicles_served += served

        return vehicles_served

    def _update_waiting_time(self):
        """
        Every vehicle that remains in a queue waits for
        one additional simulation step.
        """

        self.ns_waiting += self.ns_queue
        self.ew_waiting += self.ew_queue

    def _calculate_reward(self):
        """
        Reward is based primarily on queue length
        and accumulated waiting time.

        Lower traffic cost = better reward.
        """

        total_queue = self.ns_queue + self.ew_queue

        total_waiting = self.ns_waiting + self.ew_waiting

        reward = -(total_queue + total_waiting)

        return float(reward)

    def reset(self, *, seed=None, options=None):
        """
        Start a new episode.
        """

        super().reset(seed=seed)

        if seed is not None:
            self.rng.seed(seed)

        # Initial traffic.
        self.ns_queue = self.rng.randint(2, 6)
        self.ew_queue = self.rng.randint(2, 6)

        self.ns_waiting = 0.0
        self.ew_waiting = 0.0

        # Start with North/South green.
        self.current_phase = 0

        self.step_count = 0

        state = self._get_state()

        info = {
            "vehicles_waiting": self.ns_queue + self.ew_queue
        }

        return state, info

    def step(self, action):
        """
        Advance the simulation by one time step.
        """

        # Make sure the action is valid.
        action = int(action)

        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")

        # -------------------------------------------------
        # 1. Apply traffic-light action
        # -------------------------------------------------

        if action == 1:
            self.current_phase = 1 - self.current_phase

        # -------------------------------------------------
        # 2. Generate new vehicle arrivals
        # -------------------------------------------------

        self._generate_arrivals()

        # -------------------------------------------------
        # 3. Process vehicles that can pass
        # -------------------------------------------------

        vehicles_served = self._process_departures()

        # -------------------------------------------------
        # 4. Update waiting time
        # -------------------------------------------------

        self._update_waiting_time()

        # -------------------------------------------------
        # 5. Calculate reward
        # -------------------------------------------------

        reward = self._calculate_reward()

        # -------------------------------------------------
        # 6. Advance simulation clock
        # -------------------------------------------------

        self.step_count += 1

        terminated = False

        truncated = self.step_count >= self.max_steps

        # -------------------------------------------------
        # 7. Get new state
        # -------------------------------------------------

        state = self._get_state()

        # -------------------------------------------------
        # 8. Additional information
        # -------------------------------------------------

        info = {
            "vehicles_served": vehicles_served,
            "ns_queue": self.ns_queue,
            "ew_queue": self.ew_queue,
            "ns_waiting": self.ns_waiting,
            "ew_waiting": self.ew_waiting,
            "phase": self.current_phase,
        }

        return state, reward, terminated, truncated, info

    def render(self):
        """
        Display the current traffic situation.
        """

        phase_name = (
            "North/South GREEN"
            if self.current_phase == 0
            else "East/West GREEN"
        )

        print("\n--- Traffic Intersection ---")
        print(f"Step: {self.step_count}")
        print(f"Signal: {phase_name}")
        print(f"NS Queue: {self.ns_queue}")
        print(f"EW Queue: {self.ew_queue}")
        print(f"NS Waiting: {self.ns_waiting:.1f}")
        print(f"EW Waiting: {self.ew_waiting:.1f}")