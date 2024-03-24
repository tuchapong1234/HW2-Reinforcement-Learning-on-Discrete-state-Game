from __future__ import annotations

from collections import defaultdict

import numpy as np
from enum import Enum
import os
import json


np.random.seed(10)

class ControlType(Enum):
    MONTE_CARLO = 1
    TEMPORAL_DIFFERENCE = 2
    Q_LEARNING = 3
    DOUBLE_Q_LEARNING = 4

class BlackJackAgent():
    def __init__(
            self,
            control_type: ControlType,
            learning_rate: float,
            initial_epsilon: float,
            epsilon_decay: float,
            final_epsilon: float,
            discount_factor: float = 0.95,
    ):
        """
        Initialize the Blackjack Agent.

        Args:
            control_type (ControlType): The control type for the agent.
            learning_rate (float): The learning rate for updating Q-values.
            initial_epsilon (float): The initial exploration rate.
            epsilon_decay (float): The rate at which epsilon decays over time.
            final_epsilon (float): The final exploration rate.
            discount_factor (float, optional): The discount factor for future rewards. Defaults to 0.95.
        """
        self.control_type = control_type
        self.lr = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        self.num_of_action = 2
        self.q_values = defaultdict(lambda: np.zeros(self.num_of_action))
        self.n_values = defaultdict(lambda: np.zeros(self.num_of_action))
        self.training_error = []

        if self.control_type == ControlType.MONTE_CARLO:
            self.obs_hist = []
            self.action_hist = []
            self.reward_hist = []
        elif self.control_type == ControlType.DOUBLE_Q_LEARNING:
            self.qa_values = defaultdict(lambda: np.zeros(self.num_of_action))
            self.qb_values = defaultdict(lambda: np.zeros(self.num_of_action))

    def get_action(self, obs: tuple[int, int, bool]) -> int:
        """
        Get action based on epsilon-greedy policy.

        Args:
            obs (tuple[int, int, bool]): The observation state.

        Returns:
            int: The chosen action.
        """
        if np.random.random() < self.epsilon:
            return np.random.choice(self.num_of_action)
        else:
            return int(np.argmax(self.q_values[obs]))

    def __Q_Learning_Update__(
        self,
        obs: tuple[int, int, bool],
        action: int,
        reward: float,
        next_obs: tuple[int, int, bool],
        terminated: bool
    ):
        """
        Update Q-values using Q-Learning.

        Args:
            obs (tuple[int, int, bool]): Current observation state.
            action (int): Action taken.
            reward (float): Reward received.
            next_obs (tuple[int, int, bool]): Next observation state.
            terminated (bool): Whether the episode terminated.
        """
        future_q_value = (not terminated) * np.max(self.q_values[next_obs])
        temporal_difference = (
            reward + self.discount_factor * future_q_value - self.q_values[obs][action]
        )

        self.q_values[obs][action] = (
            self.q_values[obs][action] + self.lr * temporal_difference
        )
        self.training_error.append(temporal_difference)

    def __Double_Q_Learning_Update__(
        self,
        obs: tuple[int, int, bool],
        action: int,
        reward: float,
        next_obs: tuple[int, int, bool],
        terminated: bool
    ):
        """
        Update Q-values using Double Q-Learning.

        Args:
            obs (tuple[int, int, bool]): Current observation state.
            action (int): Action taken.
            reward (float): Reward received.
            next_obs (tuple[int, int, bool]): Next observation state.
            terminated (bool): Whether the episode terminated.
        """
        if np.random.uniform(0, 1) < 0.5:
            max_next_action = np.argmax(self.qb_values[next_obs])
            future_q_value = (not terminated) * self.qa_values[next_obs][max_next_action]
            temporal_difference = (
                reward + self.discount_factor * future_q_value - self.qb_values[obs][action]
            )
            self.qb_values[obs][action] = (
                self.qb_values[obs][action] + self.lr * temporal_difference
            )
            self.training_error.append(temporal_difference)
        else:
            max_next_action = np.argmax(self.qa_values[next_obs])
            future_q_value = (not terminated) * self.qb_values[next_obs][max_next_action]
            temporal_difference = (
                reward + self.discount_factor * future_q_value - self.qa_values[obs][action]
            )
            self.qa_values[obs][action] = (
                self.qa_values[obs][action] + self.lr * temporal_difference
            )
            self.training_error.append(temporal_difference)
        for i in range(self.num_of_action):
            self.q_values[obs][i] = (self.qa_values[obs][i] + self.qb_values[obs][i]) / 2

    def __Monte_Carlo_Update__(
        self,
        obs: tuple[int, int, bool],
        action: int,
        reward: float,
        terminated: bool
    ):
        """
        Update Q-values using Monte Carlo method.

        Args:
            obs (tuple[int, int, bool]): Current observation state.
            action (int): Action taken.
            reward (float): Reward received.
            terminated (bool): Whether the episode terminated.
        """
        self.obs_hist.append(obs)
        self.action_hist.append(action)
        self.reward_hist.append(reward)
        if terminated:
            Gt = 0.0
            error_list = []
            for i in reversed(range(len(self.reward_hist))):
                Gt = (self.discount_factor * Gt) + float(self.reward_hist[i])
                self.n_values[self.obs_hist[i]][self.action_hist[i]] += 1.0
                lr = 1.0 / self.n_values[self.obs_hist[i]][self.action_hist[i]]
                temporal_difference = (
                    Gt - self.q_values[self.obs_hist[i]][self.action_hist[i]]
                )
                self.q_values[self.obs_hist[i]][self.action_hist[i]] += (lr * temporal_difference)
                error_list.insert(0, temporal_difference)
            self.training_error.extend(error_list.copy())
            self.obs_hist.clear()
            self.action_hist.clear()
            self.reward_hist.clear()

    def __Temporal_Difference_Update__(
        self,
        obs: tuple[int, int, bool],
        action: int,
        reward: float,
        next_obs: tuple[int, int, bool],
        next_action: int,
        terminated: bool,
    ):
        """
        Update Q-values using Temporal Difference learning.

        Args:
            obs (tuple[int, int, bool]): Current observation state.
            action (int): Action taken.
            reward (float): Reward received.
            next_obs (tuple[int, int, bool]): Next observation state.
            next_action (int): Action to be taken in the next state.
            terminated (bool): Whether the episode terminated.
        """
        future_q_value = (not terminated) * self.q_values[next_obs][next_action]
        temporal_difference = (
            reward + self.discount_factor * future_q_value - self.q_values[obs][action]
        )

        self.q_values[obs][action] = (
            self.q_values[obs][action] + self.lr * temporal_difference
        )
        self.training_error.append(temporal_difference)
        
    def update(
        self,
        obs: tuple[int, int, bool] = (0, 0, False),
        action: int = 0,
        reward: float = 0.0,
        next_obs: tuple[int, int, bool] = (0, 0, False),
        next_action: int = 0,
        terminated: bool = False,
    ):
        """
        Update the agent's Q-values based on the control type.

        Args:
            obs (tuple[int, int, bool], optional): Current observation state. Defaults to (0, 0, False).
            action (int, optional): Action taken. Defaults to 0.
            reward (float, optional): Reward received. Defaults to 0.0.
            next_obs (tuple[int, int, bool], optional): Next observation state. Defaults to (0, 0, False).
            next_action (int, optional): Action to be taken in the next state. Defaults to 0.
            terminated (bool, optional): Whether the episode terminated. Defaults to False.
        """
        if self.control_type == ControlType.Q_LEARNING:
            self.__Q_Learning_Update__(
                obs=obs,
                action=action,
                reward=reward,
                next_obs=next_obs,
                terminated=terminated
            )
        elif self.control_type == ControlType.MONTE_CARLO:
            self.__Monte_Carlo_Update__(
                obs=obs,
                action=action,
                reward=reward,
                terminated=terminated
            )
        elif self.control_type == ControlType.TEMPORAL_DIFFERENCE:
            self.__Temporal_Difference_Update__(
                obs=obs,
                action=action,
                reward=reward,
                next_obs=next_obs,
                next_action=next_action,
                terminated=terminated
            )
        elif self.control_type == ControlType.DOUBLE_Q_LEARNING:
            self.__Double_Q_Learning_Update__(
                obs=obs,
                action=action,
                reward=reward,
                next_obs=next_obs,
                terminated=terminated
            )

    def decay_epsilon(self):
        """
        Decay epsilon value.
        """
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)

    def save_model(self, path, filename):
        """
        Save the model parameters to a JSON file.

        Args:
            path (str): Path to save the model.
            filename (str): Name of the file.
        """
        # Convert tuple keys to strings
        try:
            q_values_str_keys = {str(k): v.tolist() for k, v in self.q_values.items()}
        except:
            q_values_str_keys = {str(k): v for k, v in self.q_values.items()}
        if self.control_type == ControlType.MONTE_CARLO:
            try:
                n_values_str_keys = {str(k): v.tolist() for k, v in self.n_values.items()}
            except:
                n_values_str_keys = {str(k): v for k, v in self.n_values.items()}
        
        # Save model parameters to a JSON file
        if self.control_type == ControlType.MONTE_CARLO:
            model_params = {
                'q_values': q_values_str_keys,
                'n_values': n_values_str_keys
            }
        else:
            model_params = {
                'q_values': q_values_str_keys,
            }
        full_path = os.path.join(path, filename)
        with open(full_path, 'w') as f:
            json.dump(model_params, f)

            
    def load_model(self, path, filename):
        """
        Load model parameters from a JSON file.

        Args:
            path (str): Path where the model is stored.
            filename (str): Name of the file.

        Returns:
            dict: The loaded Q-values.
        """
        full_path = os.path.join(path, filename)
        try:         
            with open(full_path, 'r') as file:
                data = json.load(file)
                data_q_values = data['q_values']
                for state, action_values in data_q_values.items():
                    state = state.replace('(', '')
                    state = state.replace(')', '')
                    tuple_state = tuple(map(int, state.split(', ')))
                    self.q_values[tuple_state] = action_values.copy()
                    if self.control_type == ControlType.DOUBLE_Q_LEARNING:
                        self.qa_values[tuple_state] = action_values.copy()
                        self.qb_values[tuple_state] = action_values.copy()
                if self.control_type == ControlType.MONTE_CARLO:
                    data_n_values = data['n_values']
                    for state, n_values in data_n_values.items():
                        state = state.replace('(', '')
                        state = state.replace(')', '')
                        tuple_state = tuple(map(int, state.split(', ')))
                        self.n_values[tuple_state] = n_values.copy()
                return self.q_values
        except:
            pass