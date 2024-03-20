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
            control_type : ControlType,
            learning_rate: float,
            initial_epsilon: float,
            epsilon_decay: float,
            final_epsilon: float,
            discount_factor: float = 0.95,
    ):
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
            self.q_values[obs][i] = self.qa_values[obs][i] + self.qb_values[obs][i]


    def __Monte_Carlo_Update__(
        self,
        obs: tuple[int, int, bool],
        action: int,
        reward: float,
        terminated: bool
    ):
        self.obs_hist.append(obs)
        self.action_hist.append(action)
        self.reward_hist.append(reward)
        if terminated:
            Gt = 0.0
            for i in reversed(range(len(self.reward_hist))):
                Gt = (self.discount_factor * Gt) + float(self.reward_hist[i])
                self.n_values[self.obs_hist[i]][self.action_hist[i]] += 1.0
                lr = 1.0 / self.n_values[self.obs_hist[i]][self.action_hist[i]]
                temporal_difference = (
                    Gt - self.q_values[self.obs_hist[i]][self.action_hist[i]]
                )
                self.q_values[self.obs_hist[i]][self.action_hist[i]] += (lr * temporal_difference)
                self.training_error.insert(0, temporal_difference)
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
        obs: tuple[int, int, bool] = tuple[0, 0, False],
        action: int = 0,
        reward: float = 0.0,
        next_obs: tuple[int, int, bool] = tuple[0, 0, False],
        next_action: int = 0,
        terminated: bool = False,
    ):
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
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)

    def save_model(self, path, filename):
        # Convert tuple keys to strings
        try:
            q_values_str_keys = {str(k): v.tolist() for k, v in self.q_values.items()}
        except:
            q_values_str_keys = {str(k): v for k, v in self.q_values.items()}
        
        # Save model parameters to a JSON file
        model_params = {
            'q_values': q_values_str_keys,
            # 'biases': self.biases.tolist()
        }
        # path = os.path.abspath(path)
        full_path = os.path.join(path, filename)
        with open(full_path, 'w') as f:
            json.dump(model_params, f)

            
    def load_model(self, path, filename):
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
                return self.q_values
        except:
            pass