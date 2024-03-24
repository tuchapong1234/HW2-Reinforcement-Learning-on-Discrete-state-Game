# HW2-Reinforcement-Learning-on-Discrete-state-Game
## Exploring various learning algorithms
### Different between learning algorithms


| Metric          | Q-Learning               |  MC                |SARSA               | Double Q-Learning                |
|:---------------:|:--------------------:|:--------------------:|:--------------------:|:--------------------:|
|Additional Initial Parameters |  -     | history <br>[ $S_{t}$, $A_{t}$, $R_{t}$]      | -      | 2 Different $Q$ values      |
|Future Q in temporal difference error | $\max(Q(s_{t+1}, a_{t}))$ | $G_{t}$ (Return) | $Q(s_{t+1}, a_{t+1})$ | Update $Q_{a}$ with $Q_{b}(s_{t+1}, \text{argmax}(Q_{a}(s_{t+1}, a_{t})))$ |
| Implementation Step | 1. Initial <br> 2. get $At$ from $\pi$ <br> 3. Update $Q$ by temporal difference $\max(Q(s_{t+1}, a_{t}))$ | 1. Initial <br> 2. get $At$ from $\pi$ <br> 3. Update $Q$ by history, $G_{t}$   | 1. Initial <br> 2. get $At$ from $\pi$ <br> 3. Update $Q$ by temporal difference $Q(s_{t+1}, a_{t+1})$ | 1. Initial <br> 2. get $At$ from $\pi_{a}$, $\pi_{b}$  <br>  3. Randomly Update $Q_{a}$, $Q_{b}$ <br> 4. Update $Q$ from 3. by temporal difference  $Q_{b}(s_{t+1}, \text{argmax}(Q_{a}(s_{t+1}, a_{t})))$ |
| Saved Parameter | $Q$     | $Q$, $N_{t}$      | $Q$      | $Q = avg(Q_{a} + Q_{b})$      |


### Python Code 

1. Q-Learning

1.1. Initial Parameters
       
```bash
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
```

1.2. get $At$ from $\pi$
    
```bash
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
```
   
1.3. Update $Q$ by temporal difference $\max(Q(s_{t+1}, a_{t}))$
    
```bash
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
```

2. MC

2.1. Initial Parameters

   Same as Q-Learning

   Additional Step
   
```bash
if self.control_type == ControlType.MONTE_CARLO:
            self.obs_hist = []
            self.action_hist = []
            self.reward_hist = []
        elif self.control_type == ControlType.DOUBLE_Q_LEARNING:
            self.qa_values = defaultdict(lambda: np.zeros(self.num_of_action))
            self.qb_values = defaultdict(lambda: np.zeros(self.num_of_action))
```

2.2. get $At$ from $\pi$
  
   Same as Q-Learning
   
2.3. Update $Q$ by history, $G_{t}$
    
```bash
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
```

3. SARSA

3.1. Initial Parameters
  
   Same as Q-Learning
   
3.2. get $At$ from $\pi$
  
   Same as Q-Learning
   
3.3. Update $Q$ by temporal difference $Q(s_{t+1}, a_{t+1})$
  
   Same as Q-Learning

   Change Future Q to $Q(s_{t+1}, a_{t+1})$
   
```bash
future_q_value = (not terminated) * self.q_values[next_obs][next_action]
```

4. Double Q-Learning
    
4.1. Initial Parameters

   Same as Q-Learning

   Additional Step

```bash
elif self.control_type == ControlType.DOUBLE_Q_LEARNING:
      self.qa_values = defaultdict(lambda: np.zeros(self.num_of_action))
      self.qb_values = defaultdict(lambda: np.zeros(self.num_of_action))
```

   
4.2. get $At$ from $\pi_{a}$, $\pi_{b}$

    same as Q-Learning but change to Q that has been average
   
4.3. Randomly Update $Q_{a}$, $Q_{b}$ & 4. Update $Q$ from 3. by temporal difference  $Q_{b}(s_{t+1}, \text{argmax}(Q_{a}(s_{t+1}, a_{t})))$
    
```bash
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
```  

## Results and analysis

### Parameters Studies

To achieve optimal performance and stability in training RL models suitable for different algorithmsใ

| Metric          | Q-Learning               |  MC                |SARSA               | Double Q-Learning                |
|:--------------:|:--------------:|:--------------:|:--------------:|:--------------:|
| Discount Factor | ![image](https://github.com/tuchapong1234/HW2-Reinforcement-Learning-on-Discrete-state-Game/assets/113016544/4baf40e8-84aa-40d1-8b43-ba73a399a951)| ![image](https://github.com/tuchapong1234/HW2-Reinforcement-Learning-on-Discrete-state-Game/assets/113016544/f94e1455-827d-4e92-b94f-6b2e925f317f)| ![image](https://github.com/tuchapong1234/HW2-Reinforcement-Learning-on-Discrete-state-Game/assets/113016544/3408657c-6f41-495e-9006-dd38ff735db9)| ![image](https://github.com/tuchapong1234/HW2-Reinforcement-Learning-on-Discrete-state-Game/assets/113016544/be7c1c3e-e8ec-440d-92bd-61ebafa187b9)|
| Epsilon | ![image](https://github.com/tuchapong1234/HW2-Reinforcement-Learning-on-Discrete-state-Game/assets/113016544/ed55142b-ce32-4beb-bf6d-26cdb8a0bf90)| ![image](https://github.com/tuchapong1234/HW2-Reinforcement-Learning-on-Discrete-state-Game/assets/113016544/5567c98b-9182-4ed3-8298-62bfbce5bdd5)| ![image](https://github.com/tuchapong1234/HW2-Reinforcement-Learning-on-Discrete-state-Game/assets/113016544/26cf364d-eb53-4d2a-a496-abe5a1fa275c)| ![image](https://github.com/tuchapong1234/HW2-Reinforcement-Learning-on-Discrete-state-Game/assets/113016544/8155ebf7-f8d3-40de-b7cc-378ff5f4f4a5)|
| Learning Rate | ![image](https://github.com/tuchapong1234/HW2-Reinforcement-Learning-on-Discrete-state-Game/assets/113016544/126f50d0-5469-475b-af41-d9ea0b561f34)| - | ![image](https://github.com/tuchapong1234/HW2-Reinforcement-Learning-on-Discrete-state-Game/assets/113016544/50628616-0c75-40f1-9051-d1599148511e)| ![image](https://github.com/tuchapong1234/HW2-Reinforcement-Learning-on-Discrete-state-Game/assets/113016544/9bcd2800-bfae-42ca-9b47-ff5ac1a3e6c8)|

### Value & Policy After Training 1,000,000 iterations.

| Metric          | Q-Learning               |  MC                |SARSA               | Double Q-Learning                |
|:--------------:|:--------------:|:--------------:|:--------------:|:--------------:|
| With Usable Ace | ![image](https://github.com/tuchapong1234/HW2-Reinforcement-Learning-on-Discrete-state-Game/assets/113016544/aa6d2a53-2a5f-4e87-b363-edb57bd21ff8)| ![image](https://github.com/tuchapong1234/HW2-Reinforcement-Learning-on-Discrete-state-Game/assets/113016544/aa8cb257-f961-429f-84a3-30eebc71f0de)| ![image](https://github.com/tuchapong1234/HW2-Reinforcement-Learning-on-Discrete-state-Game/assets/113016544/9c84adad-2bb7-48c9-921c-15bbd6e0be48)| ![image](https://github.com/tuchapong1234/HW2-Reinforcement-Learning-on-Discrete-state-Game/assets/113016544/fd24838b-51c2-42d3-b2a6-546830f44ae7)|
| Without Usable Ace | ![image](https://github.com/tuchapong1234/HW2-Reinforcement-Learning-on-Discrete-state-Game/assets/113016544/b614fa0a-c9a0-4f9e-ab4d-a6411807efc2)| ![image](https://github.com/tuchapong1234/HW2-Reinforcement-Learning-on-Discrete-state-Game/assets/113016544/a736ac89-b621-4de1-9540-0307709c94f1)| ![image](https://github.com/tuchapong1234/HW2-Reinforcement-Learning-on-Discrete-state-Game/assets/113016544/5e7b1897-355b-4322-bce0-2a03bcdef248)| ![image](https://github.com/tuchapong1234/HW2-Reinforcement-Learning-on-Discrete-state-Game/assets/113016544/940cbe9d-5b70-4b31-9992-7b58669729bd)|

### Compare Results of 4 algorithms (Q-Learning, MC, TD, Double Q-Learning) 

In training to find the optimal policy, we conducted training using iterations, specifically 1,000,000 iterations. Subsequently, we plotted the reward values obtained on the training graph. On the left-hand side, we have the Multi Cumulative Return graph, displaying the cumulative return values of each algorithm. On the right-hand side, we have the Multi Episode Return filtered graph, showing the episode return values that have been filtered using the moving average for each algorithm.

#### Compare Training Result

![image](https://github.com/tuchapong1234/HW2-Reinforcement-Learning-on-Discrete-state-Game/assets/113016544/e3540002-4b34-4a5c-92f8-ab9b5426075d)

| Metric          | Q-Learning               |  MC                |SARSA               | Double Q-Learning                |
|:--------------:|:--------------:|:--------------:|:--------------:|:--------------:|
| Expected Return | -0.135179 | -0.134615 | -0.133471 | -0.135757 |

From the training graph, we can observe that the trend of each algorithm is similar, mainly due to using the same epsilon value, resulting in equal learning rates for all algorithms. Additionally, upon examining the Multi Episode Return filtered graph, it becomes evident that the Episode Return values for all algorithms can converge similarly.

When arranging the algorithms based on their expected reward values during training, the order would be as follows:

1. SARSA
2. Monte Carlo (MC)
3. Q-Learning
4. Double Q-Learning

#### Compare Testing Result

![image](https://github.com/tuchapong1234/HW2-Reinforcement-Learning-on-Discrete-state-Game/assets/113016544/067eb0dc-86ea-4e7e-9b83-404fcdbd9874)

| Metric          | Q-Learning               |  MC                |SARSA               | Double Q-Learning                |
|:--------------:|:--------------:|:--------------:|:--------------:|:--------------:|
| Expected Return | -0.048 | -0.04802 | -0.04315 | -0.04693 |

From the testing graph, if we arrange the algorithms based on their expected reward values during training, the order would be as follows:

1. SARSA
2. Double Q-Learning
3. Q-Learning
4. Monte Carlo (MC)

### Conclusion 

#### Monte Carlo (MC)

Monte Carlo (MC) ranks last because its Q-Value update directly employs the return values. Consequently, this leads to higher variance due to the environmental randomness. Therefore, when comparing iterations of training, MC yields the poorest results. Additionally, MC carries a risk of overfitting to the environment. This can be observed from the fact that during training, MC yields results in the second position, but during testing, its results fall to the last position.

#### SARSA

The main reason SARSA provides the best results is due to its method of updating Q-Values, which involves using estimated values instead of directly utilizing return values. Consequently, SARSA is less impacted by the randomness of the environment, resulting in lower variance. However, this comes at the cost of higher bias due to the inherent inaccuracies in estimating values. Nevertheless, when compared to Q-Learning and Double Q-Learning, SARSA exhibits lower bias, hence yielding the best overall performance.

#### Q-Learning

The reason Q-Learning ranks third is because of its method of updating Q-Values, which involves using the maximum Q-Value of the next state. This can introduce a positive bias because we cannot guarantee that the action with the maximum Q-Value will always be the best action.

#### Double Q-Learning

The reason Double Q-Learning outperforms Q-Learning is because it maintains two sets of Q-Values, Qa and Qb. During updates, it selects the best action from one set of Q-Values and uses the Q-Value of that action from the other set. This reduces the positive bias present in regular Q-Learning because if the chosen action is truly the best action, it will result in a high Q-Value. However, if it is not the best action, its Q-Value will be lower.
