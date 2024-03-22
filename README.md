# HW2-Reinforcement-Learning-on-Discrete-state-Game
## Exploring various learning algorithms
### Different between learning algorithms


| Metric          | Q-Learning               |  MC                |SARSA               | Double Q-Learning                |
|:---------------:|:--------------------:|:--------------------:|:--------------------:|:--------------------:|
|`Additional` Initial Parameters |  -     | history <br>[ $S_{t}$, $A_{t}$, $R_{t}$]      | -      | 2 Different $Q$ values      |
|Future Q in temporal difference error | $\max(Q(s_{t+1}, a_{t}))$ | $G_{t}$ (Return) | $Q(s_{t+1}, a_{t+1})$ | Update $Q_{a}$ with $Q_{b}(s_{t+1}, \text{argmax}(Q_{a}(s_{t+1}, a_{t})))$ |
| Implementation Step | 1. Initial <br> 2. get $At$ from $\pi$ <br> 3. Update $Q$ by temporal difference $\max(Q(s_{t+1}, a_{t}))$ | 1. Initial <br> 2. get $At$ from $\pi$ <br> 3. Update $Q$ by history, $G_{t}$   | 1. Initial <br> 2. get $At$ from $\pi$ <br> 3. Update $Q$ by temporal difference $Q(s_{t+1}, a_{t+1})$ | 1. Initial <br> 2. get $At$ from $\pi_{a}$, $\pi_{b}$ <br> 3. Randomly Update $Q_{a}$, $Q_{b}$ <br> 4. Update $Q$ from 3. by temporal difference  $Q_{b}(s_{t+1}, \text{argmax}(Q_{a}(s_{t+1}, a_{t})))$ |
| Additional info | Row 4, Column 2     | Row 4, Column 3      | Row 4, Column 4      | Row 4, Column 5      |
| More info       | Row 5, Column 2     | Row 5, Column 3      | Row 5, Column 4      | Row 5, Column 5      |

### Python Code 

1. Q-Learning

    1. Initial Parameters
    2. get $At$ from $\pi$
    3. Update $Q$ by temporal difference $\max(Q(s_{t+1}, a_{t}))$

2. MC

    1. Initial Parameters
    2. get $At$ from $\pi$ 
    3. Update $Q$ by history, $G_{t}$

3. SARSA

    1. Initial Parameters 
    2. get $At$ from $\pi$ 
    3. Update $Q$ by temporal difference $Q(s_{t+1}, a_{t+1})$

4. Double Q-Learning
    1. Initial Parameters 
    2. get $At$ from $\pi_{a}$, $\pi_{b}$ 
    3. Randomly Update $Q_{a}$, $Q_{b}$ 
    4. Update $Q$ from 3. by temporal difference  $Q_{b}(s_{t+1}, \text{argmax}(Q_{a}(s_{t+1}, a_{t})))$
