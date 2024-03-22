# HW2-Reinforcement-Learning-on-Discrete-state-Game
## Exploring various learning algorithms
### Different between learning algorithms


| Metric          | Q-Learning               |  MC                |SARSA               | Double Q-Learning                |
|:---------------:|:--------------------:|:--------------------:|:--------------------:|:--------------------:|
|Additional Initial Parameters |  -     | history <br>[ $S_{t}$, $A_{t}$, $R_{t}$]      | -      | 2 Different $Q$ values      |
|Future Q in temporal difference error | $\max(Q(s_{t+1}, a_{t}))$ | $G_{t}$ (Return) | $Q(s_{t+1}, a_{t+1})$ | Update $Q_{a}$ with $Q_{b}(s_{t+1}, \text{argmax}(Q_{a}(s_{t+1}, a_{t})))$ |
| Implementation Step | 1. Initial <br> 2. get $At$ from $\pi$ <br> 3. Update $Q$ by temporal difference $\max(Q(s_{t+1}, a_{t}))$ | 1. Initial <br> 2. get $At$ from $\pi$ <br> 3. Update $Q$ by history, $G_{t}$   | 1. Initial <br> 2. get $At$ from $\pi$ <br> 3. Update $Q$ by temporal difference $Q(s_{t+1}, a_{t+1})$ | 1. Initial <br> 2. get $At$ from $\pi_{a}$, $\pi_{b}$ <br> 3. Randomly Update $Q_{a}$, $Q_{b}$ <br> 4. Update $Q$ from 3. by temporal difference  $Q_{b}(s_{t+1}, \text{argmax}(Q_{a}(s_{t+1}, a_{t})))$ |
| Additional info | Row 4, Column 2     | Row 4, Column 3      | Row 4, Column 4      | Row 4, Column 5      |
| More info       | Row 5, Column 2     | Row 5, Column 3      | Row 5, Column 4      | Row 5, Column 5      |

### Python Code 

1. Q-Learning

    1. Initial Parameters

   <img src="https://github.com/tuchapong1234/HW2-Reinforcement-Learning-on-Discrete-state-Game/assets/113016544/43679bd3-739d-4650-917c-5fa5d496daa0" alt="Image" width="500" height="400">
   
    2. get $At$ from $\pi$
   
   <img src="https://github.com/tuchapong1234/HW2-Reinforcement-Learning-on-Discrete-state-Game/assets/113016544/4dd6c032-bdde-4452-9004-4977a3e2d597" alt="Image" width="500" height="100">
   
    3. Update $Q$ by temporal difference $\max(Q(s_{t+1}, a_{t}))$
  
    <img src="https://github.com/tuchapong1234/HW2-Reinforcement-Learning-on-Discrete-state-Game/assets/113016544/dc420a83-250d-49d3-9418-1d425700107c" alt="Image" width="500" height="150">


2. MC

    1. Initial Parameters

   Same as Q-Learning

   Additional Step

   <img src="https://github.com/tuchapong1234/HW2-Reinforcement-Learning-on-Discrete-state-Game/assets/113016544/314ff5d2-ee9d-4e1d-86ec-5090a8399458" alt="Image" width="500" height="125">

    2. get $At$ from $\pi$
  
   Same as Q-Learning
   
    3. Update $Q$ by history, $G_{t}$

    <img src="https://github.com/tuchapong1234/HW2-Reinforcement-Learning-on-Discrete-state-Game/assets/113016544/1ab06243-991c-4723-8dc5-7ef33f885951" alt="Image" width="500" height="250">
   

4. SARSA

    1. Initial Parameters
  
   Same as Q-Learning
   
    2. get $At$ from $\pi$
  
   Same as Q-Learning
   
    3. Update $Q$ by temporal difference $Q(s_{t+1}, a_{t+1})$
  
   Same as Q-Learning

   Change Future Q to $Q(s_{t+1}, a_{t+1})$

    <img src="https://github.com/tuchapong1234/HW2-Reinforcement-Learning-on-Discrete-state-Game/assets/113016544/840e51a2-a3ee-4397-b0a4-f55c1a073948" alt="Image" width="500" height="40">

5. Double Q-Learning
    1. Initial Parameters

   Same as Q-Learning

   Additional Step
  
   <img src="https://github.com/tuchapong1234/HW2-Reinforcement-Learning-on-Discrete-state-Game/assets/113016544/7258bde0-5fb0-4b8d-a161-fb9e6017fed9" alt="Image" width="500" height="125">
   
    2. get $At$ from $\pi_{a}$, $\pi_{b}$
   
   <img src="https://github.com/tuchapong1234/HW2-Reinforcement-Learning-on-Discrete-state-Game/assets/113016544/43679bd3-739d-4650-917c-5fa5d496daa0" alt="Image" width="500" height="400">
   
    3. Randomly Update $Q_{a}$, $Q_{b}$ & 4. Update $Q$ from 3. by temporal difference  $Q_{b}(s_{t+1}, \text{argmax}(Q_{a}(s_{t+1}, a_{t})))$
  
   <img src="https://github.com/tuchapong1234/HW2-Reinforcement-Learning-on-Discrete-state-Game/assets/113016544/ddda9b79-f3f6-4564-98dc-fa1a655307a3" alt="Image" width="500" height="400">

### For results and analysis, please refer to the 'blackjack_tutorial.ipynb' file.