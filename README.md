# FrozenLake 4x4 Reinforcement learning
### Frozen Lake description 
```
SFFF       (S: starting point, safe)
FHFH       (F: frozen surface, safe)
FFFH       (H: hole, fall to your doom)
HFFG       (G: goal, where the frisbee is located)
```
__state space__

```
West(0), South(1), East(2), North(3) 
```
__action space__

Game is finish when you reach goal or fall in hole. 
```
if you reach goal reward == 1
if you fall in hole reward == 0
if you step on frozen surface reward == 0
if you step on starting point reward == 0
```
__reward function__


### Value iteration 
![value iteration pseudocode](./images/value_iteration.jpg)

### Monte carlo 
![monte carlo iteration pseudocode](./images/monte_carlo.jpg)

