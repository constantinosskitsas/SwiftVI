# Space-Efficiency-in-Finite-Horizon-MDPs
## **Introduction**
Can an infinite-horizon Markov decision process (IHMDP) run robustly to model parameters and scale to large numbers of actions, states, and supports? These models find application in cases where a decision-making agent acts in an uncertain environment from facility management to healthcare and service provisioning. However, computing optimal policies such an agent should follow by dynamic programming value iteration raises high computational cost. This scalability question has been largely neglected. In this paper, we propose \name, a suite of algorithms that achieve robust and scalable MDP value iteration via organizing the set of actions for each state in priority queues and utilizing lower and upper bounds for the backup Q-values for actions. Our championed solution prunes the set of actions to be considered for each state utilizing a tight upper bound and a single priority queue. A thorough experimental study \pkd{under diverse settings} confirms that \name algorithms achieve high efficiency gains robustly to model parameters.
## Required Libraries
-pthread Basic library for threads in C++

## Algorithms 
      Value Iteration : value iteration with Initialized lower bound
      Upper Value Iteration : value iteration with Initialized Upper bound
      IVI : Interval Value Iteration
      VIH : Value Iteration Heap(Proposed Algorithm)
      VIAE: Value Iteration Action Elimination
      VIAEH: Value Iteration Action Elimination Heap
      VIAEHL: Value Iteration Action Elimination Heap Lower bound Approximatioi (Proposed Algorithm)
      BAO :Best Action Only Value Iteration
      

### How to run experiments :
g++ -pthread -std=c++17 -o algo_test *.cpp && ./algo_test
## Additional Information : 

## Reference

Please cite our work in your publications if it helps your research:

```
The paper is under submission. 
```  

The code was implemented in collaboration with Emil Morre Christensen @Dugtoud.
