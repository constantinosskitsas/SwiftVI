# SwiftVI: Time-Efficient Value Iteration for MDPs
## **Introduction**
Markov decision process (MDPs) find application wherever a decision-making agent acts and learns in an uncertain environment from facility management to healthcare and service provisioning. However, finding the optimal policy such an agent should follow raises high computational cost, calling for solutions that scale to large numbers of actions and states? In this paper, we propose SwiftVI, a suite of algorithms that solve MDPs scalably by organizing the set of actions for each state in priority queues and deriving bounds for backup Q-values. Our championed solution prunes the set of actions at each state utilizing a tight upper bound and a single priority queue. A thorough experimental study confirms that SwiftVI algorithms achieve high efficiency gains robustly to model parameters.
## Required Libraries
-pthread Basic library for threads in C++

## Algorithms 
      VI : value iteration with Initialized lower bound
      UVI : value iteration with Initialized Upper bound
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

The code was implemented in collaboration with xxx
