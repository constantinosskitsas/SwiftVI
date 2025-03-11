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
      VIAEHL: Value Iteration Action Elimination Heap Lower bound Approximation
      BAO :Best Action Only Value Iteration
      

### How to run experiments :
g++ -pthread -std=c++17 -o algo_test *.cpp && ./algo_test

When done the program will override a file in /Skitas with a table including runtimes of algorithms included in the experiment.

### How to change experiments :
|   Parameter   |     Meaning     |   Command   |   Initial Value   |
|:--------:|:------------:|:--------:|:--------:|
|  ExperimentID  |  Type of Experiment  |  -EID  |1|
|  States   |  Number of States  | -S|500|
|  Actions        |   Number of Actions    | -A|100|
|  Suported States     |    Number of Supported States    | -SS|50|
|  Start Position      |    Value of Start Position    | -SP|50|
|  Incremental Position  |    Step for every iteration     | -IP|50|
|  End Position     |    Value of end Position    | -EP|500|
## Additional Information : 

## Reference

Please cite our work in your publications if it helps your research:

```
The paper is under submission. 
```  

The code was implemented in collaboration with xxx
