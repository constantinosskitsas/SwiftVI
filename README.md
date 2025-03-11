# SwiftVI: Time-Efficient Value Iteration for MDPs
## **Introduction**
Markov decision process (MDPs) find application wherever a decision-making agent acts and learns in an uncertain environment from facility management to healthcare and service provisioning. However, learning model parameters and planning the optimal policy such an agent should follow raises high computational cost, calling for solutions that scale to large numbers of actions and states. In this paper, we propose Swift, a suite of algorithms that plan and learn with MDPs scalably by organizing the set of actions for each state in priority queues and deriving bounds for backup Q-values. Our championed solution prunes the set of actions at each state utilizing a tight upper bound and a single priority queue. A thorough experimental study confirms that Swift algorithms achieve high efficiency gains robustly to model parameters.
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

### Artifact Evaluation MLSys 2025:
We recomend a sample experiment with States vs Runtime evaluation in random graph (Figure 1)
Results stored in Results//RandomGraphs_3.txt and Results//AVG_RandomGraphs_3.txt
```shell
g++ -pthread -std=c++17 -o algo_test *.cpp 
./algo_test -EID 3 -A 100 -SS 50 -SP 50 -IP 50 -EP 500 -R 10
```
For faster results (different runtimes from the paper) compile with O3 flag
```shell
g++ -pthread -std=c++17 -O3 -o algo_test *.cpp 
./algo_test -EID 3 -A 100 -SS 50 -SP 50 -IP 50 -EP 500 -R 10
```
### Example output :
Running above command with O3 flag on an Intel Xeon Silver 4316 machine produced following runtimes in Results//AVG_RandomGraphs_3.txt:

|   VIH   |   VIAEHL  |   BAO   |
|--------------|--------------|--------------|
|     31.8     |     47.3     |    103.3     |
|    115.2     |    171.3     |    270.5     |
|    282.2     |    446.1     |    540.8     |
|    600.1     |    910.4     |    956.5     |
|    968.3     |   1481.2     |   1437.4     |
|   1267.4     |   1961.9     |   1796.9     |
|   1805.3     |   2840.2     |   2474.8     |
|   2367.9     |    3769      |   3170.8     |
|   2932.8     |   4734.9     |   3846.8     |
|   3331.9     |   5488.6     |   4307.8     |

It took ~10 min to run.


### How to change experiments :
|   Parameter   |     Meaning     |   Command   |   Initial Value   |
|:--------:|:------------:|:--------:|:--------:|
|  ExperimentID  |  Type of Experiment  |  -EID  |1|
|  States   |  Number of States  | -S|500|
|  Actions        |   Number of Actions    | -A|100|
|  Suported States     |    Number of Supported States    | -SS|50|
|  Start Position      |    Value of Start Position(Overwrites the evaluated parameter i.e -S,-A or -SS)    | -SP|50|
|  Incremental Position  |    Step for every iteration     | -IP|50|
|  End Position     |    Value of end Position    | -EP|500|
|  Repetitions     |    Experiment Repetitions    | -R|10|
## Additional Information : 
Each Experiment ID represents an experimental setting (figure) in the paper.
First step is to compile 
```shell
g++ -pthread -std=c++17 -o algo_test *.cpp
```
We run our experiments without optimization flag. However, if you compile with O3 it significantly reduces the overal runtime.
```shell
g++ -pthread -std=c++17 -O3 algo_test *.cpp
```
Experiment ID 1 and 2 Represent the Cloud management experiment Figure(3).
Reproduce figure 3.a (left). Results stored in 	Results//VMS.txt and Results//VMSavg.txt
	
```shell
./algo_test -EID 1
```
Reproduce figure 3.b (right).Results stored in Results//VMA.txt and Results//VMAavg.txt 
```shell
./algo_test -EID 2
```
Reproduce figure 1.a (left). Random graph, Runtime vs States.
Results stored in Results//RandomGraphs_3.txt and Results//AVG_RandomGraphs_3.txt
```shell
./algo_test -EID 3 -A 100 -SS 50 -SP 50 -IP 50 -EP 500 -R 10
```
Reproduce figure 2.a (left). Random graph, Runtime vs States. 
Best algorithm only.
Results stored in Results//RandomGraphs_4.txt and Results//AVG_RandomGraphs_4.txt
```shell
./algo_test -EID 4 -A 100 -SS 50 -SP 50 -IP 50 -EP 2000 -R 10
```

Reproduce figure 1.b (right). Random graph, Runtime vs Actions. 
Results stored in Results//RandomGraphs_5.txt and Results//AVG_RandomGraphs_5.txt
```shell
./algo_test -EID 5 -S 100 -SS 50 -SP 50 -IP 50 -EP 500 -R 10
```
Reproduce figure 2.b (right). Random graph, Runtime vs Actions. 
Best only.
Results stored in Results//RandomGraphs_6.txt and Results//AVG_RandomGraphs_6.txt
```shell
./algo_test -EID 6 -S 100 -SS 10 -SP 50 -IP 50 -EP 2000 -R 10
```
Reproduce figure 4.a (left). Terain Maze 2d States vs Action. 
Best only.
Results stored in Results//Ter_Maze2D.txt and Results//avgTer_Maze2D.txt
```shell
./algo_test -EID 9 -S 50  -SP 50 -IP 50 -EP 250 -R 10
```
Reproduce figure 4.b (right). Terain Maze 3d States vs Action. 
Best only.
Results stored in Results//Ter_Maze3D.txt and Results//avgTer_Maze3D.txt
```shell
./algo_test -EID 10 -S 10 -SP 10 -IP 5 -EP 40 -R 10
```
## Reference

Please cite our work in your publications if it helps your research:

```
The paper is accepted to MLsys2025. 
```  

