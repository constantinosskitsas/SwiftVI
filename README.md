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
      

### How to run experiments 
if -std=c++17 does not work, try c++17.
```shell
g++ -pthread -std=c++17 -o algo_test *.cpp && ./algo_test
g++ -pthread -std=c++20 -o algo_test *.cpp && ./algo_test
```
### Artifact Evaluation MLSys 2025:
We recommend a sample experiment with States vs Runtime evaluation in random graph (Figure 1)
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

| VI      | UVI     | BVI     | VIAE    | VIAEH   | VIH     | VIAEHLB | BAO    |
|---------|---------|---------|---------|---------|---------|---------|--------|
| 122.1   | 122.3   | 236     | 111.1   | 112.4   | 31.3    | 46.3    | 99.5   |
| 516.3   | 536.3   | 953.3   | 450.3   | 442.2   | 120     | 181     | 285.4  |
| 1655    | 1736.4  | 2839.5  | 1312.4  | 1218.1  | 294.9   | 466.5   | 564.2  |
| 3729.2  | 3882.6  | 6041.5  | 2740.9  | 2701.3  | 604.9   | 919.3   | 964.4  |
| 6083.5  | 6306.8  | 9692.1  | 4397.1  | 4284.2  | 940.4   | 1438.9  | 1395.4 |
| 8525.9  | 8838.7  | 14068.9 | 6393.8  | 6096.9  | 1328.9  | 2059.5  | 1890.9 |
| 11400.7 | 11985.5 | 19128.5 | 8774.6  | 8212.3  | 1837.5  | 2885.1  | 2519.5 |
| 14442.4 | 15221.4 | 24829.3 | 11424.6 | 10456.7 | 2343.3  | 3734.2  | 3148.4 |
| 18084.3 | 18956   | 30747.9 | 14212.6 | 12973.6 | 2898.7  | 4682.4  | 3802.5 |
| 21587.7 | 22722.6 | 37376.9 | 17204.2 | 15505.1 | 3400.2  | 5591.4  | 4394.4 |


It took ~1.5 hours to run.


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

Please cite our work in your publications if it helps your research.
The paper is accepted to MLsys2025. 

BibTeX:
```
@inproceedings{
mortensen2025swiftvi,
title={Swift{VI}: Time-Efficient Planning and Learning with {MDP}s},
author={Kasper Overgaard Mortensen and Konstantinos Skitsas and Emil Morre Christensen and Mohammad Sadegh Talebi and Andreas Pavlogiannis and Davide Mottin and Panagiotis Karras},
booktitle={Eighth Conference on Machine Learning and Systems},
year={2025},
url={https://openreview.net/forum?id=ArP6dfc2Mx}
}
```

Links to paper and material:  
https://mlsys.org/virtual/2025/poster/3279  
https://openreview.net/forum?id=ArP6dfc2Mx

The project is based on and expands upon Emil Morre Christensen's master's thesis where the theoretical justification for the Swift methods were developed.
You can find a repository for the master's thesis here: https://github.com/Dugtoud/Time-Efficient-VI-for-MDPs 
 



