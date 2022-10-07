#ifndef VI_ALGORITHM_H
#define VI_ALGORITHM_H

#include <algorithm>
#include <chrono>
#include <tuple>
#include <random>
#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <sstream>

#include "MDP_type_definitions.h"

using namespace std;

V_type value_iteration(S_type S, R_type R, A_type A, P_type P, double gamma, double epsilon);
V_type value_iterationGS(S_type S, R_type R, A_type A, P_type P, double gamma, double epsilon);
V_type value_iterationGSTM(S_type S, R_type R, A_type A, P_type P, double gamma, double epsilon,int D3);

#endif
