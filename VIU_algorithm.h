#ifndef VIU_ALGORITHM_H
#define VIU_ALGORITHM_H

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

V_type value_iteration_upper(S_type S, R_type R, A_type A, P_type P, double gamma, double epsilon);
V_type value_iteration_upperGS(S_type S, R_type R, A_type A, P_type P, double gamma, double epsilon);
V_type value_iteration_upperGSTM(S_type S, R_type R, A_type A, P_type P, double gamma, double epsilon,int D3);
V_type value_iteration_upperGSPS(S_type S, R_type R, A_type A, P_type P, double gamma, double epsilon);

#endif
