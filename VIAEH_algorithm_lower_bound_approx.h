#ifndef VIAEH_ALGORITHM_LOWER_BOUND_APPROX_H
#define VIAEH_ALGORITHM_LOWER_BOUND_APPROX_H

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

V_type value_iteration_action_elimination_heaps_lower_bound_approx(S_type S, R_type R, A_type A, P_type P, double gamma, double epsilon);
V_type value_iteration_action_elimination_heaps_lower_bound_approxGS(S_type S, R_type R, A_type A, P_type P, double gamma, double epsilon);
V_type value_iteration_action_elimination_heaps_lower_bound_approxA(S_type S, R_type R, A_type A, P_type P, double gamma, double epsilon);
V_type value_iteration_action_elimination_heaps_lower_bound_approxGSSki(S_type S, R_type R, A_type A, P_type P, double gamma, double epsilon);
V_type value_iteration_action_elimination_heaps_lower_bound_approxGSTM(S_type S, R_type R, A_type A, P_type P, double gamma, double epsilon,int D3);

#endif
