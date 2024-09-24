#ifndef VIH_ALGORITHM_H
#define VIH_ALGORITHM_H

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
#define SIZE_T long long int
using namespace std;

bool cmp_action_value_pairs(const q_action_pair_type &a, q_action_pair_type &b);
V_type value_iteration_with_heapGS(S_type S, R_type R, A_type A, P_type P, double gamma, double epsilon);
V_type value_iteration_with_heap(S_type S, R_type R, A_type A, P_type P, double gamma, double epsilon);
V_type value_iteration_with_heapGSTM(S_type S, R_type R, A_type A, P_type P, double gamma, double epsilon, int D3);
V_type value_iteration_with_heapGSPS(S_type S, R_type R, A_type A, P_type P, double gamma, double epsilon);

#endif
