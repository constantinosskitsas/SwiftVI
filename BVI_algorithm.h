#ifndef BVI_ALGORITHM_H
#define BVI_ALGORITHM_H

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

V_type bounded_value_iteration(S_type S, R_type R, A_type A, P_type P, double gamma, double epsilon);
V_type bounded_value_iterationGS(S_type S, R_type R, A_type A, P_type P, double gamma, double epsilon);
V_type bounded_value_iterationGSTest(S_type S, R_type R, A_type A, P_type P, double gamma, double epsilon);
//V_type bounded_value_iterationGSTest(S_type S, R_type R, A_type A, P_type P, double gamma, double epsilon);

#endif
