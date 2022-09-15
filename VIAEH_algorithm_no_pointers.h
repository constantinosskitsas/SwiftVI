#ifndef VIAEH_ALGORITHM_NO_POINTERS_H
#define VIAEH_ALGORITHM_NO_POINTERS_H

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

V_type value_iteration_action_elimination_heaps_no_pointers(S_type S, R_type R, A_type A, P_type P, double gamma, double epsilon);
V_type value_iteration_action_elimination_heaps_no_pointersGS(S_type S, R_type R, A_type A, P_type P, double gamma, double epsilon);

#endif
