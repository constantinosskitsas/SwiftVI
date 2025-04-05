#ifndef __REPS_SWIFTVI__TIME_EFFICIENT_VALUE_ITERATION_ANCVI_ALGORITHM_H_
#define __REPS_SWIFTVI__TIME_EFFICIENT_VALUE_ITERATION_ANCVI_ALGORITHM_H_

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

V_type anc_valueiteration(S_type S, R_type R, A_type A, P_type P, double gamma, double epsilon);


#endif // __REPS_SWIFTVI__TIME_EFFICIENT_VALUE_ITERATION_ANCVI_ALGORITHM_H_