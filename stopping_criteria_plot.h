#ifndef STOPPING_CRITERIA_PLOT_H
#define STOPPING_CRITERIA_PLOT_H

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

vector<int> first_convergence_iteration(S_type S, R_type R, A_type A, P_type P, double gamma, double epsilon);
#endif
