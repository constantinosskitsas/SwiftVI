#ifndef TOP_ACTION_CHANGE_PLOT_H
#define TOP_ACTION_CHANGE_PLOT_H

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

tuple<int, vector<int>> top_action_change(S_type S, R_type R, A_type A, P_type P, double gamma, double epsilon);

#endif
