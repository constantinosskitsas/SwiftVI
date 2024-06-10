#ifndef MDP_GENERATION_H
#define MDP_GENERATION_H

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
#include <cstdlib>

#include "MDP_type_definitions.h"

using namespace std;

MDP_type FixedGridWorld();
MDP_type GridWorld(int X, int Y, int seed, int wrong_box);

MDP_type generate_random_MDP_with_variable_parameters(int S, int A_num, double action_prob, double non_zero_prob, double upper_bound_reward, int seed);

MDP_type generate_random_MDP_with_variable_parameters_and_reward(int S, int A_num, double action_prob, double non_zero_prob, double reward_factor, double reward_prob, double upper_bound_reward, int seed);

MDP_type generate_random_MDP_with_variable_parameters_fixed_nonzero_trans_states(int S, int A_num, double action_prob, int num_of_nonzero_transition_states, double upper_bound_reward, int seed);

MDP_type generate_random_MDP_normal_distributed_rewards(int S, int A_num, double action_prob, int num_of_nonzero_transition_states, int seed, double reward_dist_mean, double reward_dist_variance);

MDP_type generate_random_MDP_exponential_distributed_rewards(int S, int A_num, double action_prob, int num_of_nonzero_transition_states, double lambda, int seed);
MDP_type readMDPS(string Rseed, string S);
P_type P_fixed_size1(int S, int A_num, int num_of_nonzero_transition_states, default_random_engine &e);
MDP_type RiverSwim(int S);
MDP_type ErgodicRiverSwim(int S);
MDP_type Maze(int X, int Y, int seed);
bool check(int pos[], int ra, int siz);
int posDi(int X1, int Y1, int Swi, int Dir);
MDP_type Maze3d(int X, int Y, int Z, int seed);
// int[] to3D( int idx );
int to1D(int x, int y, int z, int xMax, int yMax);

#endif
