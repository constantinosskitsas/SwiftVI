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
#include <unordered_set>

#include "MDP_type_definitions.h"
#include "pretty_printing_MDP.h"
#include "MDP_generation.h"
#include "VI_algorithms_helper_methods.h"
#include "VI_algorithm.h"
#include "BVI_algorithm.h"
#include "VIAE_algorithm.h"
#include "VIAEH_algorithm.h"
#include "VIH_algorithm.h"
#include "experiments.h"

using namespace std;
using namespace std::chrono;

R_type R_uniform_random_distribution(int S, int A_num, double upper_bound_reward, default_random_engine &e)
{

	uniform_real_distribution<double> uniform_reward(0, upper_bound_reward);
	R_type R;
	for (int i = 0; i < S; ++i)
	{
		vector<double> R_s;
		for (int j = 0; j < A_num; ++j)
		{
			double rand_reward = uniform_reward(e);
			R_s.push_back(rand_reward);
		}
		R.push_back(R_s);
	}
	return R;
}

R_type R_normal_distribution(int S, int A_num, double reward_dist_mean, double reward_dist_variance, default_random_engine &e)
{

	normal_distribution<double> normal_distributed_reward(reward_dist_mean, reward_dist_variance);
	R_type R;
	for (int i = 0; i < S; ++i)
	{
		vector<double> R_s;
		for (int j = 0; j < A_num; ++j)
		{
			double rand_reward = normal_distributed_reward(e);
			R_s.push_back(rand_reward);
		}
		R.push_back(R_s);
	}
	return R;
}

R_type R_exponential_distribution(int S, int A_num, double lambda, default_random_engine &e)
{

	exponential_distribution<double> exponential_distributed_reward(lambda);

	R_type R;
	for (int i = 0; i < S; ++i)
	{
		vector<double> R_s;
		for (int j = 0; j < A_num; ++j)
		{
			double rand_reward = exponential_distributed_reward(e);
			R_s.push_back(rand_reward);
		}
		R.push_back(R_s);
	}
	return R;
}

R_type R_uniform_distribution_reward_prob(int S, int A_num, double upper_bound_reward, double reward_factor, double reward_prob, default_random_engine &e)
{
	// Generate a reward for each state-action pair
	uniform_real_distribution<double> uniform_reward(0, upper_bound_reward);

	// The prob that an reward is large
	bernoulli_distribution b_reward(reward_prob);

	R_type R;
	for (int i = 0; i < S; ++i)
	{
		vector<double> R_s;
		for (int j = 0; j < A_num; ++j)
		{
			double rand_reward = uniform_reward(e);

			// Is the reward to be a factor reward_factor larger
			if (b_reward(e))
			{
				rand_reward = reward_factor * rand_reward;
			}
			R_s.push_back(rand_reward);
		}
		R.push_back(R_s);
	}
	return R;
}

// Generate an action set for each state
A_type A_generate(int S, int A_num, double action_prob, default_random_engine &e)
{

	// bernouli distribution that returns true with probability A_prob
	// TODO: static or not?
	bernoulli_distribution b_A(action_prob);

	// OBS: check if action set is empty and redo if it is. Need to have at least one action.
	A_type A;
	int average_size = 0;
	for (int i = 0; i < S; ++i)
	{
		vector<int> A_s;
		for (int i = 0; i < A_num; ++i)
		{
			if (b_A(e))
			{
				// The action i is chosen to be in set A(s)
				A_s.push_back(i);
			}
		}
		// Have insert this to error when there are no actions, which is a mistake!
		if (A_s.size() == 0)
		{
			printf("NO ACTIONS\n");
			A_s.push_back(0);
		}
		average_size += A_s.size();
		A.push_back(A_s);
	}
	double average = double(average_size) / double(S);

	return A;
}

P_type P_nonzero_probability(int S, int A_num, double non_zero_prob, default_random_engine &e)
{
	// Generate probability distribution P

	// To select random
	uniform_int_distribution<> rand_state(0, S - 1);

	// bernouli distribution that returns true with probability A_prob
	bernoulli_distribution b_P(non_zero_prob);
	// From this random number generator, it generates an double value between 0 and 1, i.e. an probability
	// From this random number generator, it generates an double value between 0 and upper_bound_reward, i.e. an unifrmly random reward
	uniform_real_distribution<double> uniform_prob_dist(0, 1);
	P_type P;
	for (int i = 0; i < S; ++i)
	{
		// we now fix state s
		vector<pair<vector<double>, vector<int>>> P_s;

		// for each action
		for (int i = 0; i < A_num; ++i)
		{
			// prob dist induced by choosing action a
			vector<double> P_s_a;

			// keeps the states that have a non-zero probability to transition to
			vector<int> P_s_a_nonzero_states;
			for (int j = 0; j < S; ++j)
			{
				bool a_is_nonzero = b_P(e);
				if (a_is_nonzero)
				{
					double trans_prob = uniform_prob_dist(e);
					P_s_a.push_back(trans_prob);

					// add the state index to the vector of nonzero states
					P_s_a_nonzero_states.push_back(j);
				}
				else
				{
					P_s_a.push_back(double(0));
				}
			}
			// make it a distribution that sums to
			double sum = accumulate(P_s_a.begin(), P_s_a.end(), 0.0);
			if (sum == 0.0)
			{
				printf("NO TRANSITIONS\n");
				int rand_trans_state = rand_state(e);
				printf("state %d\n", rand_trans_state);
				P_s_a[rand_trans_state] = 1.0;
			}
			for (int j = 0; j < S; ++j)
			{
				P_s_a[j] *= (1.0 / sum);
			}

			// TODO use emplace_back here instead for better performance
			P_s.push_back(make_pair(P_s_a, P_s_a_nonzero_states));
		}
		P.push_back(P_s);
	}

	return P;
}

P_type P_fixed_size(int S, int A_num, int num_of_nonzero_transition_states, default_random_engine &e)
{

	// Generate probability distribution P

	// To select random
	uniform_int_distribution<> rand_state(0, S - 1);

	// From this random number generator, it generates an double value between 0 and 1, i.e. an probability
	// From this random number generator, it generates an double value between 0 and upper_bound_reward, i.e. an unifrmly random reward
	uniform_real_distribution<double> uniform_prob_dist(0, 1);
	P_type P;
	for (int i = 0; i < S; ++i)
	{
		// we now fix state s
		vector<pair<vector<double>, vector<int>>> P_s;

		// for each action
		for (int i = 0; i < A_num; ++i)
		{
			// prob dist induced by choosing action a
			// init with all zeros and only change those that are nonzero

			vector<double> P_s_a(S, double(0));

			// keeps the states that have a non-zero probability to transition to

			// generate list of indicies and shuffle using the random engine e defined above
			int states[S];
			for (int i = 0; i < S; i++)
			{
				states[i] = i;
			}
			shuffle(states, states + S, e);
			vector<int> P_s_a_nonzero_states(states, states + num_of_nonzero_transition_states);
			sort(P_s_a_nonzero_states.begin(), P_s_a_nonzero_states.end());

			// give the nonzero states probabilities
			for (int nonzero_state : P_s_a_nonzero_states)
			{
				double trans_prob = uniform_prob_dist(e);
				P_s_a[nonzero_state] = trans_prob;
			}

			// make it a distribution that sums to
			double sum = accumulate(P_s_a.begin(), P_s_a.end(), 0.0);
			for (int j = 0; j < S; ++j)
			{
				P_s_a[j] *= (1.0 / sum);
			}

			// TODO use emplace_back here instead for better performance
			P_s.push_back(make_pair(P_s_a, P_s_a_nonzero_states));
		}
		P.push_back(P_s);
	}

	return P;
}

P_type P_fixed_size1(int S, int A_num, int num_of_nonzero_transition_states, default_random_engine &e)
{

	// Generate probability distribution P

	// To select random
	uniform_int_distribution<> rand_state(0, num_of_nonzero_transition_states - 1);

	// From this random number generator, it generates an double value between 0 and 1, i.e. an probability
	// From this random number generator, it generates an double value between 0 and upper_bound_reward, i.e. an unifrmly random reward
	uniform_real_distribution<double> uniform_prob_dist(0, 1);
	P_type P;
	for (int i = 0; i < S; ++i)
	{
		// we now fix state s
		vector<pair<vector<double>, vector<int>>> P_s;

		// for each action
		for (int i = 0; i < A_num; ++i)
		{
			// prob dist induced by choosing action a
			// init with all zeros and only change those that are nonzero

			vector<double> P_s_a(num_of_nonzero_transition_states, double(0));

			// keeps the states that have a non-zero probability to transition to

			// generate list of indicies and shuffle using the random engine e defined above
			int states[S];
			for (int i = 0; i < S; i++)
			{
				states[i] = i;
			}
			shuffle(states, states + S, e);
			vector<int> P_s_a_nonzero_states(states, states + num_of_nonzero_transition_states);
			sort(P_s_a_nonzero_states.begin(), P_s_a_nonzero_states.end());

			// give the nonzero states probabilities

			for (int i = 0; i < num_of_nonzero_transition_states; i++)
			{
				double trans_prob = uniform_prob_dist(e);
				P_s_a[i] = trans_prob;
			}

			// make it a distribution that sums to
			double sum = accumulate(P_s_a.begin(), P_s_a.end(), 0.0);
			for (int j = 0; j < num_of_nonzero_transition_states; ++j)
			{
				P_s_a[j] *= (1.0 / sum);
			}

			// TODO use emplace_back here instead for better performance
			P_s.push_back(make_pair(P_s_a, P_s_a_nonzero_states));
		}
		P.push_back(P_s);
	}

	return P;
}

MDP_type generate_random_MDP_with_variable_parameters(int S, int A_num, double action_prob, double non_zero_prob, double upper_bound_reward, int seed)
{

	// make random engine
	static default_random_engine e(seed);

	R_type R = R_uniform_random_distribution(S, A_num, upper_bound_reward, e);
	A_type A = A_generate(S, A_num, action_prob, e);
	P_type P = P_nonzero_probability(S, A_num, non_zero_prob, e);

	MDP_type MDP = make_tuple(R, A, P);

	return MDP;
}

// with some probability, make each reward a factor larger than the others
MDP_type generate_random_MDP_with_variable_parameters_and_reward(int S, int A_num, double action_prob, double non_zero_prob, double reward_factor, double reward_prob, double upper_bound_reward, int seed)
{

	// We give it a seed to control the generation
	static default_random_engine e(seed);

	R_type R = R_uniform_distribution_reward_prob(S, A_num, upper_bound_reward, reward_factor, reward_prob, e);
	A_type A = A_generate(S, A_num, action_prob, e);
	P_type P = P_nonzero_probability(S, A_num, non_zero_prob, e);

	MDP_type MDP = make_tuple(R, A, P);

	return MDP;
}

// fixed number of non-zero transition states in each state
MDP_type generate_random_MDP_with_variable_parameters_fixed_nonzero_trans_states(int S, int A_num, double action_prob, int num_of_nonzero_transition_states, double upper_bound_reward, int seed)
{

	// We give it a seed to control the generation
	static default_random_engine e(seed);

	R_type R = R_uniform_random_distribution(S, A_num, upper_bound_reward, e);
	A_type A = A_generate(S, A_num, action_prob, e);
	P_type P = P_fixed_size(S, A_num, num_of_nonzero_transition_states, e);

	MDP_type MDP = make_tuple(R, A, P);

	return MDP;
}

MDP_type generate_random_MDP_normal_distributed_rewards(int S, int A_num, double action_prob, int num_of_nonzero_transition_states, int seed, double reward_dist_mean, double reward_dist_variance)
{

	static default_random_engine e(seed);

	R_type R = R_normal_distribution(S, A_num, reward_dist_mean, reward_dist_variance, e);
	A_type A = A_generate(S, A_num, action_prob, e);
	P_type P = P_fixed_size1(S, A_num, num_of_nonzero_transition_states, e);

	MDP_type MDP = make_tuple(R, A, P);

	return MDP;
}

MDP_type generate_random_MDP_exponential_distributed_rewards(int S, int A_num, double action_prob, int num_of_nonzero_transition_states, double lambda, int seed)
{

	static default_random_engine e(seed);

	R_type R = R_exponential_distribution(S, A_num, lambda, e);
	A_type A = A_generate(S, A_num, action_prob, e);
	P_type P = P_fixed_size(S, A_num, num_of_nonzero_transition_states, e);

	MDP_type MDP = make_tuple(R, A, P);

	return MDP;
}

MDP_type readMDPS(string Rseed, string S)
{
	string Rfile = (Rseed + "a/" + "Rewards" + S + ".txt");
	string Afile = (Rseed + "a/" + "Actions" + S + ".txt");
	string Pfile = (Rseed + "a/" + "Transitions" + S + ".txt");
	ifstream Rdata;
	ifstream Adata;
	ifstream Pdata;
	double num;
	int nofstate;
	vector<double> mathekodika;
	vector<int> axriste;
	vector<pair<vector<double>, vector<int>>> P_sp;
	P_type P;
	Pdata.open(Pfile);
	while (!Pdata.eof())
	{				  // keep reading until end-of-file
		Pdata >> num; // sets EOF flag if no value found
		if (num == -123456)
		{
			P_sp.push_back(make_pair(mathekodika, axriste));
			mathekodika.clear();
			axriste.clear();
		}
		else if (num == -654321 && !P_sp.empty())
		{
			P.push_back(P_sp);
			P_sp.clear();
			// vector<pair<vector<double>,vector<int>>> P_sp;
		}
		else
		{
			Pdata >> nofstate;
			mathekodika.push_back(num);
			axriste.push_back(nofstate);
		}
	}
	A_type A;
	vector<int> axriste1;
	Rdata.close();
	Adata.open(Afile);
	while (!Adata.eof())
	{					   // keep reading until end-of-file
		Adata >> nofstate; // sets EOF flag if no value found
		if (nofstate == -654321 && !axriste1.empty())
		{
			A.push_back(axriste1);
			axriste1.clear();
		}
		else
			axriste1.push_back(nofstate);
	}
	Adata.close();
	R_type R;
	vector<double> axriste2;
	Rdata.close();
	Rdata.open(Rfile);
	while (!Rdata.eof())
	{				  // keep reading until end-of-file
		Rdata >> num; // sets EOF flag if no value found
		if (num == -654321 && !axriste2.empty())
		{
			R.push_back(axriste2);
			axriste2.clear();
		}
		else
			axriste2.push_back(num);
	}
	Rdata.close();
	MDP_type MDP = make_tuple(R, A, P);
	return MDP;
}

MDP_type ErgodicRiverSwim(int S)
{
	// Create R
	R_type R;
	vector<double> R_s0;
	R_s0.push_back(0.05);
	R_s0.push_back(0);
	R.push_back(R_s0);
	for (int i = 1; i < S - 1; ++i)
	{
		vector<double> R_s;
		for (int j = 0; j < 2; ++j)
		{
			R_s.push_back(0);
		}
		R.push_back(R_s);
	}
	R_s0.clear();
	R_s0.push_back(0);
	R_s0.push_back(1);
	R.push_back(R_s0);

	// create A
	A_type A;
	vector<int> A_s;
	A_s.push_back(0);
	A_s.push_back(1);
	for (int i = 0; i < S; ++i)
	{
		A.push_back(A_s);
	}

	P_type P;
	vector<pair<vector<double>, vector<int>>> P_s;
	vector<int> P_s_a_nonzero_states;
	// vector<double> P_s_a(2, double(0));
	vector<double> P_s_a;
	P_s_a_nonzero_states.push_back(0);
	P_s_a_nonzero_states.push_back(1);
	P_s_a.push_back(0.95);
	P_s_a.push_back(0.05);
	P_s.push_back(make_pair(P_s_a, P_s_a_nonzero_states));
	P_s_a_nonzero_states.clear();
	P_s_a.clear();
	P_s_a_nonzero_states.push_back(0);
	P_s_a.push_back(0.6);
	P_s_a_nonzero_states.push_back(1);
	P_s_a.push_back(0.4);
	P_s.push_back(make_pair(P_s_a, P_s_a_nonzero_states));
	P.push_back(P_s);
	P_s_a_nonzero_states.clear();
	P_s_a.clear();
	P_s.clear();

	for (int i = 1; i < S - 1; ++i)
	{
		// we now fix state s

		P_s_a_nonzero_states.push_back(i - 1);
		P_s_a_nonzero_states.push_back(i + 1);
		P_s_a.push_back(0.95);
		P_s_a.push_back(0.05);
		P_s.push_back(make_pair(P_s_a, P_s_a_nonzero_states));
		P_s_a_nonzero_states.clear();
		P_s_a.clear();
		P_s_a_nonzero_states.push_back(i - 1);
		P_s_a.push_back(0.05);
		P_s_a_nonzero_states.push_back(i);
		P_s_a.push_back(0.55);
		P_s_a_nonzero_states.push_back(i + 1);
		P_s_a.push_back(0.40);
		P_s.push_back(make_pair(P_s_a, P_s_a_nonzero_states));
		P.push_back(P_s);

		P_s_a_nonzero_states.clear();
		P_s_a.clear();
		P_s.clear();
	}

	P_s_a_nonzero_states.push_back(S - 2);
	P_s_a.push_back(1);
	P_s.push_back(make_pair(P_s_a, P_s_a_nonzero_states));
	P_s_a_nonzero_states.clear();
	P_s_a.clear();
	P_s_a_nonzero_states.push_back(S - 1);
	P_s_a.push_back(0.6);
	P_s_a_nonzero_states.push_back(S - 2);
	P_s_a.push_back(0.4);
	P_s.push_back(make_pair(P_s_a, P_s_a_nonzero_states));
	P.push_back(P_s);

	MDP_type MDP = make_tuple(R, A, P);

	return MDP;
}

MDP_type RiverSwim(int S)
{
	// Create R
	R_type R;
	vector<double> R_s0;
	R_s0.push_back(0.015);
	R_s0.push_back(0);
	R.push_back(R_s0);
	for (int i = 1; i < S - 1; ++i)
	{
		vector<double> R_s;
		for (int j = 0; j < 2; ++j)
		{
			R_s.push_back(0);
		}
		R.push_back(R_s);
	}
	R_s0.clear();
	R_s0.push_back(0);
	R_s0.push_back(1);
	R.push_back(R_s0);

	// create A
	A_type A;
	vector<int> A_s;
	A_s.push_back(0);
	A_s.push_back(1);
	for (int i = 0; i < S; ++i)
	{
		A.push_back(A_s);
	}

	P_type P;
	vector<pair<vector<double>, vector<int>>> P_s;
	vector<int> P_s_a_nonzero_states;
	// vector<double> P_s_a(2, double(0));
	vector<double> P_s_a;
	P_s_a_nonzero_states.push_back(0);
	P_s_a.push_back(1);
	P_s.push_back(make_pair(P_s_a, P_s_a_nonzero_states));
	P_s_a_nonzero_states.clear();
	P_s_a.clear();
	P_s_a_nonzero_states.push_back(0);
	P_s_a.push_back(0.6);
	P_s_a_nonzero_states.push_back(1);
	P_s_a.push_back(0.4);
	P_s.push_back(make_pair(P_s_a, P_s_a_nonzero_states));
	P.push_back(P_s);
	P_s_a_nonzero_states.clear();
	P_s_a.clear();
	P_s.clear();

	for (int i = 1; i < S - 1; ++i)
	{
		// we now fix state s

		P_s_a_nonzero_states.push_back(i - 1);
		P_s_a.push_back(1);
		P_s.push_back(make_pair(P_s_a, P_s_a_nonzero_states));
		P_s_a_nonzero_states.clear();
		P_s_a.clear();
		P_s_a_nonzero_states.push_back(i - 1);
		P_s_a.push_back(0.05);
		P_s_a_nonzero_states.push_back(i);
		P_s_a.push_back(0.55);
		P_s_a_nonzero_states.push_back(i + 1);
		P_s_a.push_back(0.40);
		P_s.push_back(make_pair(P_s_a, P_s_a_nonzero_states));
		P.push_back(P_s);

		P_s_a_nonzero_states.clear();
		P_s_a.clear();
		P_s.clear();
	}

	P_s_a_nonzero_states.push_back(S - 2);
	P_s_a.push_back(1);
	P_s.push_back(make_pair(P_s_a, P_s_a_nonzero_states));
	P_s_a_nonzero_states.clear();
	P_s_a.clear();
	P_s_a_nonzero_states.push_back(S - 1);
	P_s_a.push_back(0.6);
	P_s_a_nonzero_states.push_back(S - 2);
	P_s_a.push_back(0.4);
	P_s.push_back(make_pair(P_s_a, P_s_a_nonzero_states));
	P.push_back(P_s);

	MDP_type MDP = make_tuple(R, A, P);

	return MDP;
}
// width =5
// x=2 and y =3
// i = x + width*y;
// i=   2+5*3
// i=17
// x = i % width;    // % is the "modulo operator", the remainder of i / width;
// y = i / width;    // where "/" is an integer division
// x= 17%5 2
int posDi(int X1, int Y1, int Swi, int Dir)
{
	if (Dir == 0)
		X1 = X1 + 1;
	else if (Dir == 1)
		Y1 = Y1 + 1;
	else if (Dir == 2)
		X1 = X1 - 1;
	else if (Dir == 3)
		Y1 = Y1 - 1;
	else if (Dir == 4)
	{
		Y1 = Y1 + 1;
		X1 = X1 + 1;
	}
	else if (Dir == 5)
	{
		X1 = X1 - 1;
		Y1 = Y1 - 1;
	}

	else if (Dir == 6)
	{
		X1 = X1 - 1;
		Y1 = Y1 + 1;
	}
	else if (Dir == 7)
	{
		X1 = X1 + 1;
		Y1 = Y1 - 1;
	}

	return X1 + Swi * Y1;
}

bool check(int pos[], int ra, int siz)
{
	for (int i = 0; i < siz; i++)
		if (pos[i] == ra)
			return false;
	return true;
}

MDP_type Maze(int X, int Y, int seed)
{
	static default_random_engine e(seed);
	uniform_real_distribution<double> uniform_prob_dist(0, 1);
	double trans_prob = uniform_prob_dist(e);
	int G_X = X - 2;
	int G_Y = Y - 2;

	// 0->x+1 y+0 East
	// 1->x+0 y+1 NOrth
	// 2->x-1 y+0 West
	// 3->x+0 y-1 South
	// 4->x+1 y+1 North East
	// 5->x-1 y-1 South West
	// 6->x-1 y+1 North West
	// 7->x+1 y-1 South East

	double cP[8];
	vector<int> A_s;
	vector<int> A_sD;
	A_type A;
	A_type A_direction;
	int S = X * Y;
	int x_curr = -1;
	int y_curr = -1;
	int counter = 0;
	float pi = 0.10;
	// float pit1=0.4;
	// float pit3=0.65;
	// float pit6=0.85;
	// float pit10=1;

	for (int i = 0; i < S; ++i)
	{
		x_curr = i % X;
		y_curr = i / X;
		if (x_curr == G_X && y_curr == G_Y)
		{
			A_s.push_back(0);
			A_sD.push_back(0);
			A.push_back(A_s);
			A_direction.push_back(A_sD);
			A_s.clear();
			A_sD.clear();
			continue;
		}
		// 0->x+1 y+0 East
		counter = 0;
		while (counter < 2)
		{
			for (int j = 0; j < 8; j++)
				cP[j] = uniform_prob_dist(e);
			if (x_curr + 1 < X && cP[1] > pi)
			{
				A_s.push_back(counter);
				A_sD.push_back(0);
				counter++;
			}
			// 1->x+0 y+1 NOrth

			if (y_curr + 1 < Y && cP[2] > pi)
			{
				A_s.push_back(counter);
				A_sD.push_back(1);
				counter++;
			}
			// 2->x-1 y+0 West
			if (x_curr - 1 > 0 && cP[3] > pi)
			{
				A_s.push_back(counter);
				A_sD.push_back(2);
				counter++;
			}
			// 3->x+0 y-1 South
			if (y_curr - 1 > 0 && cP[4] > pi)
			{
				A_s.push_back(counter);
				A_sD.push_back(3);
				counter++;
			}
			// 4->x+1 y+1 North East
			if (x_curr + 1 < X && y_curr + 1 < Y && cP[5] > pi)
			{
				A_s.push_back(counter);
				A_sD.push_back(4);
				counter++;
			}
			// 5->x-1 y-1 South West
			if (x_curr - 1 > 0 && y_curr - 1 > 0 && cP[6] > pi)
			{
				A_s.push_back(counter);
				A_sD.push_back(5);
				counter++;
			}
			// 6->x-1 y+1 North West
			if (x_curr - 1 > 0 && y_curr + 1 < Y && cP[7] > pi)
			{
				A_s.push_back(counter);
				A_sD.push_back(6);
				counter++;
			}
			// 7->x+1 y-1 South East
			if (x_curr + 1 < X && y_curr - 1 > 0 && cP[0] > pi)
			{
				A_s.push_back(counter);
				A_sD.push_back(7);
				counter++;
			}
		}

		A.push_back(A_s);
		A_direction.push_back(A_sD);
		A_s.clear();
		A_sD.clear();
	}

	A_s.push_back(0);
	A_sD.push_back(0);
	A.push_back(A_s);
	A_direction.push_back(A_sD);

	float pit1 = 0.4;
	float pit2 = 0.50;
	float pit3 = 0.7;
	float pit4 = 0.7;
	float pit5 = 0.70;
	float pit6 = 0.70;
	float pit7 = 0.9;
	float pit8 = 0.9;
	float pit9 = 0.9;
	float pit10 = 1;

	/*
		float pit1=0.1;
		float pit2=0.20;
		float pit3=0.30;
		float pit4=0.40;
		float pit5=0.5;
		float pit6=0.60;
		float pit7=0.7;
		float pit8=0.8;
		float pit9=0.9;
		float pit10=1;*/
	// Create R
	R_type R;
	int metrw = 0;
	int metrw1 = 0;
	for (int i = 0; i < S; ++i)
	{
		vector<double> R_s;
		x_curr = i % X;
		y_curr = i / X;
		if (x_curr == G_X && y_curr == G_Y)
		{
			R_s.push_back(0);
		}
		else
		{
			/*if (uniform_prob_dist(e)<0.05){
				for (auto a : A[i]) {
					R_s.push_back(-X);
					metrw1++;
				}
				}
				else{
					for (auto a : A[i]) {
					R_s.push_back(-1);
					metrw++;
				}
				}
			*/
			for (auto a : A[i])
			{
				if (uniform_prob_dist(e) < pit1)
				{
					R_s.push_back(-1);
					metrw1++;
				}
				else if (uniform_prob_dist(e) < pit2)
				{
					R_s.push_back(-1);
					metrw++;
				}
				else if (uniform_prob_dist(e) < pit3)
				{
					R_s.push_back(-3);
					metrw++;
				}
				else if (uniform_prob_dist(e) < pit4)
				{
					R_s.push_back(-3);
					metrw++;
				}
				else if (uniform_prob_dist(e) < pit5)
				{
					R_s.push_back(-5);
					metrw++;
				}
				else if (uniform_prob_dist(e) < pit6)
				{
					R_s.push_back(-5);
					metrw++;
				}
				else if (uniform_prob_dist(e) < pit7)
				{
					R_s.push_back(-7);
					metrw++;
				}
				else if (uniform_prob_dist(e) < pit8)
				{
					R_s.push_back(-7);
					metrw++;
				}
				else if (uniform_prob_dist(e) < pit9)
				{
					R_s.push_back(-7);
					metrw++;
				}
				else
				{
					R_s.push_back(-10);
					metrw++;
				}
			}
		}

		R.push_back(R_s);
	}
	// cout<<"traps"<<metrw<<endl;
	// cout<<"normal"<<metrw1<<endl;
	vector<double> R_s;
	R_s.push_back(0);
	R.push_back(R_s);

	P_type P;
	vector<pair<vector<double>, vector<int>>> P_s;
	vector<int> P_s_a_nonzero_states;
	// vector<double> P_s_a(4, double(0));
	vector<double> P_s_a;

	// 0->x+1 y+0 East
	// 1->x+0 y+1 NOrth
	// 2->x-1 y+0 West
	// 3->x+0 y-1 South
	// 4->x+1 y+1 North East
	// 5->x-1 y-1 South West
	// 6->x-1 y+1 North West
	// 7->x+1 y-1 South East
	for (int i = 0; i < S; ++i)
	{
		// we now fix state s

		x_curr = i % X;
		y_curr = i / X;
		if (x_curr == G_X && y_curr == G_Y)
		{
			P_s_a_nonzero_states.push_back(S);
			P_s_a.push_back(1);
			P_s.push_back(make_pair(P_s_a, P_s_a_nonzero_states));
			P.push_back(P_s);
			P_s_a_nonzero_states.clear();
			P_s_a.clear();
			P_s.clear();
		}
		else
		{
			for (auto a : A[i])
			{
				P_s_a_nonzero_states.push_back(posDi(x_curr, y_curr, X, A_direction[i][a]));
				P_s_a.push_back(0.8);
				if (A[i].size() > 4)
				{
					int cc = 0;
					int pos[4] = {-1, -1, -1, -1};
					while (cc < 4)
					{
						int ra = rand() % A[i].size();
						if (ra != a && check(pos, ra, 4))
						{
							pos[cc] = ra;
							cc = cc + 1;
							P_s_a_nonzero_states.push_back(posDi(x_curr, y_curr, X, A_direction[i][ra]));
							P_s_a.push_back(0.05);
						}
					}
				}
				else if (A[i].size() == 4)
				{
					int cc = 0;
					int pos[3] = {-1, -1, -1};
					while (cc < 3)
					{
						int ra = rand() % A[i].size();
						if (ra != a && check(pos, ra, 3))
						{
							pos[cc] = ra;
							cc = cc + 1;
							P_s_a_nonzero_states.push_back(posDi(x_curr, y_curr, X, A_direction[i][ra]));
							if (cc == 1)
								P_s_a.push_back(0.1);
							else
								P_s_a.push_back(0.05);
						}
					}
				}
				else if (A[i].size() == 3)
				{
					int cc = 0;
					int pos[2] = {-1, -1};
					while (cc < 2)
					{
						int ra = rand() % A[i].size();
						if (ra != a && check(pos, ra, 2))
						{
							pos[cc] = ra;
							cc = cc + 1;
							P_s_a_nonzero_states.push_back(posDi(x_curr, y_curr, X, A_direction[i][ra]));
							P_s_a.push_back(0.1);
						}
					}
				}
				else if (A[i].size() == 2)
				{
					int cc = 0;
					while (cc < 1)
					{
						int ra = rand() % A[i].size();
						if (ra != a)
						{
							cc = cc + 1;
							P_s_a_nonzero_states.push_back(posDi(x_curr, y_curr, X, A_direction[i][ra]));
							P_s_a.push_back(0.2);
						}
					}
				}
				P_s.push_back(make_pair(P_s_a, P_s_a_nonzero_states));
				P_s_a.clear();
				P_s_a_nonzero_states.clear();
			}
			P.push_back(P_s);
			P_s_a.clear();
			P_s.clear();
		}
	}
	P_s_a_nonzero_states.push_back(S);
	P_s_a.push_back(1);
	P_s.push_back(make_pair(P_s_a, P_s_a_nonzero_states));
	P.push_back(P_s);
	// cout<<"hiend"<<endl;
	MDP_type MDP = make_tuple(R, A, P);

	return MDP;
}

int to1D(int x, int y, int z, int xMax, int yMax)
{
	return (z * xMax * yMax) + (y * xMax) + x;
}
/*
 int[] to3D( int idx ) {
	int z = idx / (xMax * yMax);
	idx -= (z * xMax * yMax);
	int y = idx / xMax;
	int x = idx % xMax;
	return new int[]{ x, y, z };
}*/

int posDi3D(int X1, int Y1, int Z1, int Xmax, int Ymax, int Dir)
{
	if (Dir == 0)
		X1 = X1 + 1;
	else if (Dir == 1)
		Y1 = Y1 + 1;
	else if (Dir == 2)
		X1 = X1 - 1;
	else if (Dir == 3)
		Y1 = Y1 - 1;
	else if (Dir == 4)
	{
		Y1 = Y1 + 1;
		X1 = X1 + 1;
	}
	else if (Dir == 5)
	{
		X1 = X1 - 1;
		Y1 = Y1 - 1;
	}

	else if (Dir == 6)
	{
		X1 = X1 - 1;
		Y1 = Y1 + 1;
	}
	else if (Dir == 7)
	{
		X1 = X1 + 1;
		Y1 = Y1 - 1;
	}
	else if (Dir == 8)
	{
		X1 = X1 + 1;
		Z1 = Z1 + 1;
	}
	else if (Dir == 9)
	{
		Y1 = Y1 + 1;
		Z1 = Z1 + 1;
	}
	else if (Dir == 10)
	{
		X1 = X1 - 1;
		Z1 = Z1 + 1;
	}
	else if (Dir == 11)
	{
		Y1 = Y1 - 1;
		Z1 = Z1 + 1;
	}
	else if (Dir == 12)
	{
		Y1 = Y1 + 1;
		X1 = X1 + 1;
		Z1 = Z1 + 1;
	}
	else if (Dir == 13)
	{
		X1 = X1 - 1;
		Y1 = Y1 - 1;
		Z1 = Z1 + 1;
	}

	else if (Dir == 14)
	{
		X1 = X1 - 1;
		Y1 = Y1 + 1;
		Z1 = Z1 + 1;
	}
	else if (Dir == 15)
	{
		X1 = X1 + 1;
		Y1 = Y1 - 1;
		Z1 = Z1 + 1;
	}
	else if (Dir == 16)
	{
		X1 = X1 + 1;
		Z1 = Z1 - 1;
	}
	else if (Dir == 17)
	{
		Y1 = Y1 + 1;
		Z1 = Z1 - 1;
	}
	else if (Dir == 18)
	{
		X1 = X1 - 1;
		Z1 = Z1 - 1;
	}
	else if (Dir == 19)
	{
		Y1 = Y1 - 1;
		Z1 = Z1 - 1;
	}
	else if (Dir == 20)
	{
		Y1 = Y1 + 1;
		X1 = X1 + 1;
		Z1 = Z1 - 1;
	}
	else if (Dir == 21)
	{
		X1 = X1 - 1;
		Y1 = Y1 - 1;
		Z1 = Z1 - 1;
	}

	else if (Dir == 22)
	{
		X1 = X1 - 1;
		Y1 = Y1 + 1;
		Z1 = Z1 - 1;
	}
	else if (Dir == 23)
	{
		X1 = X1 + 1;
		Y1 = Y1 - 1;
		Z1 = Z1 - 1;
	}

	return (Z1 * Xmax * Ymax) + (Y1 * Xmax) + X1;
}
// always 4 actions-> if we go to wall stay in the same state
// if we slip to the wall stay in the same state
MDP_type GridWorld(int X, int Y, int seed, int wrong_box)
{
	int G_X = X - 1;
	int G_Y = Y - 1;
	unordered_set<int> WBoxes;
	int S = X * Y;
	int goalState= S - 1;
	// 0->x+1 y+0 East
	// 1->x+0 y+1 NOrth
	// 2->x-1 y+0 West
	// 3->x+0 y-1 South
	// int wrong_box = 2;
	int x_curr = -1;
	int y_curr = -1;
	int counter;
	vector<int> A_s;
	vector<int> A_sD;
	A_type A;
	A_type A_direction;
	if (wrong_box > 0)
	{
		
		for (int i = 0; i < wrong_box; ++i)
		{
			int X_wrong = rand() % X;
			int Y_wrong = rand() % Y;
			WBoxes.insert((X_wrong + (Y_wrong * X)));
		}
		//lets assume that terminal state has left/riht/up/down
		if(WBoxes.count(goalState-1) > 0 && WBoxes.count(goalState+1)>0&& WBoxes.count(goalState+X)>0 && WBoxes.count(goalState-X)>0){
			int validBox = rand() % 4;
			if (validBox==0)
				WBoxes.erase(goalState-1);
			else if(validBox==1)
				WBoxes.erase(goalState+1);
			else if (validBox==2)
				WBoxes.erase(goalState+X);
			else 
				WBoxes.erase(goalState-X);
		}
		//this does not quarantine that a path exists from source to target.
	}
	else
	{
		//int scaling=wrong_box*-1;
		WBoxes.insert(0 + (Y / 2) * X);
		WBoxes.insert(X - 1 + (Y / 2) * X);
		WBoxes.insert(X / 2 + 0);
		WBoxes.insert((X / 2) + ((Y - 1) * X));
		int dd = (X / 2) + ((Y / 2) * X);
		for (int j=wrong_box;j<=-1;j++){
		WBoxes.insert(dd);
		WBoxes.insert(dd + j);
		WBoxes.insert(dd - j);
		WBoxes.insert(dd - X * j);
		WBoxes.insert(dd + X * j);
		}
	}

	
	for (int i = 0; i < S; i++)
	{
		x_curr = i % X;
		y_curr = i / Y;
		counter = 0;
		if (x_curr + 1 < X && WBoxes.count(posDi(x_curr, y_curr, X, 0)) <= 0)
		{
			A_s.push_back(0);
			A_sD.push_back(0);
		}
		else
		{
			A_s.push_back(0);
			A_sD.push_back(-1);
		}
		if (y_curr + 1 < Y && WBoxes.count(posDi(x_curr, y_curr, X, 1)) <= 0)
		{
			A_s.push_back(1);
			A_sD.push_back(1);
		}
		else
		{
			A_s.push_back(1);
			A_sD.push_back(-1);
		}
		if (x_curr - 1 >= 0 && WBoxes.count(posDi(x_curr, y_curr, X, 2)) <= 0)
		{
			A_s.push_back(2);
			A_sD.push_back(2);
		}
		else
		{
			A_s.push_back(2);
			A_sD.push_back(-1);
		}
		if (y_curr - 1 >= 0 && WBoxes.count(posDi(x_curr, y_curr, X, 3)) <= 0)
		{
			A_s.push_back(3);
			A_sD.push_back(3);
		}
		else
		{
			A_s.push_back(3);
			A_sD.push_back(-1);
		}

		A.push_back(A_s);
		A_direction.push_back(A_sD);
		A_s.clear();
		A_sD.clear();
	}
	R_type R;
	for (int i = 0; i < S; i++)
	{
		vector<double> R_s;
		x_curr = i % X;
		y_curr = i / Y;

		for (auto a : A_direction[i])
		{
			if (posDi(x_curr, y_curr, X, a) == S - 1) // make it Goal State
				R_s.push_back(1);
			else
				R_s.push_back(0);
		}
		R.push_back(R_s);
	}
	P_type P;
	vector<pair<vector<double>, vector<int>>> P_s;
	vector<int> P_s_a_nonzero_states;
	vector<double> P_s_a;
	double totalP = 0;
	bool ev = false;
	for (int i = 0; i < S; ++i)
	{
		x_curr = i % X;
		y_curr = i / Y;
		for (auto a : A[i])
		{
			for (auto a1 : A[i])
			{
				if (a1 == a)
				{
					int pos = posDi(x_curr, y_curr, X, a);
					if (A_direction[i][a] >= 0)
						P_s_a_nonzero_states.push_back(posDi(x_curr, y_curr, X, A_direction[i][a]));
					else
					{
						P_s_a_nonzero_states.push_back(i);
					}
					P_s_a.push_back(0.7);
					continue;
				}

				if (((A[i][a] + A[i][a1]) % 2) == 1)
				{
					if (A_direction[i][a1] < 0)
					{
						P_s_a_nonzero_states.push_back(i);
						P_s_a.push_back(0.1);
					}
					else
					{
						P_s_a_nonzero_states.push_back(posDi(x_curr, y_curr, X, A_direction[i][a1]));
						// totalP = totalP + 0.1;
						P_s_a.push_back(0.1);
					}
				}
			}
			P_s_a_nonzero_states.push_back(i);
			P_s_a.push_back(0.1);
			P_s.push_back(make_pair(P_s_a, P_s_a_nonzero_states));
			P_s_a.clear();
			P_s_a_nonzero_states.clear();
		}
		P.push_back(P_s);
		P_s_a.clear();
		P_s.clear();
	}
	MDP_type MDP = make_tuple(R, A, P);
	return MDP;
}

MDP_type Maze3d(int X, int Y, int Z, int seed)
{
	static default_random_engine e(seed);
	uniform_real_distribution<double> uniform_prob_dist(0, 1);
	double trans_prob = uniform_prob_dist(e);
	int G_X = X - 2;
	int G_Y = Y - 2;
	int G_Z = Z - 2;

	// 0->x+1 y+0 East
	// 1->x+0 y+1 NOrth
	// 2->x-1 y+0 West
	// 3->x+0 y-1 South
	// 4->x+1 y+1 North East
	// 5->x-1 y-1 South West
	// 6->x-1 y+1 North West
	// 7->x+1 y-1 South East

	// 8->x+1 y+0 z+1 East up
	// 9->x+0 y+1 z+1 NOrth up
	// 10->x-1 y+0 z+1 West up
	// 11->x+0 y-1 z+1 South up
	// 12->x+1 y+1 z+1 North East up
	// 13->x-1 y-1 z+1 South West up
	// 14->x-1 y+1 z+1 North West up
	// 15->x+1 y-1 z+1 South East up

	// 16->x+1 y+0 z-1 East down
	// 17->x+0 y+1 z-1 NOrth down
	// 18->x-1 y+0 z-1 West down
	// 19->x+0 y-1 z-1 South down
	// 20->x+1 y+1 z-1 North East down
	// 21->x-1 y-1 z-1 South West down
	// 22->x-1 y+1 z-1 North West down
	// 23->x+1 y-1 z-1 South East down

	double cP[24];
	vector<int> A_s;
	vector<int> A_sD;
	A_type A;
	A_type A_direction;
	int S = X * Y * Z;
	int x_curr = -1;
	int y_curr = -1;
	int z_curr = -1;
	int idx = -1;
	int counter = 0;
	float pi = 0.10; // remember to make sure that a path to the goal state exists.(however with low probability the chances of no path are small.)
	// float pit1=0.4;
	// float pit3=0.65;
	// float pit6=0.85;
	// float pit10=1;

	for (int i = 0; i < S; ++i)
	{
		idx = i;
		z_curr = idx / (X * Y);
		idx -= (z_curr * X * Y);
		y_curr = idx / X;
		x_curr = idx % X;

		if (x_curr == G_X && y_curr == G_Y && y_curr == G_Z)
		{
			A_s.push_back(0);
			A_sD.push_back(0);
			A.push_back(A_s);
			A_direction.push_back(A_sD);
			A_s.clear();
			A_sD.clear();
			continue;
		}
		// 0->x+1 y+0 East
		counter = 0;
		// while (counter < 2)
		//{
		for (int j = 0; j < 24; j++)
			cP[j] = uniform_prob_dist(e);
		if (x_curr + 1 < X && cP[1] > pi)
		{
			A_s.push_back(counter);
			A_sD.push_back(0);
			counter++;
		}
		// 1->x+0 y+1 NOrth

		if (y_curr + 1 < Y && cP[2] > pi)
		{
			A_s.push_back(counter);
			A_sD.push_back(1);
			counter++;
		}
		// 2->x-1 y+0 West
		if (x_curr - 1 > 0 && cP[3] > pi)
		{
			A_s.push_back(counter);
			A_sD.push_back(2);
			counter++;
		}
		// 3->x+0 y-1 South
		if (y_curr - 1 > 0 && cP[4] > pi)
		{
			A_s.push_back(counter);
			A_sD.push_back(3);
			counter++;
		}
		// 4->x+1 y+1 North East
		if (x_curr + 1 < X && y_curr + 1 < Y && cP[5] > pi)
		{
			A_s.push_back(counter);
			A_sD.push_back(4);
			counter++;
		}
		// 5->x-1 y-1 South West
		if (x_curr - 1 > 0 && y_curr - 1 > 0 && cP[6] > pi)
		{
			A_s.push_back(counter);
			A_sD.push_back(5);
			counter++;
		}
		// 6->x-1 y+1 North West
		if (x_curr - 1 > 0 && y_curr + 1 < Y && cP[7] > pi)
		{
			A_s.push_back(counter);
			A_sD.push_back(6);
			counter++;
		}
		// 7->x+1 y-1 South East
		if (x_curr + 1 < X && y_curr - 1 > 0 && cP[0] > pi)
		{
			A_s.push_back(counter);
			A_sD.push_back(7);
			counter++;
		}

		if (x_curr + 1 < X && z_curr + 1 < Z && cP[8] > pi)
		{
			A_s.push_back(counter);
			A_sD.push_back(8);
			counter++;
		}
		// 1->x+0 y+1 NOrth

		if (y_curr + 1 < Y && z_curr + 1 < Z && cP[9] > pi)
		{
			A_s.push_back(counter);
			A_sD.push_back(9);
			counter++;
		}
		// 2->x-1 y+0 West
		if (x_curr - 1 > 0 && z_curr + 1 < Z && cP[10] > pi)
		{
			A_s.push_back(counter);
			A_sD.push_back(10);
			counter++;
		}
		// 3->x+0 y-1 South
		if (y_curr - 1 > 0 && z_curr + 1 < Z && cP[11] > pi)
		{
			A_s.push_back(counter);
			A_sD.push_back(11);
			counter++;
		}
		// 4->x+1 y+1 North East
		if (x_curr + 1 < X && z_curr + 1 < Z && y_curr + 1 < Y && cP[12] > pi)
		{
			A_s.push_back(counter);
			A_sD.push_back(12);
			counter++;
		}
		// 5->x-1 y-1 South West
		if (x_curr - 1 > 0 && z_curr + 1 < Z && y_curr - 1 > 0 && cP[13] > pi)
		{
			A_s.push_back(counter);
			A_sD.push_back(13);
			counter++;
		}
		// 6->x-1 y+1 North West
		if (x_curr - 1 > 0 && z_curr + 1 < Z && y_curr + 1 < Y && cP[14] > pi)
		{
			A_s.push_back(counter);
			A_sD.push_back(14);
			counter++;
		}
		// 7->x+1 y-1 South East
		if (x_curr + 1 < X && z_curr + 1 < Z && y_curr - 1 > 0 && cP[15] > pi)
		{
			A_s.push_back(counter);
			A_sD.push_back(15);
			counter++;
		}
		if (x_curr + 1 < X && z_curr - 1 > 0 && cP[16] > pi)
		{
			A_s.push_back(counter);
			A_sD.push_back(16);
			counter++;
		}
		// 1->x+0 y+1 NOrth

		if (y_curr + 1 < Y && z_curr - 1 > 0 && cP[17] > pi)
		{
			A_s.push_back(counter);
			A_sD.push_back(17);
			counter++;
		}
		// 2->x-1 y+0 West
		if (x_curr - 1 > 0 && z_curr - 1 > 0 && cP[18] > pi)
		{
			A_s.push_back(counter);
			A_sD.push_back(18);
			counter++;
		}
		// 3->x+0 y-1 South
		if (y_curr - 1 > 0 && z_curr - 1 > 0 && cP[19] > pi)
		{
			A_s.push_back(counter);
			A_sD.push_back(19);
			counter++;
		}
		// 4->x+1 y+1 North East
		if (x_curr + 1 < X && z_curr - 1 > 0 && y_curr + 1 < Y && cP[20] > pi)
		{
			A_s.push_back(counter);
			A_sD.push_back(20);
			counter++;
		}
		// 5->x-1 y-1 South West
		if (x_curr - 1 > 0 && z_curr - 1 > 0 && y_curr - 1 > 0 && cP[21] > pi)
		{
			A_s.push_back(counter);
			A_sD.push_back(21);
			counter++;
		}
		// 6->x-1 y+1 North West
		if (x_curr - 1 > 0 && z_curr - 1 > 0 && y_curr + 1 < Y && cP[22] > pi)
		{
			A_s.push_back(counter);
			A_sD.push_back(22);
			counter++;
		}
		// 7->x+1 y-1 South East
		if (x_curr + 1 < X && z_curr - 1 > 0 && y_curr - 1 > 0 && cP[23] > pi)
		{
			A_s.push_back(counter);
			A_sD.push_back(23);
			counter++;
		}
		//}

		A.push_back(A_s);
		A_direction.push_back(A_sD);
		A_s.clear();
		A_sD.clear();
	}

	A_s.push_back(0);
	A_sD.push_back(0);
	A.push_back(A_s);
	A_direction.push_back(A_sD);

	/*float pit1 = 0.3;
	float pit2 = 0.5;
	float pit3 = 0.5;
	float pit4 = 0.5;
	float pit5 = 0.70;
	float pit6 = 0.70;
	float pit7 = 0.9;
	float pit8 = 0.9;
	float pit9 = 0.9;
	float pit10 = 1;*/

	float pit1 = 0.1;
	float pit2 = 0.20;
	float pit3 = 0.30;
	float pit4 = 0.40;
	float pit5 = 0.5;
	float pit6 = 0.60;
	float pit7 = 0.7;
	float pit8 = 0.8;
	float pit9 = 0.9;
	float pit10 = 1;
	// Create R
	R_type R;
	int metrw = 0;
	int metrw1 = 0;
	for (int i = 0; i < S; ++i)
	{
		idx = i;
		vector<double> R_s;
		z_curr = idx / (X * Y);
		idx -= (z_curr * X * Y);
		y_curr = idx / X;
		x_curr = idx % X;
		if (x_curr == G_X && y_curr == G_Y && z_curr == G_Z)
		{
			R_s.push_back(0);
		}
		else
		{
			/*if (uniform_prob_dist(e)<0.05){
				for (auto a : A[i]) {
					R_s.push_back(-X);
					metrw1++;
				}
				}
				else{
					for (auto a : A[i]) {
					R_s.push_back(-1);
					metrw++;
				}
				}
			*/
			for (auto a : A[i])
			{
				if (uniform_prob_dist(e) < pit1)
				{
					R_s.push_back(-1);
					metrw1++;
				}
				else if (uniform_prob_dist(e) < pit2)
				{
					R_s.push_back(-1.5);
					metrw++;
				}
				else if (uniform_prob_dist(e) < pit3)
				{
					R_s.push_back(-2);
					metrw++;
				}
				else if (uniform_prob_dist(e) < pit4)
				{
					R_s.push_back(-2.5);
					metrw++;
				}
				else if (uniform_prob_dist(e) < pit5)
				{
					R_s.push_back(-3);
					metrw++;
				}
				else if (uniform_prob_dist(e) < pit6)
				{
					R_s.push_back(-3.5);
					metrw++;
				}
				else if (uniform_prob_dist(e) < pit7)
				{
					R_s.push_back(-4);
					metrw++;
				}
				else if (uniform_prob_dist(e) < pit8)
				{
					R_s.push_back(-4.5);
					metrw++;
				}
				else if (uniform_prob_dist(e) < pit9)
				{
					R_s.push_back(-5);
					metrw++;
				}
				else
				{
					R_s.push_back(-10);
					metrw++;
				}
			}
		}

		R.push_back(R_s);
	}
	// cout<<"traps"<<metrw<<endl;
	// cout<<"normal"<<metrw1<<endl;
	vector<double> R_s;
	R_s.push_back(0);
	R.push_back(R_s);

	P_type P;
	vector<pair<vector<double>, vector<int>>> P_s;
	vector<int> P_s_a_nonzero_states;
	// vector<double> P_s_a(4, double(0));
	vector<double> P_s_a;

	// 0->x+1 y+0 East
	// 1->x+0 y+1 NOrth
	// 2->x-1 y+0 West
	// 3->x+0 y-1 South
	// 4->x+1 y+1 North East
	// 5->x-1 y-1 South West
	// 6->x-1 y+1 North West
	// 7->x+1 y-1 South East
	for (int i = 0; i < S; ++i)
	{
		// we now fix state s
		idx = i;
		z_curr = idx / (X * Y);
		idx -= (z_curr * X * Y);
		y_curr = idx / X;
		x_curr = idx % X;
		if (x_curr == G_X && y_curr == G_Y && z_curr == G_Z)
		{
			P_s_a_nonzero_states.push_back(S);
			P_s_a.push_back(1);
			P_s.push_back(make_pair(P_s_a, P_s_a_nonzero_states));
			P.push_back(P_s);
			P_s_a_nonzero_states.clear();
			P_s_a.clear();
			P_s.clear();
		}
		else
		{
			for (auto a : A[i])
			{
				P_s_a_nonzero_states.push_back(posDi3D(x_curr, y_curr, z_curr, X, Y, A_direction[i][a]));
				P_s_a.push_back(0.8);
				if (A[i].size() > 10)
				{
					int cc = 0;
					int pos[10] = {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1};
					while (cc < 10)
					{
						int ra = rand() % A[i].size();
						if (ra != a && check(pos, ra, 10))
						{
							pos[cc] = ra;
							cc = cc + 1;
							P_s_a_nonzero_states.push_back(posDi3D(x_curr, y_curr, z_curr, X, Y, A_direction[i][ra]));
							P_s_a.push_back(0.02);
						}
					}
				}
				else if (A[i].size() > 5)
				{
					int cc = 0;
					int pos[5] = {-1, -1, -1, -1, -1};
					while (cc < 5)
					{
						int ra = rand() % A[i].size();
						if (ra != a && check(pos, ra, 5))
						{
							pos[cc] = ra;
							cc = cc + 1;
							P_s_a_nonzero_states.push_back(posDi3D(x_curr, y_curr, z_curr, X, Y, A_direction[i][ra]));
							P_s_a.push_back(0.04);
						}
					}
				}
				else if (A[i].size() == 5)
				{
					int cc = 0;
					int pos[4] = {-1, -1, -1, -1};
					while (cc < 4)
					{
						int ra = rand() % A[i].size();
						if (ra != a && check(pos, ra, 4))
						{
							pos[cc] = ra;
							cc = cc + 1;
							P_s_a_nonzero_states.push_back(posDi3D(x_curr, y_curr, z_curr, X, Y, A_direction[i][ra]));
							P_s_a.push_back(0.05);
						}
					}
				}
				else if (A[i].size() == 4)
				{
					int cc = 0;
					int pos[3] = {-1, -1, -1};
					while (cc < 3)
					{
						int ra = rand() % A[i].size();
						if (ra != a && check(pos, ra, 3))
						{
							pos[cc] = ra;
							cc = cc + 1;
							P_s_a_nonzero_states.push_back(posDi3D(x_curr, y_curr, z_curr, X, Y, A_direction[i][ra]));
							if (cc == 1)
								P_s_a.push_back(0.1);
							else
								P_s_a.push_back(0.05);
						}
					}
				}
				else if (A[i].size() == 3)
				{
					int cc = 0;
					int pos[2] = {-1, -1};
					while (cc < 2)
					{
						int ra = rand() % A[i].size();
						if (ra != a && check(pos, ra, 2))
						{
							pos[cc] = ra;
							cc = cc + 1;
							P_s_a_nonzero_states.push_back(posDi3D(x_curr, y_curr, z_curr, X, Y, A_direction[i][ra]));
							P_s_a.push_back(0.1);
						}
					}
				}
				else if (A[i].size() == 2)
				{
					int cc = 0;
					while (cc < 1)
					{
						int ra = rand() % A[i].size();
						if (ra != a)
						{
							cc = cc + 1;
							P_s_a_nonzero_states.push_back(posDi3D(x_curr, y_curr, z_curr, X, Y, A_direction[i][ra]));
							P_s_a.push_back(0.2);
						}
					}
				}
				P_s.push_back(make_pair(P_s_a, P_s_a_nonzero_states));
				P_s_a.clear();
				P_s_a_nonzero_states.clear();
			}
			P.push_back(P_s);
			P_s_a.clear();
			P_s.clear();
		}
	}
	P_s_a_nonzero_states.push_back(S);
	P_s_a.push_back(1);
	P_s.push_back(make_pair(P_s_a, P_s_a_nonzero_states));
	P.push_back(P_s);
	// cout<<"hiend"<<endl;
	MDP_type MDP = make_tuple(R, A, P);

	return MDP;
}