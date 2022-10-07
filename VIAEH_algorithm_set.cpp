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

#include <set>
#include <iterator>

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

V_type value_iteration_action_elimination_heaps_set(S_type S, R_type R, A_type A, P_type P, double gamma, double epsilon)
{

	// TODO if the arrays has A_max size, then a = A_max hs no entry as 0 is an action. One fix is to make it 1 bigger to have space for this index
	// finds out how big a vector we need to store an action in an entry
	int A_max = find_max_A(A) + 1;

	// Find the maximum reward in the reward table
	auto [r_star_min, r_star_max, r_star_values] = find_all_r_values(R);

	// 1. Improved Upper Bound
	double **V_U = new double *[2];
	for (int i = 0; i < 2; ++i)
	{
		V_U[i] = new double[S];
	}
	for (int s = 0; s < S; s++)
	{
		V_U[0][s] = (gamma / (1.0 - gamma)) * r_star_max + r_star_values[s];
		V_U[1][s] = 1.0;
	}

	// 2. Improved Lower Bound
	double **V_L = new double *[2];
	for (int i = 0; i < 2; ++i)
	{
		V_L[i] = new double[S];
	}

	for (int s = 0; s < S; s++)
	{
		V_L[0][s] = (gamma / (1.0 - gamma)) * r_star_min + r_star_values[s];
		V_L[1][s] = 1.0;
	}

	// keep track of work done in each iteration in microseconds
	// start from iteration 1, so put a 0 value into the first iteration
	vector<microseconds> work_per_iteration(1);

	// init criteria variables to know which value to return based on why the algorithm terminated
	// set to true if we have converged!
	bool bounded_convergence_criteria = false;
	bool upper_convergence_criteria = false;
	bool lower_convergence_criteria = false;

	// record actions eliminated in each iteration, where a pair is (state, action)
	// push empty vector for 0-index. Iterations start with 1
	vector<vector<pair<int, int>>> actions_eliminated;
	actions_eliminated.push_back(vector<pair<int, int>>());

	// pre-compute convergence criteria for efficiency to not do it in each iteration of while loop
	const double convergence_bound_precomputed = (epsilon * (1.0 - gamma)) / gamma;
	const double two_epsilon = 2 * epsilon;

	// SET DATA STRUCTURE INITIALIZATION
	// set<q_action_pair_type> *action_sets[S];
	set<q_action_pair_type> **action_sets = new set<q_action_pair_type> *[S];

	// SET INITIALIZATION OF ACTION SETS
	for (int s = 0; s < S; s++)
	{
		action_sets[s] = new set<q_action_pair_type>();
	}

	//**********************************************************************
	// FILL THE HEAPS AS A PREPROCESSING STEP BEFORE ITERATIONS BEGIN

	for (int s = 0; s < S; s++)
	{

		// get the action set
		set<q_action_pair_type> &action_set_s = (*action_sets[s]);

		// initialise the heaps with values from the first iteration
		for (int a : A[s])
		{

			// need the distribution to calculate first q-values
			auto &[P_s_a, P_s_a_nonzero] = P[s][a];

			// use the even iteration, as this is the one used in the i = 1 iteration, that we want to pre-do
			double q_1_U_s_a = V_U[0][s]; // R[s][a] + gamma * sum_of_mult_nonzero_only(P_s_a, V_U[0], P_s_a_nonzero);

			// create the elements for the priority_queues
			// OBS: multiply min element q_value with -1
			q_action_pair_type q_a_tuple = make_pair(q_1_U_s_a, a);

			// insert the tuples into the heaps
			action_set_s.insert(q_a_tuple);
		}
	}

	//**********************************************************************
	// ACTUAL ITERATIVE VI EFFICIENT ALGORITHM

	// keep count of number of iterations
	int iterations = 0;

	// while any of the criteria are NOT, !, met, run the loop
	// while NEITHER has converged
	while ((!bounded_convergence_criteria) && (!upper_convergence_criteria) && (!lower_convergence_criteria))
	{

		// Increment iteration counter i
		iterations++;

		// Record actions eliminated in this iteration over all states
		vector<pair<int, int>> actions_eliminated_in_iteration;

		// begin timing of this iteration
		auto start_of_iteration = high_resolution_clock::now();

		// If i is even, then (i & 1) is 0, and the one to change is V[0]
		double *V_U_current_iteration = V_U[(iterations & 1)];
		double *V_U_previous_iteration = V_U[1 - (iterations & 1)];

		double *V_L_current_iteration = V_L[(iterations & 1)];
		double *V_L_previous_iteration = V_L[1 - (iterations & 1)];

		// for all states in each iteration
		for (int s = 0; s < S; s++)
		{

			// TODO if non-negative rewards, 0 is a lower bound of the maximization. to be changed if we want negative rewards
			V_U_current_iteration[s] = numeric_limits<double>::min();
			V_L_current_iteration[s] = numeric_limits<double>::min();

			// get the set data structure
			set<q_action_pair_type> &action_set_s = (*action_sets[s]);

			// ranged for loop over all actions in the action set of state s
			// This is changed to use the max_heap as an array of actions in A[s]
			for (auto &q_a_pair : action_set_s)
			{

				int a = q_a_pair.second;

				// reference to the probability vector of size S
				auto &[P_s_a, P_s_a_nonzero] = P[s][a];

				double Q_L_s_a = R[s][a] + gamma * sum_of_mult_nonzero_only(P_s_a, V_L_previous_iteration, P_s_a_nonzero);

				if (Q_L_s_a > V_L_current_iteration[s])
				{
					V_L_current_iteration[s] = Q_L_s_a;
				}
			}

			// FIND MAX
			bool not_same_action_both_times = true;

			while (not_same_action_both_times)
			{
				// change to -> notation
				int top_pair_action = (*(action_set_s.rbegin())).second;

				// calculate updated value of current top action
				auto &[P_s_top_pair, P_s_top_pair_nonzero] = P[s][top_pair_action];
				double q_i_s_a = R[s][top_pair_action] + gamma * sum_of_mult_nonzero_only(P_s_top_pair, V_U_previous_iteration, P_s_top_pair_nonzero);

				// erase top (end) element and insert new value (with hint that it is near the endperhaps)
				action_set_s.erase(--action_set_s.end());

				q_action_pair_type new_top_action_pair_max = make_pair(q_i_s_a, top_pair_action);

				// try to insert with the hint being that it is still the max element
				action_set_s.insert(action_set_s.end(), new_top_action_pair_max);

				// record possible new max action
				int updated_heap_max_top_action = (*(action_set_s.rbegin())).second;

				// Check if max action changed
				not_same_action_both_times = (top_pair_action != updated_heap_max_top_action);
			}

			// RECORD MAX VALUE BASED ON THE ABOVE ACTION UPDATED
			V_U_current_iteration[s] = (*(action_set_s.rbegin())).first;

			// WE NOW START THE ACTION ELIMINATION PROCESS
			int top_pair_action_min = (*(action_set_s.begin())).second;

			auto &[P_s_action_min, P_s_action_min_nonzero] = P[s][top_pair_action_min];
			double q_i_s_a_min = R[s][top_pair_action_min] + gamma * sum_of_mult_nonzero_only(P_s_action_min, V_U_previous_iteration, P_s_action_min_nonzero);

			q_action_pair_type new_top_action_pair_min = make_pair(q_i_s_a_min, top_pair_action_min);

			// ERASE OLD VALUE AND INSERT NEW MIN VALUE
			action_set_s.erase(action_set_s.begin());
			action_set_s.insert(action_set_s.begin(), new_top_action_pair_min);

			double updated_min_upper_bound = (*(action_set_s.begin())).first;

			// UPDATE TOP VALUE AND ELIMINATE UNTIL NOT POSSIBLE ANYMORE
			while (updated_min_upper_bound < V_L_current_iteration[s])
			{

				// record action eliminated to return
				int removableAction = (*(action_set_s.begin())).second;
				actions_eliminated_in_iteration.emplace_back(s, removableAction);

				// REMOVE THE TOP MIN ACTION
				action_set_s.erase(action_set_s.begin());

				// WHAT IS NOW THE MIN ACTION
				top_pair_action_min = (*(action_set_s.begin())).second;

				// FIND UPDATED VALUE FOR NEW MIN ACTION TO MAXIMIZE CHANGE OF ELIMINATABLE VALUE
				auto &[P_s_action_min, P_s_action_min_nonzero] = P[s][top_pair_action_min];
				q_i_s_a_min = R[s][top_pair_action_min] + gamma * sum_of_mult_nonzero_only(P_s_action_min, V_U_previous_iteration, P_s_action_min_nonzero);

				q_action_pair_type new_top_action_pair_min = make_pair(q_i_s_a_min, top_pair_action_min);

				// INSERT NEW MIN VALUE
				action_set_s.erase(action_set_s.begin());
				action_set_s.insert(action_set_s.begin(), new_top_action_pair_min);

				updated_min_upper_bound = (*(action_set_s.begin())).first;
			}
		}

		// see if any of the convergence criteria are met
		// 1. bounded criteria
		bounded_convergence_criteria = abs_max_diff(V_U[(iterations & 1)], V_L[(iterations & 1)], S) <= two_epsilon;

		// 2. upper criteria
		upper_convergence_criteria = abs_max_diff(V_U[0], V_U[1], S) <= convergence_bound_precomputed;

		// 3. lower criteria
		lower_convergence_criteria = abs_max_diff(V_L[0], V_L[1], S) <= convergence_bound_precomputed;

		// end timing of this iteration and record it in work vector
		auto end_of_iteration = high_resolution_clock::now();
		auto duration_of_iteration = duration_cast<microseconds>(end_of_iteration - start_of_iteration);
		work_per_iteration.push_back(duration_of_iteration);
		actions_eliminated.push_back(move(actions_eliminated_in_iteration));
	}
	// case return value on which convergence criteria was met
	vector<double> result(S); // set it so have size S from beginning to use copy

	if (bounded_convergence_criteria)
	{
		result = V_upper_lower_average(V_U[(iterations & 1)], V_L[(iterations & 1)], S);
	}
	else if (upper_convergence_criteria)
	{
		copy(V_U[(iterations & 1)], V_U[(iterations & 1)] + S, result.begin());
	}
	else if (lower_convergence_criteria)
	{
		copy(V_L[(iterations & 1)], V_L[(iterations & 1)] + S, result.begin());
	}
	V_type result_tuple = make_tuple(result, iterations, work_per_iteration, actions_eliminated);

	// DEALLOCATE MEMORY
	for (int i = 0; i < 2; ++i)
	{
		delete[] V_U[i];
	}
	delete[] V_U;

	for (int i = 0; i < 2; ++i)
	{
		delete[] V_L[i];
	}
	delete[] V_L;

	for (int s = 0; s < S; s++)
	{
		// dormal delete operation as it is an object, not array pointer
		delete action_sets[s];
	}
	delete[] action_sets;

	return result_tuple;
}
