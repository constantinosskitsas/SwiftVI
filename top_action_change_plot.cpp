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

// returns true if the first argument is less than the second.
bool cmp_action_value_pairs_top_action(const q_action_pair_type &a, q_action_pair_type &b)
{
	return a.first < b.first;
}

tuple<int, vector<int>> top_action_change(S_type S, R_type R, A_type A, P_type P, double gamma, double epsilon)
{

	// Find the maximum reward in the reward table
	double R_max = find_max_R(R);

	// TODO if the arrays has A_max size, then a = A_max hs no entry as 0 is an action. One fix is to make it 1 bigger to have space for this index
	int A_max = find_max_A(A) + 1;

	// Generate a size S vector with default 0 value
	// make this one 1 to make the difference large in first iteration i = 1 and then change these values withoput using them
	double V[2][S];
	fill(V[0], V[0] + S, R_max / (1.0 - gamma));
	fill(V[1], V[1] + S, 0);

	// HEAP INITIALIZATION
	q_action_pair_type s_heaps[S][A_max];

	// TODO: Set all heap sizes to 0 at a start
	int heap_size[S];

	for (int s = 0; s < S; s++)
	{
		// Put the initial q(s,a) elements into the heap
		// fill each one with the maximum value of each action
		// vector<q_action_pair_type> s_h(A[s].size(),(R_max / (1 - gamma)));
		q_action_pair_type *s_h = s_heaps[s];

		for (int a_index = 0; a_index < A[s].size(); a_index++)
		{
			// get the action of the index
			int a = A[s][a_index];

			auto &[P_s_a, P_s_a_nonzero] = P[s][a];
			// use the even iteration, as this is the one used in the i = 1 iteration, that we want to pre-do
			double q_1_s_a = R[s][a] + gamma * sum_of_mult_nonzero_only(P_s_a, V[0], P_s_a_nonzero);
			q_action_pair_type q_a_pair = make_pair(q_1_s_a, a);
			s_h[a_index] = q_a_pair;
		}

		// set the heap size
		heap_size[s] = A[s].size();

		// make it a heap for this state s
		make_heap(s_h, s_h + heap_size[s], cmp_action_value_pairs_top_action);
	}

	// keep count of number of iterations
	int iterations = 0;

	// NEW
	// record top action changes per iteration and in total
	vector<int> top_action_changes_per_iteration(1);
	int total_top_action_changes = 0;

	while (abs_max_diff(V[0], V[1], S) > ((epsilon * (1.0 - gamma)) / gamma))
	{

		// Increment iteration counter i
		iterations++;

		// NEW
		// count number of top action changes over all states in current iteration
		int top_action_changes_current_iteration_over_all_states = 0;

		// If i is even, then (i & 1) is 0, and the one to change is V[0]
		double *V_current_iteration = V[(iterations & 1)];
		double *V_previous_iteration = V[1 - (iterations & 1)];

		// for all states in each iteration
		for (int s = 0; s < S; s++)
		{
			// TODO if non-negative rewards, 0 is a lower bound of the maximization. to be changed if we want negative rewards
			q_action_pair_type *s_h = s_heaps[s];

			// update the top value
			int top_action = s_h[0].second;

			// the updated top value
			auto &[P_s_top_action, P_s_top_action_nonzero] = P[s][top_action];
			double updated_top_action_value = R[s][top_action] + gamma * sum_of_mult_nonzero_only(P_s_top_action, V_previous_iteration, P_s_top_action_nonzero);

			// The updated pair
			q_action_pair_type updated_pair = make_pair(updated_top_action_value, top_action);

			// now, we update the value
			pop_heap(s_h, s_h + heap_size[s], cmp_action_value_pairs_top_action);

			// can set the last element of an vector with v.back()
			// TODO check that -1 is correct
			s_h[heap_size[s] - 1] = updated_pair;

			// push the last element into the heap and keep the heap property
			push_heap(s_h, s_h + heap_size[s], cmp_action_value_pairs_top_action);

			// the new top action
			int updated_top_action = s_h[0].second;

			while (top_action != updated_top_action)
			{

				// NEW
				// add one to the number of action changes in current iteration over all states
				top_action_changes_current_iteration_over_all_states++;

				// update the top value
				top_action = s_h[0].second;

				// the updated top value
				auto &[P_s_top_action, P_s_top_action_nonzero] = P[s][top_action];
				updated_top_action_value = R[s][top_action] + gamma * sum_of_mult_nonzero_only(P_s_top_action, V_previous_iteration, P_s_top_action_nonzero);
				// The updated pair
				q_action_pair_type updated_pair = make_pair(updated_top_action_value, top_action);

				// now, we update the value
				pop_heap(s_h, s_h + heap_size[s], cmp_action_value_pairs_top_action);

				// can set the last element of an vector with v.back()
				// TODO check that -1 is correct
				s_h[heap_size[s] - 1] = updated_pair;

				// push the last element into the heap and keep the heap property
				push_heap(s_h, s_h + heap_size[s], cmp_action_value_pairs_top_action);

				// the new top action
				updated_top_action = s_h[0].second;
			}
			V_current_iteration[s] = s_h[0].first;
		}

		// NEW
		// record the number of new top action changes in accumulation variable
		top_action_changes_per_iteration.push_back(top_action_changes_current_iteration_over_all_states);
		total_top_action_changes += top_action_changes_current_iteration_over_all_states;
	}
	tuple<int, vector<int>> result_tuple = make_tuple(total_top_action_changes, top_action_changes_per_iteration);
	return result_tuple;
}
