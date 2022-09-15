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

#include <queue>
#include <unordered_set>
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

typedef tuple<double, int, int> q_action_timestamp_tuple;
typedef priority_queue<q_action_timestamp_tuple> pq_type;

V_type value_iteration_action_elimination_heaps_lazy_update(S_type S, R_type R, A_type A, P_type P, double gamma, double epsilon){
		int A_max = find_max_A(A) + 1;
		
		//Find the maximum reward in the reward table
		auto [r_star_min, r_star_max, r_star_values] = find_all_r_values(R);

		//1. Improved Upper Bound
		double** V_U = new double*[2];
		for(int i = 0; i < 2; ++i) {
				V_U[i] = new double[S];
		}
		for(int s = 0; s < S; s++) {
				V_U[0][s] = (gamma / (1.0 - gamma)) * r_star_max + r_star_values[s];
				V_U[1][s] = 1.0;
		}

		//2. Improved Lower Bound
		double** V_L = new double*[2];
		for(int i = 0; i < 2; ++i) {
				V_L[i] = new double[S];
		}

		for(int s = 0; s < S; s++) {
				V_L[0][s] = (gamma / (1.0 - gamma)) * r_star_min + r_star_values[s];
				V_L[1][s] = 1.0;
		}
		
		//keep track of work done in each iteration in microseconds
		//start from iteration 1, so put a 0 value into the first iteration
		vector<microseconds> work_per_iteration(1);

		//init criteria variables to know which value to return based on why the algorithm terminated
		//set to true if we have converged!
		bool bounded_convergence_criteria = false;
		bool upper_convergence_criteria = false;
		bool lower_convergence_criteria = false;

		//pre-compute convergence criteria for efficiency to not do it in each iteration of while loop
		const double convergence_bound_precomputed = (epsilon * (1.0 - gamma)) / gamma;
		const double two_epsilon = 2 * epsilon;

		//record actions eliminated in each iteration, where a pair is (state, action)
		//push empty vector for 0-index. Iterations start with 1
		vector<vector<pair<int, int>>> actions_eliminated;
		actions_eliminated.push_back(vector<pair<int, int>>());
		
		//HEAP TIMESTAMPS AND IF ACTION ELIMINATED INITIALIZED
		//int heap_time_stamps[S][A_max];
		int** heap_time_stamps = new int*[S];
		for(int i = 0; i < S; ++i) {
				heap_time_stamps[i] = new int[A_max];
		}
	
		//For testing purposes, fill each entry with -1, so I can see the ones not in the set
		for (int s = 0; s < S; s++){
				fill(heap_time_stamps[s], heap_time_stamps[s] + A_max, 0);
		}

		//PRIORITY QUEUE INITIALIZATION

		//list of pointers to priority queues
		//pq_type *max_priority_queues[S];
		//pq_type *min_priority_queues[S];
		pq_type** max_priority_queues = new pq_type*[S];
		pq_type** min_priority_queues = new pq_type*[S];
		

		//UNORDERED SET INITIALIZATION OF ACTION SETS
		//unordered_set<int> *action_sets[S];
		unordered_set<int>** action_sets = new unordered_set<int>*[S];

		for (int s = 0; s < S; s++){
				max_priority_queues[s] = new pq_type();
				min_priority_queues[s] = new pq_type();
				action_sets[s] = new unordered_set<int>();
		}

		//**********************************************************************
		//FILL THE HEAPS AS A PREPROCESSING STEP BEFORE ITERATIONS BEGIN
		
		for (int s = 0; s < S; s++){

				//The heaps of current state s
				//VERY IMPORTANT to get them as refernces, &, otherwise, changed do not persist
				pq_type &max_pq_s = (*max_priority_queues[s]);
				pq_type &min_pq_s = (*min_priority_queues[s]);

				//get the action set
				unordered_set<int> &action_set_s = (*action_sets[s]);

				//initialise the heaps with values from the first iteration
				for (int a : A[s]){

						//need the distribution to calculate first q-values
						auto& [P_s_a, P_s_a_nonzero] = P[s][a];
						
						//use the even iteration, as this is the one used in the i = 1 iteration, that we want to pre-do
						double q_1_U_s_a = R[s][a] + gamma * sum_of_mult_nonzero_only(P_s_a, V_U[0], P_s_a_nonzero);

						//create the elements for the priority_queues
						//OBS: multiply min element q_value with -1
						q_action_timestamp_tuple q_a_tuple_max = make_tuple(q_1_U_s_a, a, 0);
						q_action_timestamp_tuple q_a_tuple_min = make_tuple((double(-1)) * q_1_U_s_a, a, 0);

						//insert the tuples into the heaps
						max_pq_s.push(q_a_tuple_max);
						min_pq_s.push(q_a_tuple_min);

						//add action to unordered action set
						//TODO: perhaps insert whole A[s] in one take
						action_set_s.insert(a);
				}
		}

		//**********************************************************************
		//ACTUAL ITERATIVE VI EFFICIENT ALGORITHM

		//keep count of number of iterations
		int iterations = 0;

		//while any of the criteria are NOT, !, met, run the loop
		//while NEITHER has converged
		while ( (!bounded_convergence_criteria) && (!upper_convergence_criteria) && (!lower_convergence_criteria) ){

				//Increment iteration counter i
				iterations++;	
				
				//Record actions eliminated in this iteration over all states
				vector<pair<int, int>> actions_eliminated_in_iteration;

				//begin timing of this iteration
				auto start_of_iteration = high_resolution_clock::now();
				
				//If i is even, then (i & 1) is 0, and the one to change is V[0]
				double *V_U_current_iteration = V_U[(iterations & 1)];
				double *V_U_previous_iteration = V_U[1 - (iterations & 1)];

				double *V_L_current_iteration = V_L[(iterations & 1)];
				double *V_L_previous_iteration = V_L[1 - (iterations & 1)];
				
				//for all states in each iteration
				for (int s = 0; s < S; s++) {

						//TODO if non-negative rewards, 0 is a lower bound of the maximization. to be changed if we want negative rewards
						V_U_current_iteration[s] = numeric_limits<double>::min();
						V_L_current_iteration[s] = numeric_limits<double>::min();

						//The heaps of current state s
						pq_type &max_pq_s = (*max_priority_queues[s]);
						pq_type &min_pq_s = (*min_priority_queues[s]);
					
						//get the action set
						unordered_set<int> &action_set_s = (*action_sets[s]);

						//get timestamps	
						int *timestamps_s = heap_time_stamps[s];

						//keep the rewards from the upper bound for 
						double Q_U_s[action_set_s.size()];

						//ranged for loop over all actions in the action set of state s
						//This is changed to use the max_heap as an array of actions in A[s]
						for (int a : action_set_s){

								//reference to the probability vector of size S
								auto& [P_s_a, P_s_a_nonzero] = P[s][a];
								
								double Q_L_s_a = R[s][a] + gamma * sum_of_mult_nonzero_only(P_s_a, V_L_previous_iteration, P_s_a_nonzero); 

								if (Q_L_s_a > V_L_current_iteration[s]) {
										V_L_current_iteration[s] = Q_L_s_a;
								}
						}

						//FIND MAX
						bool not_same_action_both_times = true;
						
						while (not_same_action_both_times){
								//START UPDATE update_top_action_max, write in seperate function in the future

								//check that it is a "real" entry (not outdated or already removed)
								//pop element until we have the most updated and not already eliminated
								while (timestamps_s[get<1>(max_pq_s.top())] != get<2>(max_pq_s.top())) {
										max_pq_s.pop();
								}
								int top_tuple_action = get<1>(max_pq_s.top());
									
								//calculate updated value of current top action
								auto& [P_s_top_tuple, P_s_top_tuple_nonzero] = P[s][top_tuple_action];
								double q_i_s_a = R[s][top_tuple_action] + gamma * sum_of_mult_nonzero_only(P_s_top_tuple, V_U_previous_iteration, P_s_top_tuple_nonzero); 

								//pop value and insert new value into prority_queuea with new timestamp
								max_pq_s.pop();
								q_action_timestamp_tuple new_top_action_tuple_max = make_tuple(q_i_s_a, top_tuple_action, iterations);
								max_pq_s.push(new_top_action_tuple_max);
								timestamps_s[top_tuple_action] = iterations;

								//insert value into min-heap - will be a duplicate
								q_action_timestamp_tuple new_top_action_tuple_min = make_tuple((double(-1)) * q_i_s_a, top_tuple_action, iterations);
								min_pq_s.push(new_top_action_tuple_min);

								//look at the updated max_action and see it has changed
								while (timestamps_s[get<1>(max_pq_s.top())] != get<2>(max_pq_s.top())) {
										max_pq_s.pop();
								}
								int updated_heap_max_top_action = get<1>(max_pq_s.top());
								not_same_action_both_times = (top_tuple_action != updated_heap_max_top_action);
						}
					
						//RECORD MAX VALUE BASED ON THE ABOVE ACTION UPDATED
						V_U_current_iteration[s] = get<0>(max_pq_s.top());

						//WE NOW START THE ACTION ELIMINATION PROCESS BASED ON THE TWO HEAPS
						//when we want to look at the top, we need to shuffle away the non-real elements
						while (timestamps_s[get<1>(min_pq_s.top())] == -1) {
								min_pq_s.pop();
						}
						int top_tuple_action_min = get<1>(min_pq_s.top());
						
						auto& [P_s_action_min, P_s_action_min_nonzero] = P[s][top_tuple_action_min];
						double q_i_s_a_min = R[s][top_tuple_action_min] + gamma * sum_of_mult_nonzero_only(P_s_action_min, V_U_previous_iteration, P_s_action_min_nonzero); 
					
						//INSERT NEW VALUE INTO BOTH PRIORITY QUEUES AND RECORD NEW TIMESTAMP
						min_pq_s.pop();
						q_action_timestamp_tuple new_top_action_tuple_min = make_tuple( (double(-1)) * q_i_s_a_min, top_tuple_action_min, iterations);
						min_pq_s.push(new_top_action_tuple_min);
					
						q_action_timestamp_tuple new_top_action_tuple_max = make_tuple(q_i_s_a_min, top_tuple_action_min, iterations);
						max_pq_s.push(new_top_action_tuple_max);
						timestamps_s[top_tuple_action_min] = iterations;
						
						//RECORD NEW MIN VALUE 
						//If the action is already eliminated, pop it (lazy update)
						while (timestamps_s[get<1>(min_pq_s.top())] == -1) {
								min_pq_s.pop();
						}
						double updated_min_upper_bound = double(-1) * get<0>(min_pq_s.top());
					
						//UPDATE TOP VALUE AND ELIMINATE UNTIL NOT POSSIBLE ANYMORE
						while(updated_min_upper_bound < V_L_current_iteration[s]){

								//REMOVE THE TOP MIN ACTION
								int removableAction = get<1>(min_pq_s.top());

								//record action eliminated to return
								actions_eliminated_in_iteration.emplace_back(s, removableAction);											
								
								//remove by setting timestamp to -1, works in both queues
								timestamps_s[removableAction] = -1;

								//also, delete from unordered_set
								action_set_s.erase(removableAction);

								//pop the element from the queue
								min_pq_s.pop();
								
								//UPDATE NEW TOP MIN VALUE
								while (timestamps_s[get<1>(min_pq_s.top())] == -1) {
										min_pq_s.pop();
								}
								top_tuple_action_min = get<1>(min_pq_s.top());
								
								auto& [P_s_action_min, P_s_action_min_nonzero] = P[s][top_tuple_action_min];
								q_i_s_a_min = R[s][top_tuple_action_min] + gamma * sum_of_mult_nonzero_only(P_s_action_min, V_U_previous_iteration, P_s_action_min_nonzero); 
							
								//INSERT NEW VALUE INTO BOTH PRIORITY QUEUES AND RECORD NEW TIMESTAMP
								min_pq_s.pop();
								new_top_action_tuple_min = make_tuple( (double(-1)) * q_i_s_a_min, top_tuple_action_min, iterations);
								min_pq_s.push(new_top_action_tuple_min);
							
								new_top_action_tuple_max = make_tuple(q_i_s_a_min, top_tuple_action_min, iterations);
								max_pq_s.push(new_top_action_tuple_max);
								timestamps_s[top_tuple_action_min] = iterations;
								
								//RECORD NEW MIN VALUE 
								while (timestamps_s[get<1>(min_pq_s.top())] == -1) {
										min_pq_s.pop();
								}
								updated_min_upper_bound = double(-1) * get<0>(min_pq_s.top());
						}
				}

				//see if any of the convergence criteria are met
				//1. bounded criteria
				bounded_convergence_criteria = abs_max_diff(V_U[(iterations & 1)], V_L[(iterations & 1)], S) <= two_epsilon;

				//2. upper criteria
				upper_convergence_criteria = abs_max_diff(V_U[0], V_U[1], S) <= convergence_bound_precomputed;

				//3. lower criteria
				lower_convergence_criteria = abs_max_diff(V_L[0], V_L[1], S) <= convergence_bound_precomputed;
				
				//end timing of this iteration and record it in work vector
				auto end_of_iteration = high_resolution_clock::now();
				auto duration_of_iteration = duration_cast<microseconds>(end_of_iteration - start_of_iteration);
				work_per_iteration.push_back(duration_of_iteration);
				actions_eliminated.push_back(move(actions_eliminated_in_iteration));

		}
		//case return value on which convergence criteria was met
		vector<double> result(S); //set it so have size S from beginning to use copy
		
		if (bounded_convergence_criteria) {
				result = V_upper_lower_average(V_U[(iterations & 1)], V_L[(iterations & 1)], S);
		} else if (upper_convergence_criteria) {
				copy(V_U[(iterations & 1)], V_U[(iterations & 1)] + S, result.begin());
		} else if (lower_convergence_criteria) {
				copy(V_L[(iterations & 1)], V_L[(iterations & 1)] + S, result.begin());
		}
		V_type result_tuple = make_tuple(result, iterations, work_per_iteration, actions_eliminated);

		//SEG FAULT HERE, NOT WHEN I DELETE THIS
		for(int i = 0; i < 2; ++i) {
				delete [] V_U[i];
		}
		delete [] V_U;

		for(int i = 0; i < 2; ++i) {
				delete [] V_L[i];
		}
		delete [] V_L;

		for(int i = 0; i < S; ++i) {
				delete [] heap_time_stamps[i];
		}
		delete [] heap_time_stamps; 

		for (int s = 0; s < S; ++s) {
				//these are all with no brackets, as they are object deletions, not array pointers
				delete max_priority_queues[s];
				delete min_priority_queues[s];
				delete action_sets[s];
		}
		delete [] max_priority_queues;
		delete [] min_priority_queues;
		delete [] action_sets;

		return result_tuple;
}



V_type value_iteration_action_elimination_heaps_lazy_updateGS(S_type S, R_type R, A_type A, P_type P, double gamma, double epsilon){
		int A_max = find_max_A(A) + 1;
		
		//Find the maximum reward in the reward table
		auto [r_star_min, r_star_max, r_star_values] = find_all_r_values(R);

		//1. Improved Upper Bound
		double** V_U = new double*[1];
		for(int i = 0; i < 1; ++i) {
				V_U[i] = new double[S];
		}
		for(int s = 0; s < S; s++) {
				V_U[0][s] = (gamma / (1.0 - gamma)) * r_star_max + r_star_values[s];
		}

		//2. Improved Lower Bound
		double** V_L = new double*[1];
		for(int i = 0; i < 1; ++i) {
				V_L[i] = new double[S];
		}

		for(int s = 0; s < S; s++) {
				V_L[0][s] = (gamma / (1.0 - gamma)) * r_star_min + r_star_values[s];
		}
		
		//keep track of work done in each iteration in microseconds
		//start from iteration 1, so put a 0 value into the first iteration
		vector<microseconds> work_per_iteration(1);

		//init criteria variables to know which value to return based on why the algorithm terminated
		//set to true if we have converged!
		bool bounded_convergence_criteria = false;
		bool upper_convergence_criteria = false;
		bool lower_convergence_criteria = false;

		//pre-compute convergence criteria for efficiency to not do it in each iteration of while loop
		const double convergence_bound_precomputed = (epsilon * (1.0 - gamma)) / gamma;
		const double two_epsilon = 2 * epsilon;

		//record actions eliminated in each iteration, where a pair is (state, action)
		//push empty vector for 0-index. Iterations start with 1
		vector<vector<pair<int, int>>> actions_eliminated;
		actions_eliminated.push_back(vector<pair<int, int>>());
		
		//HEAP TIMESTAMPS AND IF ACTION ELIMINATED INITIALIZED
		//int heap_time_stamps[S][A_max];
		int** heap_time_stamps = new int*[S];
		for(int i = 0; i < S; ++i) {
				heap_time_stamps[i] = new int[A_max];
		}
	
		//For testing purposes, fill each entry with -1, so I can see the ones not in the set
		for (int s = 0; s < S; s++){
				fill(heap_time_stamps[s], heap_time_stamps[s] + A_max, 0);
		}

		//PRIORITY QUEUE INITIALIZATION

		//list of pointers to priority queues
		//pq_type *max_priority_queues[S];
		//pq_type *min_priority_queues[S];
		pq_type** max_priority_queues = new pq_type*[S];
		pq_type** min_priority_queues = new pq_type*[S];
		

		//UNORDERED SET INITIALIZATION OF ACTION SETS
		//unordered_set<int> *action_sets[S];
		unordered_set<int>** action_sets = new unordered_set<int>*[S];

		for (int s = 0; s < S; s++){
				max_priority_queues[s] = new pq_type();
				min_priority_queues[s] = new pq_type();
				action_sets[s] = new unordered_set<int>();
		}

		//**********************************************************************
		//FILL THE HEAPS AS A PREPROCESSING STEP BEFORE ITERATIONS BEGIN
		
		for (int s = 0; s < S; s++){

				//The heaps of current state s
				//VERY IMPORTANT to get them as refernces, &, otherwise, changed do not persist
				pq_type &max_pq_s = (*max_priority_queues[s]);
				pq_type &min_pq_s = (*min_priority_queues[s]);

				//get the action set
				unordered_set<int> &action_set_s = (*action_sets[s]);

				//initialise the heaps with values from the first iteration
				for (int a : A[s]){

						//need the distribution to calculate first q-values
						auto& [P_s_a, P_s_a_nonzero] = P[s][a];
						
						//use the even iteration, as this is the one used in the i = 1 iteration, that we want to pre-do
						double q_1_U_s_a = R[s][a] + gamma * sum_of_mult_nonzero_only(P_s_a, V_U[0], P_s_a_nonzero);

						//create the elements for the priority_queues
						//OBS: multiply min element q_value with -1
						q_action_timestamp_tuple q_a_tuple_max = make_tuple(q_1_U_s_a, a, 0);
						q_action_timestamp_tuple q_a_tuple_min = make_tuple((double(-1)) * q_1_U_s_a, a, 0);

						//insert the tuples into the heaps
						max_pq_s.push(q_a_tuple_max);
						min_pq_s.push(q_a_tuple_min);

						//add action to unordered action set
						//TODO: perhaps insert whole A[s] in one take
						action_set_s.insert(a);
				}
		}

		//**********************************************************************
		//ACTUAL ITERATIVE VI EFFICIENT ALGORITHM

		//keep count of number of iterations
		int iterations = 0;
		double *V_U_current_iteration = V_U[0];
		double *V_L_current_iteration = V_L[0];
		//while any of the criteria are NOT, !, met, run the loop
		//while NEITHER has converged
		while ( (!bounded_convergence_criteria) && (!upper_convergence_criteria) && (!lower_convergence_criteria) ){
				 bounded_convergence_criteria = true;
				upper_convergence_criteria = true;
				lower_convergence_criteria = true;
				//Increment iteration counter i
				iterations++;	
				
				//Record actions eliminated in this iteration over all states
				vector<pair<int, int>> actions_eliminated_in_iteration;

				//begin timing of this iteration
				auto start_of_iteration = high_resolution_clock::now();
				
				//If i is even, then (i & 1) is 0, and the one to change is V[0]
				
				//for all states in each iteration
				for (int s = 0; s < S; s++) {

						//TODO if non-negative rewards, 0 is a lower bound of the maximization. to be changed if we want negative rewards

						//The heaps of current state s
						pq_type &max_pq_s = (*max_priority_queues[s]);
						pq_type &min_pq_s = (*min_priority_queues[s]);
					
						//get the action set
						unordered_set<int> &action_set_s = (*action_sets[s]);

						//get timestamps	
						int *timestamps_s = heap_time_stamps[s];

						//keep the rewards from the upper bound for 
						double Q_U_s[action_set_s.size()];
						double oldVU=V_U_current_iteration[s];
						double oldVL=V_L_current_iteration[s];
						double Q_max=numeric_limits<double>::min();
						//ranged for loop over all actions in the action set of state s
						//This is changed to use the max_heap as an array of actions in A[s]
						for (int a : action_set_s){

								//reference to the probability vector of size S
								auto& [P_s_a, P_s_a_nonzero] = P[s][a];
								
								double Q_L_s_a = R[s][a] + gamma * sum_of_mult_nonzero_only(P_s_a, V_L_current_iteration, P_s_a_nonzero); 

								if (Q_L_s_a > V_L_current_iteration[s]) {
										V_L_current_iteration[s] = Q_L_s_a;
								}
						}

						//FIND MAX
						bool not_same_action_both_times = true;
						
						while (not_same_action_both_times){
								//START UPDATE update_top_action_max, write in seperate function in the future

								//check that it is a "real" entry (not outdated or already removed)
								//pop element until we have the most updated and not already eliminated
								while (timestamps_s[get<1>(max_pq_s.top())] != get<2>(max_pq_s.top())) {
										max_pq_s.pop();
								}
								int top_tuple_action = get<1>(max_pq_s.top());
									
								//calculate updated value of current top action
								auto& [P_s_top_tuple, P_s_top_tuple_nonzero] = P[s][top_tuple_action];
								double q_i_s_a = R[s][top_tuple_action] + gamma * sum_of_mult_nonzero_only(P_s_top_tuple, V_U_current_iteration, P_s_top_tuple_nonzero); 

								//pop value and insert new value into prority_queuea with new timestamp
								max_pq_s.pop();
								q_action_timestamp_tuple new_top_action_tuple_max = make_tuple(q_i_s_a, top_tuple_action, iterations);
								max_pq_s.push(new_top_action_tuple_max);
								timestamps_s[top_tuple_action] = iterations;

								//insert value into min-heap - will be a duplicate
								q_action_timestamp_tuple new_top_action_tuple_min = make_tuple((double(-1)) * q_i_s_a, top_tuple_action, iterations);
								min_pq_s.push(new_top_action_tuple_min);

								//look at the updated max_action and see it has changed
								while (timestamps_s[get<1>(max_pq_s.top())] != get<2>(max_pq_s.top())) {
										max_pq_s.pop();
								}
								int updated_heap_max_top_action = get<1>(max_pq_s.top());
								not_same_action_both_times = (top_tuple_action != updated_heap_max_top_action);
						}
					
						//RECORD MAX VALUE BASED ON THE ABOVE ACTION UPDATED
						V_U_current_iteration[s] = get<0>(max_pq_s.top());

						//WE NOW START THE ACTION ELIMINATION PROCESS BASED ON THE TWO HEAPS
						//when we want to look at the top, we need to shuffle away the non-real elements
						while (timestamps_s[get<1>(min_pq_s.top())] == -1) {
								min_pq_s.pop();
						}
						int top_tuple_action_min = get<1>(min_pq_s.top());
						
						auto& [P_s_action_min, P_s_action_min_nonzero] = P[s][top_tuple_action_min];
						double q_i_s_a_min = R[s][top_tuple_action_min] + gamma * sum_of_mult_nonzero_only(P_s_action_min, V_U_current_iteration, P_s_action_min_nonzero); 
					
						//INSERT NEW VALUE INTO BOTH PRIORITY QUEUES AND RECORD NEW TIMESTAMP
						min_pq_s.pop();
						q_action_timestamp_tuple new_top_action_tuple_min = make_tuple( (double(-1)) * q_i_s_a_min, top_tuple_action_min, iterations);
						min_pq_s.push(new_top_action_tuple_min);
					
						q_action_timestamp_tuple new_top_action_tuple_max = make_tuple(q_i_s_a_min, top_tuple_action_min, iterations);
						max_pq_s.push(new_top_action_tuple_max);
						timestamps_s[top_tuple_action_min] = iterations;
						
						//RECORD NEW MIN VALUE 
						//If the action is already eliminated, pop it (lazy update)
						while (timestamps_s[get<1>(min_pq_s.top())] == -1) {
								min_pq_s.pop();
						}
						double updated_min_upper_bound = double(-1) * get<0>(min_pq_s.top());
					
						//UPDATE TOP VALUE AND ELIMINATE UNTIL NOT POSSIBLE ANYMORE
						while(updated_min_upper_bound < V_L_current_iteration[s]){

								//REMOVE THE TOP MIN ACTION
								int removableAction = get<1>(min_pq_s.top());

								//record action eliminated to return
								actions_eliminated_in_iteration.emplace_back(s, removableAction);											
								
								//remove by setting timestamp to -1, works in both queues
								timestamps_s[removableAction] = -1;

								//also, delete from unordered_set
								action_set_s.erase(removableAction);

								//pop the element from the queue
								min_pq_s.pop();
								
								//UPDATE NEW TOP MIN VALUE
								while (timestamps_s[get<1>(min_pq_s.top())] == -1) {
										min_pq_s.pop();
								}
								top_tuple_action_min = get<1>(min_pq_s.top());
								
								auto& [P_s_action_min, P_s_action_min_nonzero] = P[s][top_tuple_action_min];
								q_i_s_a_min = R[s][top_tuple_action_min] + gamma * sum_of_mult_nonzero_only(P_s_action_min, V_U_current_iteration, P_s_action_min_nonzero); 
							
								//INSERT NEW VALUE INTO BOTH PRIORITY QUEUES AND RECORD NEW TIMESTAMP
								min_pq_s.pop();
								new_top_action_tuple_min = make_tuple( (double(-1)) * q_i_s_a_min, top_tuple_action_min, iterations);
								min_pq_s.push(new_top_action_tuple_min);
							
								new_top_action_tuple_max = make_tuple(q_i_s_a_min, top_tuple_action_min, iterations);
								max_pq_s.push(new_top_action_tuple_max);
								timestamps_s[top_tuple_action_min] = iterations;
								
								//RECORD NEW MIN VALUE 
								while (timestamps_s[get<1>(min_pq_s.top())] == -1) {
										min_pq_s.pop();
								}
								updated_min_upper_bound = double(-1) * get<0>(min_pq_s.top());
						}
				if ((V_U_current_iteration[s]-V_L_current_iteration[s])> two_epsilon)
					bounded_convergence_criteria = false;
				if (abs(V_U_current_iteration[s]-oldVU)> convergence_bound_precomputed)
					upper_convergence_criteria=false;
				if (abs(V_L_current_iteration[s]-oldVL)> convergence_bound_precomputed)
					lower_convergence_criteria=false;

				}

				//see if any of the convergence criteria are met
				//1. bounded criteria
				
				
				//end timing of this iteration and record it in work vector
				auto end_of_iteration = high_resolution_clock::now();
				auto duration_of_iteration = duration_cast<microseconds>(end_of_iteration - start_of_iteration);
				work_per_iteration.push_back(duration_of_iteration);
				actions_eliminated.push_back(move(actions_eliminated_in_iteration));

		}
		//case return value on which convergence criteria was met
		vector<double> result(S); //set it so have size S from beginning to use copy
		
		if (bounded_convergence_criteria) {
				result = V_upper_lower_average(V_U[0], V_L[0], S);
		} else if (upper_convergence_criteria) {
				copy(V_U[0], V_U[0] + S, result.begin());
		} else if (lower_convergence_criteria) {
				copy(V_L[0], V_L[0] + S, result.begin());
		}
		V_type result_tuple = make_tuple(result, iterations, work_per_iteration, actions_eliminated);

		//SEG FAULT HERE, NOT WHEN I DELETE THIS
		for(int i = 0; i < 2; ++i) {
				delete [] V_U[i];
		}
		delete [] V_U;

		for(int i = 0; i < 2; ++i) {
				delete [] V_L[i];
		}
		delete [] V_L;

		for(int i = 0; i < S; ++i) {
				delete [] heap_time_stamps[i];
		}
		delete [] heap_time_stamps; 

		for (int s = 0; s < S; ++s) {
				//these are all with no brackets, as they are object deletions, not array pointers
				delete max_priority_queues[s];
				delete min_priority_queues[s];
				delete action_sets[s];
		}
		delete [] max_priority_queues;
		delete [] min_priority_queues;
		delete [] action_sets;

		return result_tuple;
}