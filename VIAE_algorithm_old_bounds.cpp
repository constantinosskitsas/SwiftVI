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

V_type value_iteration_action_elimination_old_bounds(S_type S, R_type R, A_type A, P_type P, double gamma, double epsilon){
		//Generate a size S vector with default 0 value
		//make this one 1 to make the difference large in first iteration i = 1 and then change these values withoput using them
		
		//Find the maximum reward in the reward table
		double R_max = find_max_R(R);
		double R_min = find_min_R(R);

		//odd iteration is the one that is the current iteration in the first while loop
		//double V_U[2][S];
		double** V_U = new double*[2];
		for(int i = 0; i < 2; ++i) {
				V_U[i] = new double[S];
		}
		fill(V_U[0], V_U[0] + S, R_max / (1.0 - gamma));
		fill(V_U[1], V_U[1] + S, 1);
		
		//PRINT IMPROVED UPPER BOUNDS
		printf("Upper Old Bounds:\n");
		for(int s = 0; s < 5; s++) {
				printf("state %d: %f\n", s, V_U[0][s]);
		}
		printf("\n");

		//odd iteration is the one that is the current iteration in the first while loop
		//double V_L[2][S];
		double** V_L = new double*[2];
		for(int i = 0; i < 2; ++i) {
				V_L[i] = new double[S];
		}
		fill(V_L[0], V_L[0] + S, R_min / (1.0 - gamma));
		fill(V_L[1], V_L[1] + S, 1);
		
		printf("Lower Old Bounds:\n");
		for(int s = 0; s < 5; s++) {
				printf("state %d: %f\n", s, V_L[0][s]);
		}
		printf("\n");

		//vector<double> max_R_values_per_state = find_max_R_for_each_state(R);
		//move(max_R_values_per_state.begin(), max_R_values_per_state.end(), V_L[0]);
		
		//keep track of work done in each iteration in microseconds
		//start from iteration 1
		vector<microseconds> work_per_iteration(1);

		//init criteria variables to know which value to return based on why the algorithm terminated
		//set to true if we have converged!
		bool bounded_convergence_criteria = false;
		bool upper_convergence_criteria = false;
		bool lower_convergence_criteria = false;

		//pre-compute convergence criteria for efficiency to not do it in each iteration of while loop
		const double convergence_bound_precomputed = (epsilon * (1.0 - gamma)) / gamma;
		const double two_epsilon = 2 * epsilon;

		//keep count of number of iterations
		int iterations = 0;
		
		//record actions eliminated in each iteration, where a pair is (state, action)
		//push empty vector for 0-index. Iterations start with 1
		vector<vector<pair<int, int>>> actions_eliminated;
		actions_eliminated.push_back(vector<pair<int, int>>());
		
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
						V_U_current_iteration[s] = numeric_limits<double>::min();
						V_L_current_iteration[s] = numeric_limits<double>::min();

						//keep the rewards from the upper bound for 
						int A_s_size = A[s].size(); 
						double Q_U_s[A_s_size];

						//keep index to put value into
						//ranged for loop over all actions in the action set of state s
						//for (auto a : A[s]) {
						for (int a_index = 0; a_index < A_s_size; a_index++) {

								int a = A[s][a_index];

								//reference to the probability vector of size S
								auto& [P_s_a, P_s_a_nonzero] = P[s][a];
								
								double Q_U_s_a = R[s][a] + gamma * sum_of_mult_nonzero_only(P_s_a, V_U_previous_iteration, P_s_a_nonzero); 

								//save the value of q^U(s,a)
								Q_U_s[a_index] = Q_U_s_a;

								double Q_L_s_a = R[s][a] + gamma * sum_of_mult_nonzero_only(P_s_a, V_L_previous_iteration, P_s_a_nonzero); 

								if (Q_U_s_a > V_U_current_iteration[s]) {
										V_U_current_iteration[s] = Q_U_s_a;
								}

								if (Q_L_s_a > V_L_current_iteration[s]) {
										V_L_current_iteration[s] = Q_L_s_a;
								}
						}
						
						//Beginning of action elimination procedure
						
						//The new action set for next iteration
						vector<int> A_s_new;

						//keep actions for next iteration that are NOT eliminated
						for (int a_index = 0; a_index < A[s].size(); a_index++) {
								//Check of action is to be KEPT for the next iteration
								if (Q_U_s[a_index] >= V_L_current_iteration[s]){
										A_s_new.push_back(A[s][a_index]);
								} else {
										actions_eliminated_in_iteration.emplace_back(s, A[s][a_index]);											
								}
						}

						//should be copying i.e. pass by value. TODO check if correct
						A[s] = A_s_new;
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

				//record work and actions elminated in this iteration
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

		//DEALLOCATE MEMORY
		for(int i = 0; i < 2; ++i) {
				delete [] V_U[i];
		}
		delete [] V_U;

		for(int i = 0; i < 2; ++i) {
				delete [] V_L[i];
		}
		delete [] V_L;
		
		return result_tuple;
		
}
