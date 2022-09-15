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

vector<int> first_convergence_iteration(S_type S, R_type R, A_type A, P_type P, double gamma, double epsilon){
		//Generate a size S vector with default 0 value
		//make this one 1 to make the difference large in first iteration i = 1 and then change these values withoput using them
		
		//Find the maximum reward in the reward table
		double R_max = find_max_R(R);

		//odd iteration is the one that is the current iteration in the first while loop
		double V_U[2][S];
		fill(V_U[0], V_U[0] + S, R_max / (1.0 - gamma));
		fill(V_U[1], V_U[1] + S, 1);
		
		//Find the minimum reward in the reward table
		double R_min = find_min_R(R);
		
		//odd iteration is the one that is the current iteration in the first while loop
		double V_L[2][S];
		fill(V_L[0], V_L[0] + S, R_min / (1.0 - gamma));
		fill(V_L[1], V_L[1] + S, 1);

		//Find the maximum reward in the reward table
		auto [r_star_min, r_star_max, r_star_values] = find_all_r_values(R);
		
		//odd iteration is the one that is the current iteration in the first while loop
		double V_U_improved[2][S];
		for(int s = 0; s < S; s++) {
				V_U[0][s] = (gamma / (1.0 - gamma)) * r_star_max + r_star_values[s];
				V_U[1][s] = 1.0;
		}
		
		//odd iteration is the one that is the current iteration in the first while loop
		double V_L_improved[2][S];
		for(int s = 0; s < S; s++) {
				V_L[0][s] = (gamma / (1.0 - gamma)) * r_star_min + r_star_values[s];
				V_L[1][s] = 1.0;
		}
		
		//init criteria variables to know which value to return based on why the algorithm terminated
		//set to true if we have converged!
		bool bounded_convergence_criteria = false;
		bool upper_convergence_criteria = false;
		bool lower_convergence_criteria = false;
		bool bounded_lower_max_convergence_criteria = false;

		//pre-compute convergence criteria for efficiency to not do it in each iteration of while loop
		const double convergence_bound_precomputed = (epsilon * (1.0 - gamma)) / gamma;
		const double two_epsilon = 2 * epsilon;

		//keep count of number of iterations
		int iterations = 0;
	
		//Record first iteration where each of them has converged
		//-1 is not converged yet
		vector<int> first_convergence_iteration{-1, -1, -1, -1};

		//DIFFERENT FROM THE ALGORITHMS: || INSTEAD OF &&
		//RUN UNTIL ALL HAS CONVERGED
		while ( (!bounded_convergence_criteria) || (!upper_convergence_criteria) || (!lower_convergence_criteria) || (!bounded_lower_max_convergence_criteria)){

				//Increment iteration counter i
				iterations++;	
				//printf("%d\n", iterations);	
				//If i is even, then (i & 1) is 0, and the one to change is V[0]
				double *V_U_current_iteration = V_U[(iterations & 1)];
				double *V_U_previous_iteration = V_U[1 - (iterations & 1)];

				double *V_L_current_iteration = V_L[(iterations & 1)];
				double *V_L_previous_iteration = V_L[1 - (iterations & 1)];
				
				double *V_U_improved_current_iteration = V_U_improved[(iterations & 1)];
				double *V_U_improved_previous_iteration = V_U_improved[1 - (iterations & 1)];

				double *V_L_improved_current_iteration = V_L_improved[(iterations & 1)];
				double *V_L_improved_previous_iteration = V_L_improved[1 - (iterations & 1)];
				
				//for all states in each iteration
				for (int s = 0; s < S; s++) {
						//TODO initial value to smalles possible, is it correct?
						V_U_current_iteration[s] = numeric_limits<double>::min();
						V_L_current_iteration[s] = numeric_limits<double>::min();
						V_U_improved_current_iteration[s] = numeric_limits<double>::min();
						V_L_improved_current_iteration[s] = numeric_limits<double>::min();

						//ranged for loop over all actions in the action set of state s
						for (auto a : A[s]) {
								auto& [P_s_a, P_s_a_nonzero] = P[s][a];
								double R_U_s_a = R[s][a] + gamma * sum_of_mult_nonzero_only(P_s_a, V_U_previous_iteration, P_s_a_nonzero); 
								double R_L_s_a = R[s][a] + gamma * sum_of_mult_nonzero_only(P_s_a, V_L_previous_iteration, P_s_a_nonzero); 

								if (R_U_s_a > V_U_current_iteration[s]) {
										V_U_current_iteration[s] = R_U_s_a;
								}
								if (R_L_s_a > V_L_current_iteration[s]) {
										V_L_current_iteration[s] = R_L_s_a;
								}
								double R_U_improved_s_a = R[s][a] + gamma * sum_of_mult_nonzero_only(P_s_a, V_U_improved_previous_iteration, P_s_a_nonzero); 
								double R_L_improved_s_a = R[s][a] + gamma * sum_of_mult_nonzero_only(P_s_a, V_L_improved_previous_iteration, P_s_a_nonzero); 

								if (R_U_improved_s_a > V_U_improved_current_iteration[s]) {
										V_U_improved_current_iteration[s] = R_U_improved_s_a;
								}
								if (R_L_improved_s_a > V_L_improved_current_iteration[s]) {
										V_L_improved_current_iteration[s] = R_L_improved_s_a;
								}
						}
				}

				//see if any of the convergence criteria are met
				//1. bounded criteria
				bounded_convergence_criteria = abs_max_diff(V_U_improved[(iterations & 1)], V_L_improved[(iterations & 1)], S) <= two_epsilon;
				if (first_convergence_iteration[0] == -1 && bounded_convergence_criteria){
						first_convergence_iteration[0] = iterations;
				}

				//2. upper criteria
				upper_convergence_criteria = abs_max_diff(V_U_improved[0], V_U_improved[1], S) <= convergence_bound_precomputed;
				if (first_convergence_iteration[1] == -1 && upper_convergence_criteria){
						first_convergence_iteration[1] = iterations;
				}

				//3. lower criteria
				lower_convergence_criteria = abs_max_diff(V_L_improved[0], V_L_improved[1], S) <= convergence_bound_precomputed;
				if (first_convergence_iteration[2] == -1 && lower_convergence_criteria){
						first_convergence_iteration[2] = iterations;
				}
				
				//4. CHANGED TO BE THE BOUNDED FROM OLD BOUNDS
				bounded_lower_max_convergence_criteria = abs_max_diff(V_U[(iterations & 1)], V_L[(iterations & 1)], S) <= two_epsilon;
				if (first_convergence_iteration[3] == -1 && bounded_lower_max_convergence_criteria){
						first_convergence_iteration[3] = iterations;
				}
				
		}
		return first_convergence_iteration;
}
