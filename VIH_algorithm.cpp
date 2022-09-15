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
#define SIZE_T long long int
using namespace std;
using namespace std::chrono;


//returns true if the first argument is less than the second.
bool cmp_action_value_pairs(const q_action_pair_type &a, q_action_pair_type &b){
		return a.first <= b.first;	
		//return a.first < b.first;	
}

V_type value_iteration_with_heap(S_type S, R_type R, A_type A, P_type P, double gamma, double epsilon){
			//TODO if the arrays has A_max size, then a = A_max hs no entry as 0 is an action. One fix is to make it 1 bigger to have space for this index
		int A_max = find_max_A(A) + 1;

		//Find the maximum reward in the reward table
		auto [r_star_min, r_star_max, r_star_values] = find_all_r_values(R);
		//1. Improved Upper Bound
		double** V = new double*[2];
		for(int i = 0; i < 2; ++i) {
				V[i] = new double[S];
		}

		//int siz=sqrt(S-1)-2;
		//int Xmax=siz+2;
		//siz=Xmax/2;
		for(int s = 0; s < S; s++) {
				//V[0][s] = (gamma / (1.0 - gamma)) * r_star_max + r_star_values[s];
				//V[0][s] = (gamma / (1.0 - gamma)) * r_star_max;
				V[1][s] = 1;
				/*
				//V[0][s] = 0;
				int x_curr=s%Xmax;
				int y_curr=s/Xmax;
				//double x1= sqrt( pow( abs(x_curr-siz),2)+pow(abs(y_curr-siz),2));
				int xa1= abs(x_curr-siz);
				int ya1= abs(y_curr-siz);
				double x2=0;
				if (xa1>ya1)
					x2=xa1;
				else
					x2=ya1;
				//double x1= sqrt( pow( abs(x_curr-siz),2)+pow(abs(y_curr-siz),2));
				
				//V_U[0][s] = (gamma / (1.0 - gamma)) * r_star_max + r_star_values[s];
				V[0][s] = -x2 ;
				//V[1][s] = 1 ;*/
				V[0][s] = (gamma / (1.0 - gamma)) * r_star_max + r_star_values[s];


		}//V[0][S-1] = 0;
		//record actions eliminated in each iteration, where a pair is (state, action)
		//push empty vector for 0-index. Iterations start with 1
		vector<vector<pair<int, int>>> actions_eliminated;
		actions_eliminated.push_back(vector<pair<int, int>>());
		//keep track of work done in each iteration in microseconds
		//start from iteration 1
		vector<microseconds> work_per_iteration(1);
		
		//pre-compute convergence criteria for efficiency to not do it in each iteration of while loop
		//const double convergence_bound_precomputed = (epsilon * (1.0 - gamma)) / gamma;
		const double convergence_bound_precomputed = 0.0005;

		//HEAP INITIALIZATION
		//q_action_pair_type s_heaps[S][A_max];
		q_action_pair_type** s_heaps = new q_action_pair_type*[S];
		for(int i = 0; i < S; ++i) {
				//s_heaps[i] = new q_action_pair_type[A[i].size()];
				s_heaps[i] = new q_action_pair_type[A_max];
		}

		int* heap_size = new int[S];
		
		for (int s = 0; s < S; s++) {
				//Put the initial q(s,a) elements into the heap
				//fill each one with the maximum value of each action
				//vector<q_action_pair_type> s_h(A[s].size(),(R_max / (1 - gamma)));
				q_action_pair_type *s_h = s_heaps[s];

				//for (int a_index = 0; a_index < A[s].size(); a_index++){
					for (int a_index = 0; a_index <A_max; a_index++){
						//get the action of the index
						int a = A[s][a_index];

						//auto& [P_s_a, P_s_a_nonzero] = P[s][a];
						//use the even iteration, as this is the one used in the i = 1 iteration, that we want to pre-do
						//double q_1_s_a = R[s][a] + gamma * sum_of_mult_nonzero_only(P_s_a, V[0], P_s_a_nonzero);
						double q_1_s_a = V[0][s]; //(gamma / (1.0 - gamma)) * r_star_max + r_star_values[s];

						q_action_pair_type q_a_pair = make_pair(q_1_s_a, a);
						s_h[a_index] = q_a_pair;
				}

				//set the heap size
				heap_size[s] = A_max;

				//make it a heap for this state s
				make_heap(s_h, s_h + heap_size[s], cmp_action_value_pairs);
		}
		int iterations = 0;
		//gamma=1;
		while (abs_max_diff(V[0], V[1], S) > convergence_bound_precomputed){

				//Increment iteration counter i
				iterations++;	
				
				//Record actions eliminated in this iteration over all states
				vector<pair<int, int>> actions_eliminated_in_iteration;
				
				//begin timing of this iteration
				auto start_of_iteration = high_resolution_clock::now();
				
				//If i is even, then (i & 1) is 0, and the one to change is V[0]
				double *V_current_iteration = V[(iterations & 1)];
				double *V_previous_iteration = V[1 - (iterations & 1)];

				//for all states in each iteration
				for (int s = 0; s < S; s++) {
						//TODO if non-negative rewards, 0 is a lower bound of the maximization. to be changed if we want negative rewards
						q_action_pair_type *s_h = s_heaps[s];
						
						//update the top value
						int top_action = s_h[0].second;

						//the updated top value
						auto& [P_s_top_action, P_s_top_action_nonzero] = P[s][top_action];
						double updated_top_action_value = R[s][top_action] + gamma * sum_of_mult_nonzero_only(P_s_top_action, V_previous_iteration, P_s_top_action_nonzero);
						
						//The updated pair
						q_action_pair_type updated_pair = make_pair(updated_top_action_value, top_action);

						//now, we update the value
						pop_heap(s_h, s_h + heap_size[s], cmp_action_value_pairs);
						
						//can set the last element of an vector with v.back()
						//TODO check that -1 is correct
						s_h[heap_size[s] - 1] = updated_pair;	

						//push the last element into the heap and keep the heap property
						push_heap(s_h, s_h + heap_size[s], cmp_action_value_pairs);
						
						//the new top action
						int updated_top_action = s_h[0].second;

						while (top_action != updated_top_action){
								
								//update the top value
								top_action = s_h[0].second;

								//the updated top value
								auto& [P_s_top_action, P_s_top_action_nonzero] = P[s][top_action];
								updated_top_action_value = R[s][top_action] + gamma * sum_of_mult_nonzero_only(P_s_top_action, V_previous_iteration, P_s_top_action_nonzero);
								//The updated pair
								q_action_pair_type updated_pair = make_pair(updated_top_action_value, top_action);

								//now, we update the value
								pop_heap(s_h, s_h + heap_size[s], cmp_action_value_pairs);
								
								//can set the last element of an vector with v.back()
								//TODO check that -1 is correct
								s_h[heap_size[s] - 1] = updated_pair;	

								//push the last element into the heap and keep the heap property
								push_heap(s_h, s_h + heap_size[s], cmp_action_value_pairs);
								
								//the new top action
								updated_top_action = s_h[0].second;
								
						}
						V_current_iteration[s] = s_h[0].first;

				}
				
				//end timing of this iteration and record it in work vector
				auto end_of_iteration = high_resolution_clock::now();
				auto duration_of_iteration = duration_cast<microseconds>(end_of_iteration - start_of_iteration);
				work_per_iteration.push_back(duration_of_iteration);
				actions_eliminated.push_back(move(actions_eliminated_in_iteration));

		}
		vector<double> result(V[(iterations & 1)], V[(iterations & 1)] + S);
		V_type result_tuple = make_tuple(result, iterations, work_per_iteration, actions_eliminated);

		//DEALLOCATE MEMORY
		for(int i = 0; i < 2; ++i) {
				delete [] V[i];
		}
		delete [] V;

		for(int i = 0; i < S; ++i) {
				delete [] s_heaps[i];
		}
		delete [] s_heaps;

		delete [] heap_size;



		return result_tuple;
}

V_type value_iteration_with_heapGS(S_type S, R_type R, A_type A, P_type P, double gamma, double epsilon){
			//TODO if the arrays has A_max size, then a = A_max hs no entry as 0 is an action. One fix is to make it 1 bigger to have space for this index
		int A_max = find_max_A(A) + 1;

		//Find the maximum reward in the reward table
		auto [r_star_min, r_star_max, r_star_values] = find_all_r_values(R);
		//1. Improved Upper Bound
		double** V = new double*[1];
		for(int i = 0; i < 1; ++i) {
				V[i] = new double[S];
		}
		//int siz=sqrt(S-1)-2;
		//int Xmax=siz+2;
		//gamma=1;
		for(int s = 0; s < S; s++) {
				V[0][s] = (gamma / (1.0 - gamma)) * r_star_max + r_star_values[s];
		/*int x_curr=s%Xmax;
			int y_curr=s/Xmax;

				int xa1= abs(x_curr-siz);
				int ya1= abs(y_curr-siz);
				double x2=0;
				if (xa1>ya1)
					x2=xa1;
				else
					x2=ya1;
				V[0][s] = -x2+10;*/
		}//V[0][S-1] = 0.0;
		//record actions eliminated in each iteration, where a pair is (state, action)
		//push empty vector for 0-index. Iterations start with 1
		vector<vector<pair<int, int>>> actions_eliminated;
		actions_eliminated.push_back(vector<pair<int, int>>());
		//keep track of work done in each iteration in microseconds
		//start from iteration 1
		vector<microseconds> work_per_iteration(1);
		
		//pre-compute convergence criteria for efficiency to not do it in each iteration of while loop
		//const double convergence_bound_precomputed = (epsilon * (1.0 - gamma)) / gamma;
		const double convergence_bound_precomputed = 0.0005;
		
		//HEAP INITIALIZATION
		//q_action_pair_type s_heaps[S][A_max];
		q_action_pair_type** s_heaps = new q_action_pair_type*[S];
		for(int i = 0; i < S; ++i) {
				s_heaps[i] = new q_action_pair_type[A[i].size()];
				//s_heaps[i] = new q_action_pair_type[A_max];
		}

		int* heap_size = new int[S];
		
		for (int s = 0; s < S; s++) {
				//Put the initial q(s,a) elements into the heap
				//fill each one with the maximum value of each action
				//vector<q_action_pair_type> s_h(A[s].size(),(R_max / (1 - gamma)));
				q_action_pair_type *s_h = s_heaps[s];

				for (int a_index = 0; a_index < A[s].size(); a_index++){
						//get the action of the index
						int a = A[s][a_index];

						auto& [P_s_a, P_s_a_nonzero] = P[s][a];
						//use the even iteration, as this is the one used in the i = 1 iteration, that we want to pre-do
						//double q_1_s_a = R[s][a] + gamma * sum_of_mult_nonzero_only(P_s_a, V[0], P_s_a_nonzero);
						double q_1_s_a = V[0][s]; //(gamma / (1.0 - gamma)) * r_star_max + r_star_values[s];

						q_action_pair_type q_a_pair = make_pair(q_1_s_a, a);
						s_h[a_index] = q_a_pair;
				}

				//set the heap size
				heap_size[s] = A[s].size();

				//make it a heap for this state s
				make_heap(s_h, s_h + heap_size[s], cmp_action_value_pairs);
		}
		int iterations = 0;
		bool upper_convergence_criteria = false;
		double *V_current_iteration = V[0];

		while (!upper_convergence_criteria){
				upper_convergence_criteria=true;
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
						q_action_pair_type *s_h = s_heaps[s];
						
						//update the top value
						int top_action = s_h[0].second;
						double oldv=s_h[0].first;
						//the updated top value
						auto& [P_s_top_action, P_s_top_action_nonzero] = P[s][top_action];
						double updated_top_action_value = R[s][top_action] + gamma * sum_of_mult_nonzero_only(P_s_top_action, V_current_iteration, P_s_top_action_nonzero);
						
						//The updated pair
						q_action_pair_type updated_pair = make_pair(updated_top_action_value, top_action);

						//now, we update the value
						pop_heap(s_h, s_h + heap_size[s], cmp_action_value_pairs);
						
						//can set the last element of an vector with v.back()
						//TODO check that -1 is correct
						s_h[heap_size[s] - 1] = updated_pair;	

						//push the last element into the heap and keep the heap property
						push_heap(s_h, s_h + heap_size[s], cmp_action_value_pairs);
						
						//the new top action
						int updated_top_action = s_h[0].second;

						while (top_action != updated_top_action){
								
								//update the top value
								top_action = s_h[0].second;

								//the updated top value
								auto& [P_s_top_action, P_s_top_action_nonzero] = P[s][top_action];
								updated_top_action_value = R[s][top_action] + gamma * sum_of_mult_nonzero_only(P_s_top_action, V_current_iteration, P_s_top_action_nonzero);
								//The updated pair
								q_action_pair_type updated_pair = make_pair(updated_top_action_value, top_action);

								//now, we update the value
								pop_heap(s_h, s_h + heap_size[s], cmp_action_value_pairs);
								
								//can set the last element of an vector with v.back()
								//TODO check that -1 is correct
								s_h[heap_size[s] - 1] = updated_pair;	

								//push the last element into the heap and keep the heap property
								push_heap(s_h, s_h + heap_size[s], cmp_action_value_pairs);
								
								//the new top action
								updated_top_action = s_h[0].second;
								
						}
						V_current_iteration[s] = s_h[0].first;
						if (abs(oldv-V_current_iteration[s])> convergence_bound_precomputed)
							upper_convergence_criteria=false;
				}
				
				//end timing of this iteration and record it in work vector
				auto end_of_iteration = high_resolution_clock::now();
				auto duration_of_iteration = duration_cast<microseconds>(end_of_iteration - start_of_iteration);
				work_per_iteration.push_back(duration_of_iteration);
				actions_eliminated.push_back(move(actions_eliminated_in_iteration));

		}
		vector<double> result(V[(0)], V[(0)] + S);
		V_type result_tuple = make_tuple(result, iterations, work_per_iteration, actions_eliminated);

		//DEALLOCATE MEMORY
		for(int i = 0; i < 1; ++i) {
				delete [] V[i];
		}
		delete [] V;

		for(int i = 0; i < S; ++i) {
				delete [] s_heaps[i];
		}
		delete [] s_heaps;

		delete [] heap_size;



		return result_tuple;
}