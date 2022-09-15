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

//PRETTY PRINTING HELPING FUNCTIONS

//Pretty print the reward table R of size S x A_num
//pass the argument as a reference to avoid copying
//call it as print_R(MDP_R), it automatically makes it as a reference
//changed values by R[i][j] = value changes it outside
void print_R(const R_type &R){
		printf("The Reward, R, table: \n");
		for (int i = 0; i < R.size(); ++i) {
				for (int j = 0; j < R[i].size(); ++j) {
					printf("%4.2f ", R[i][j]);
				}
				printf("\n");
		}
		printf("\n");
}

//Print A
void print_A(const A_type &A){
		printf("The action, A, table: \n");
		for (int i = 0; i < A.size(); ++i) {
				for (int j = 0; j < A[i].size(); ++j) {
					printf("%d ", A[i][j]);
				}
				printf("\n");
		}
		printf("\n");
}

//Print R
void print_P(const P_type &P){
		printf("The probability, P, table: \n");
		for (int i = 0; i < P.size(); ++i) {
				for (int j = 0; j < P[i].size(); ++j) {
						for (int l = 0; l < P[i][j].first.size(); ++l) {
								printf("%4.2f ", (P[i][j].first)[l]);
						}
						printf("\n");
				}
				printf("\n");
				printf("\n");
		}
		printf("\n");
}

void print_V(const V_result_type &V){
		printf("The state-value, V, table: \n");
		for (int i = 0; i < V.size(); ++i) {
				printf("%4.2f \n", V[i]);
		}
		printf("\n");
}

void print_V_array(const V_result_type &V){
		printf("The state-value, V, table: \n");
		for (int i = 0; i < V.size(); ++i) {
				printf("%4.2f \n", V[i]);
		}
		printf("\n");
}

void print_V_array(double arr[], int arr_size){
		for (int a = 0; a < arr_size; a++ ){
				printf("%4.2f ", arr[a]);
		}
		printf("\n");
}

void print_heap(heap_of_pairs_type heap){
	for(auto p : heap){
		printf("(%f,%d)", p.first, p.second);
	}
	printf("\n");
}

void print_max_min_heap(q_action_pair_type *max_min_heap, int heap_size){
	for(int i = 0; i < heap_size; i++){
		printf("(%f,%d)", max_min_heap[i].first, max_min_heap[i].second);
	}
	printf("\n");
}

//testing print methods
void print_int_array(int arr[], int arr_size){
		for (int a = 0; a < arr_size; a++ ){
				printf("%d ", arr[a]);
		}
		printf("\n");
}
