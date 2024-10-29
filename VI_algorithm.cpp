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
#include <memory.h>
#include <algorithm>
#include <iomanip>
//#include <omp.h> /
//#include <execution>
#include <thread>
#include <queue>
#include <utility>  // for std::pair
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
#include "PrioritizeSweep.h"
#include "updateable_priority_queue.h"
using namespace std;
using namespace std::chrono;
using namespace std::chrono;
using ComparatorType = std::function<bool(std::pair<double, int>, std::pair<double, int>)>;


    
static void fill_segment(MBIE* mb, const int s, const int a, const int start, const int end) {
	for (int i = start; i < end; i++) {
		mb->max_p[i] = mb->hatP[s][a][i];
	}
}

static void parallel_fill(MBIE* mb, const int s, const int a) {
	int num_threads = std::thread::hardware_concurrency(); // Get the number of threads supported by the system
	std::vector<std::thread> threads(num_threads);
	int chunk_size = mb->nS / num_threads; // Determine the size of the segment each thread will process

	for (int t = 0; t < num_threads; t++) {
		int start = t * chunk_size;
		int end = (t == num_threads - 1) ? mb->nS : start + chunk_size; // Ensure the last thread covers the remaining elements
		threads[t] = std::thread(fill_segment,mb,s, a, start, end);
	}

	// Join the threads with the main thread
	for (auto& thread : threads) {
		thread.join();
	}
}



V_type value_iteration(S_type S, R_type R, A_type A, P_type P, double gamma, double epsilon)
{

	// Find the maximum reward in the reward table
	auto [r_star_min, r_star_max, r_star_values] = find_all_r_values(R);

	// 2. Improved Lower Bound
	double **V = new double *[2];
	for (int i = 0; i < 2; ++i)
	{
		V[i] = new double[S];
	}
	int siz = sqrt(S - 1) - 2;
	int Xmax = siz + 2;
	siz = Xmax / 2;
	for (int s = 0; s < S; s++)
	{
		// V[0][s] = (gamma / (1.0 - gamma)) * r_star_min + r_star_values[s];

		int x_curr = s % Xmax;
		int y_curr = s / Xmax;

		double x1 = sqrt(pow(abs(x_curr - siz), 2) + pow(abs(y_curr - siz), 2));
		// V[0][s] = 0;
		V[1][s] = 0;
		// V[0][s] = 1;
		V[0][s] = -x1 * 5 - 10;
		// V[0][s] = -500;
	}
	V[0][S - 1] = 0;

	// record actions eliminated in each iteration, where a pair is (state, action)
	// push empty vector for 0-index. Iterations start with 1
	vector<vector<pair<int, int>>> actions_eliminated;
	actions_eliminated.push_back(vector<pair<int, int>>());

	// keep track of work done in each iteration in microseconds
	// start from iteration 1
	vector<microseconds> work_per_iteration(1);

	// keep count of number of iterations
	int iterations = 0;
	const double convergence_bound_precomputed = 0.0005;
	gamma = 1;
	// const double convergence_bound_precomputed = (epsilon * (1.0 - gamma)) / gamma;
	while (abs_max_diff(V[0], V[1], S) > (convergence_bound_precomputed))
	{

		// Increment iteration counter i
		iterations++;
		// Record actions eliminated in this iteration over all states
		vector<pair<int, int>> actions_eliminated_in_iteration;

		// begin timing of this iteration
		auto start_of_iteration = high_resolution_clock::now();

		// If i is even, then (i & 1) is 0, and the one to change is V[0]
		double *V_current_iteration = V[(iterations & 1)];
		double *V_previous_iteration = V[1 - (iterations & 1)];

		// for all states in each iteration
		for (int s = 0; s < S; s++)
		{
			// TODO if non-negative rewards, 0 is a lower bound of the maximization. to be changed if we want negative rewards
			// V_current_iteration[s] = numeric_limits<double>::min();
			V_current_iteration[s] = -1000000;

			// ranged for loop over all actions in the action set of state s
			for (auto a : A[s])
			{
				auto &[P_s_a, P_s_a_nonzero] = P[s][a];
				double R_s_a = R[s][a] + gamma * sum_of_mult_nonzero_only(P_s_a, V_previous_iteration, P_s_a_nonzero);
				if (R_s_a > V_current_iteration[s])
				{
					V_current_iteration[s] = R_s_a;
				}
			}
		}

		// end timing of this iteration and record it in work vector
		auto end_of_iteration = high_resolution_clock::now();
		auto duration_of_iteration = duration_cast<microseconds>(end_of_iteration - start_of_iteration);
		work_per_iteration.push_back(duration_of_iteration);
		actions_eliminated.push_back(move(actions_eliminated_in_iteration));
		/*for (int i = 0; i < S; ++i) {
			if(V_previous_iteration[i] - V_current_iteration[i]	>0)
}*/
	}
	vector<double> result(V[(iterations & 1)], V[(iterations & 1)] + S);
	V_type result_tuple = make_tuple(result, iterations, work_per_iteration, actions_eliminated);

	// DEALLOCATE MEMORY
	for (int i = 0; i < 2; ++i)
	{
		delete[] V[i];
	}
	delete[] V;

	return result_tuple;
}

UCLR::UCLR(S_type S, int _nA, double _gamma, double _epsilon, double _delta) {
	nS = S;
	nA = _nA;
	t = 0; //1
	k = 0; //1
	gamma = _gamma;
	r_delta = _delta;// / (2 * S * nA * 1);
	r_max = 0;//1.0+sqrt(log(2.0 / r_delta)/2.0);
	r_max_known = 0;
	epsilon = _epsilon;
	last_state_update = -1;
	last_action_update = -1;
	vsa = new int *[S];
	vsas = new int **[S];
	hatP = new double **[S];
	Nsas = new int **[S];
	Nsa = new int *[S];
	Rsa = new double *[S];
	vRsa = new double *[S];
	hatR = new double *[S];
	confR = new double *[S];
	confP = new double *[S];
	confP_long = new double **[S];
	StateSwift = new int [S];
	

	H = 1.0/(1.0-gamma)*log((8*nS)/(epsilon*(1-gamma)));
	w_min = (epsilon*(1-gamma))/(4*nS);
	delta_one = _delta/(2*nS*nA)*(1/(log2(nS)*log2(1/(w_min*(1-gamma))))); //log2
	L_one = log(2/delta_one); //nat log
	m = ((1280*L_one)/(pow(epsilon,2)*pow((log(log(1/(1-gamma)))),2)))*(log(nS/(epsilon*(1-gamma))))*log(1/(epsilon*(1-gamma))); //nat log

	std::cout << "m*w_min: " << m*w_min << std::endl;

	current_s = 0;
	last_s = 0;
	last_action = -1;
	
	for (int i = 0; i < S; ++i)
	{
		max_p.push_back(0.0);// (nS, 0.0);
		vsa[i] = new int[nA];
		Nsa[i] = new int[nA];
		Rsa[i] = new double[nA];
		vRsa[i] = new double[nA];
		hatR[i] = new double[nA];
		confR[i] = new double[nA];
		confP[i] = new double[nA];
		StateSwift[i] = 0;
		memset(vsa[i], 0, sizeof(int) * nA);
		memset(Nsa[i], 0, sizeof(int) * nA);
		memset(Rsa[i], 0.0, sizeof(double) * nA);
		memset(vRsa[i], 0.0, sizeof(double) * nA);
		memset(hatR[i], 0.0, sizeof(double) * nA);
	}

	for (int i = 0; i < S; i++)
	{
		Nsas[i] = new int *[nA];
		vsas[i] = new int *[nA];
		hatP[i] = new double *[nA];
		confP_long[i] = new double *[nA];
		for (int j = 0; j < nA; j++)
		{
			Nsas[i][j] = new int[S];
			vsas[i][j] = new int[S];
			hatP[i][j] = new double[S];
			confP_long[i][j] = new double[S];
			memset(Nsas[i][j], 0, sizeof(int) * S);
			memset(vsas[i][j], 0, sizeof(int) * S);
			memset(hatP[i][j], 0.0, sizeof(double) * S);
			//memset(confP[i][j], 0.0, sizeof(double) * S);
		}
	}

}

void UCLR::confidence() {
	r_max_known = 0;
	for (int s = 0; s < nS; s++)
	{
		for (int a = 0; a < nA; a++)
		{
			//Non-fixed 
			hatR[s][a] = Rsa[s][a]/(double)max(1, Nsa[s][a]);

			//Fixed known R
			if (r_max_known < Rsa[s][a]) {
				r_max_known = Rsa[s][a];
			}
			
			for (int s2 = 0; s2 < nS; s2++)
			{
				hatP[s][a][s2] = ((double) Nsas[s][a][s2])/max(1.0, (double) Nsa[s][a]);		
			}
		}	
	}
	r_max = 0;
	for (int s = 0; s < nS; s++)
	{
		for (int a = 0; a < nA; a++)
		{
			double n = max(1.0, (double) Nsa[s][a]);
			double delta2 = r_delta / (2 * nS * nA * max(1.0, (double) Nsa[s][a]));//Nsa[s][a]);  

			confR[s][a] = sqrt(log(2.0 / delta2) / (double) (2 * max(1, Nsa[s][a])));
			confP[s][a] = 0;
			double max_p = -1;
			for (int s2 = 0; s2 < nS; s2++)
			{	
				double p = hatP[s][a][s2];
				if (max_p < p) {
					max_p = p;
					confP[s][a] = min(sqrt((2.0*L_one*p*(1.0-p))/n)+(2.0*L_one)/(3.0*n),sqrt(L_one/(2.0*n)));
				}
				//not used
				confP_long[s][a][s2] = min(sqrt((2.0*L_one*p*(1.0-p))/n)+(2.0*L_one)/(3.0*n),sqrt(L_one/(2.0*n)));//sqrt((2.0 * (log(pow(2, nS) - 2) - log(delta)) / (double) max(1, Nsa[s][a])));
			//std::cout << confR[s][a] << " " << Nsa[s][a] << " " << 2*Nsa[s][a] <<  std::endl;
			//std::cout << confR[s][a] << std::endl;
			}
			if (r_max < hatR[s][a]+confR[s][a]) {
				r_max = hatR[s][a]+confR[s][a];
			}
		}	
	}

}

void UCLR::update(int s, int a) {
		/*for (int a = 0; a < nA; a++)
		{*/
			last_state_update = s;
			last_action_update = a;
			//Rsa[s][a] += vRsa[s][a]; //Non-fixed reward update (update should be 0 if R is considered known )
			Nsa[s][a] += vsa[s][a];
			vsa[s][a] = 0;
			for (int s2 = 0; s2 < nS; s2++) {
				StateSwift[s] = 0;
				if (Nsas[s][a][s2]+vsas[s][a][s2] != 0) {
					StateSwift[s] = 1;
				}
				Nsas[s][a][s2] += vsas[s][a][s2];
				vsas[s][a][s2] = 0;
				
			}
		/*}
	}*/
}

bool UCLR::end_act(int s, int action, bool verbose) {
	if (verbose) {
		std::cout << "vsa: " << vsa[s][action] << "\n";
		std::cout << "max: " << max(m*w_min, (double) Nsa[s][action]) << "\n";
		std::cout << "NSA: " << Nsa[s][action] << "\n";
		std::cout << "const: " << (nS*m)/(1-gamma) << "\n";
	}
	return ((vsa[s][action] >= max(m*w_min, (double) Nsa[s][action])) && Nsa[s][action] < (nS*m)/(1-gamma));
}

void UCLR::reset(S_type init) {
	r_max = 0; //1.0+sqrt(log(2.0 / r_delta)/2.0);
	r_max_known = 0;
	for (int i = 0; i < nS; i++)
	{
		StateSwift[i] = 0;
		for (int j = 0; j < nA; j++)
		{
			vsa[i][j] = 0;
			Rsa[i][j] = 0.0;
			vRsa[i][j] = 0.0;
			hatR[i][j] = 0.0;
			confR[i][j] = 0.0;
			confP[i][j] = 0.0;
			Nsa[i][j] = 0;
			for (int k = 0; k < nS; k++)
			{
				vsas[i][j][k] = 0;
				Nsas[i][j][k] = 0;
				hatP[i][j][k] = 0.0;
				confP_long[i][j][k] = 0.0;
			}
		}
	}
	current_s = init; //Not used
	last_action = -1;	//not used
}

void UCLR::max_proba(vector<int> sorted_indices, int s, int a)
{

	//a_i = max(0.0, hatP[s][a][i]-confP[s][a])
	//b_i = min(1.0, hatP[s][a][i]+confP[s][a])

	//double w = 0;
	double init_prob = 0;
	for (int i=0; i < nS; i++) {
		//set current prob to a
		max_p[i] = max(0.0, hatP[s][a][i]-confP[s][a]);
		init_prob += max_p[i];
	} 

	//Set delta remainder
	double delta_mass = 1 - init_prob;

	//Set index counter
	int idx = 0;

	//Assign mass remainder
	while (delta_mass > 0) {
		//Take the next highest sorted state
		int s_prime = sorted_indices[nS-1-idx];

		//Assign as much mass as possible up to its b val
		double delta_mass_prime = min(delta_mass, min(1.0, hatP[s][a][s_prime]+confP[s][a])-max(0.0, hatP[s][a][s_prime]-confP[s][a]));
		
		//Assign mass to the probability
		max_p[s_prime] += delta_mass_prime;

		//Update mass remainder
		delta_mass -= delta_mass_prime;

		//Increment index counter 
		idx += 1;
	}

	//Max_p has been updated


	//******************OLD**************
	//double min1 = min(1.0, hatP[s][a][sorted_indices[nS - 1]] + confP[s][a] / 2.0);
	
	/*#pragma omp parallel
	{   
    auto tid = omp_get_thread_num();
    auto chunksize = max_p.size() / omp_get_num_threads();
    auto begin = max_p.begin() + chunksize * tid;
    auto end = (tid == omp_get_num_threads() -1) ? max_p.end() : begin + chunksize;
    std::fill(begin, end, 0.0);
	}*/

	//
	/*for (int i =0; i<nS; i++) {
		max_p[i] = 0;
	}*/
	

	/*if (min1 == 1.0)
	{
		//std::fill(std::execution::par, max_p.begin(),max_p.end(),0.0);
		std::fill(max_p.begin(),max_p.end(),0.0);
		
		for (int i =0; i<nS; i++) {
			max_p[i] = 0;
		}
		max_p[sorted_indices[nS - 1]] = 1.0;
		
		
	}
	else
	{	
		//std::cout << "My fill" << std::endl;
		//parallel_fill(this,s,a);
		for (int i = 0; i < nS; i++){
			max_p[i] = hatP[s][a][i];
			//if (hatP[s][a][i] != 0) {
			//	std::cout << s << " " << i << "  " << hatP[s][a][i] << std::endl;
			//}
		}
		//Mega copy hack (that may or may not work)
		//std::copy(hatP[s][a], hatP[s][a]+nS,max_p.begin());
		//max_p.assign(*hatP[s][a], *hatP[s][a] + nS);
		//vector<double> max_p(hatP[s][a].begin(), hatP[s][a].end());
		
		max_p[sorted_indices[nS - 1]] += confP[s][a] / 2.0;
		//std::cout << hatP[s][a][sorted_indices[nS - 1]] << std::endl;
		//std::cout << std::endl;
		
		int l = 0;
		double sum_max_p = 0.0;
		for (int i = 0; i < max_p.size(); i++)
		{
			sum_max_p += max_p[i];
		}
		while (sum_max_p > 1.0)
		{
			max_p[sorted_indices[l]] = max(0.0, 1.0 - sum_max_p + max_p[sorted_indices[l]]);
			l++;

			// Recalculate the sum of max_p
			sum_max_p = 0.0;
			for (int i = 0; i < max_p.size(); i++)
			{
				sum_max_p += max_p[i];
			}
		}
		
	}*/
	//max_p has been set
	//for (auto i: max_p) {
	//	std::cout << i << " ";
	//}
	//std::cout << std::endl;
}

vector<int> UCLR::swiftEVI(){
	int max_iter = 2000;
	int niter = 0;
	//int nS = S;
	vector<int> sorted_indices(nS);
	
	// Fill the vector with indices
	//iota(sorted_indices.begin(), sorted_indices.end(), 0);
	vector<int> policy(nS, 0);
	std::vector<double> V0(nS);
	for (int i = 0; i < nS; i++)
	{

		for (int a_index = 0; a_index < nA; a_index++)
		{
			//FIXED R
			double r_bound = (gamma / (1.0 - gamma))*r_max_known+Rsa[i][a_index];
			
			//Non-fixed R
			//double r_bound = (gamma / (1.0 - gamma))*(r_max)+hatR[i][a_index]+confR[i][a_index];
			
			if (r_bound > V0[i]) {
				V0[i] = r_bound;
			}
		}
	}
	
	iota(sorted_indices.begin(), sorted_indices.end(), 0);
	sort(sorted_indices.begin(), sorted_indices.end(), [&](int i,int j){return V0[i]<V0[j];} );
		
	// Initialize V1
	vector<double> V1(nS, 1.0); // Initialize with ones
	double _epsilon = epsilon * (1.0 - gamma) / (2.0 * gamma);
	double temp=0;

	// Heap init
	q_action_pair_type **s_heaps = new q_action_pair_type *[nS];
	for (int i = 0; i < nS; i++)
	{
		// s_heaps[i] = new q_action_pair_type[A[i].size()];
		s_heaps[i] = new q_action_pair_type[nA];
	}

	int *heap_size = new int[nS];

	for (int s = 0; s < nS; s++)
	{ 
		if (true) {//(StateSwift[s]==1){
		// Put the initial q(s,a) elements into the heap
		// fill each one with the maximum value of each action
		// vector<q_action_pair_type> s_h(A[s].size(),(R_max / (1 - gamma)));
		q_action_pair_type *s_h = s_heaps[s];

		/*for (int a_index = 0; a_index < nA; a_index++)
		{
			double r_bound = (gamma / (1.0 - gamma))*(1.0+confR[s][a_index])+1.0+confR[s][a_index];
			if (r_bound > V0[s]) {
				V0[s] = r_bound;
			}
		}*/
		// for (int a_index = 0; a_index < A[s].size(); a_index++){
		for (int a_index = 0; a_index < nA; a_index++)
		{
			// get the action of the index
			//int a = A[s][a_index];
			//(Currently a is always its index)

			// aq_action_pair_type *s_h = s_heaps[s];uto& [P_s_a, P_s_a_nonzero] = P[s][a];
			// use the even iteration, as this is the one used in the i = 1 iteration, that we want to pre-do
			// double q_1_s_a = R[s][a] + gamma * sum_of_mult_nonzero_only(P_s_a, V[0], P_s_a_nonzero);
			//V0[s] = (gamma / (1.0 - gamma))*(1.0+sqrt(log(2.0 / r_delta)/2.0))+hatR[s][a_index]+confR[s][a_index]; //(gamma / (1.0 - gamma))*(1.0+sqrt(log(2.0 / delta)/2.0))+1.0+sqrt(log(2.0 / delta)/2.0);//(gamma / (1.0 - gamma))*1+1;//1.0 / (1.0 - gamma);
		
			double q_1_s_a = V0[s]; //(gamma / (1.0 - gamma)) * r_star_max + r_star_values[s];

			//q_action_pair_type q_a_pair = make_pair(q_1_s_a, a_index);
			q_action_pair_type q_a_pair = make_pair(q_1_s_a, a_index);
			s_h[a_index] = q_a_pair;
		}

		// set the heap size
		heap_size[s] = nA;

		// make it a heap for this state s
		make_heap(s_h, s_h + heap_size[s], cmp_action_value_pairs);
	}}
	double R_s_a=0;
	while (true)
	{
		niter++;
		//std::cout << niter << std::endl;
		//std::cout << nA << std::endl;
		for (int s = 0; s < nS; s++)
		{
			if (true){//(StateSwift[s]==1){
			q_action_pair_type *s_h = s_heaps[s];
			//nt old_action = -1;
			//std::cout << s_h[0].first << "  " << s_h[0].second << std::endl;
			//std::cout << s_h[1].first << "  " << s_h[1].second << std::endl;
			//std::cout << std::endl;
			//std::cout << std::endl;
			int heap_loops = 0;
			while (true) {
				heap_loops++;

				//std::cout << "loop" << std::endl;
				// update the top value
				int top_action = s_h[0].second;
				double old = s_h[0].first;
		
				/*for (int l = 0; l < nS; l++)
				{
					std::cout << sorted_indices[l] << " ";
				}
				
				std::cout << std::endl;*/
				if (heap_loops > 4000000) {
					std::cout << heap_loops << "   " << old << "   " << top_action << std::endl;
				}
				//std::cout << top_action << st
				max_proba(sorted_indices, s, top_action);
				//auto &[P_s_a, P_s_a_nonzero] = hatP[s][a];

				//double updated_top_action_value = std::min(hatR[s][top_action] + confR[s][top_action],1.0) + gamma * sum_of_mult(max_p, V0);
				
				//FIXED R
				double updated_top_action_value = Rsa[s][top_action] + gamma * sum_of_mult(max_p, V0);
				
				//Non-fixed R
				//double updated_top_action_value = hatR[s][top_action]+confR[s][top_action] + gamma * sum_of_mult(max_p, V0);
				
				if (heap_loops > 4000000) {
					std::cout << heap_loops << "   "<< updated_top_action_value << "   "<< old << std::endl;
				}

				q_action_pair_type updated_pair = make_pair(updated_top_action_value, top_action);
				/*if (cnt >= 119) {
			    	std::cout << updated_top_action_value << "  " << hatR[s][top_action] << "  " << confR[s][top_action] << std::endl;;
				}*/
				//if (updated_top_action_value != updated_top_action_value) {
				//	std::cout << std::endlR_s_a;
				//} 
				/*if (updated_top_action_value < old+0.000000000002 && updated_top_action_value >= old) {
					break;
				}*/
				if (heap_loops > 4000000) {
					std::cout << heap_loops << "  1 " << s_h[0].first << "   " << s_h[0].second << std::endl;
				}
				pop_heap(s_h, s_h + heap_size[s], cmp_action_value_pairs);
				double temp_top_val = s_h[0].first;
				if (heap_loops > 4000000) {
					std::cout << heap_loops << "  2 " << s_h[heap_size[s] - 1].first << "   " << s_h[heap_size[s] - 1].second << std::endl;
				}
				if (heap_loops > 4000000) {
					std::cout << heap_loops << "  3 " << s_h[0].first << "   " << s_h[0].second << std::endl;
				}
				s_h[heap_size[s] - 1] = updated_pair;
				if (heap_loops > 4000000) {
					std::cout << heap_loops << "  4 " << s_h[heap_size[s] - 1].first << "   " << s_h[heap_size[s] - 1].second << std::endl;
				}
				push_heap(s_h, s_h + heap_size[s], cmp_action_value_pairs);
				if (heap_loops > 4000000) {
					std::cout << heap_loops << "  5 " << s_h[0].first << "   " << s_h[0].second << std::endl;
				}
				if (heap_loops > 4000000) {
					std::cout << heap_loops << "  6 " << s_h[heap_size[s] - 1].first << "   " << s_h[heap_size[s] - 1].second << std::endl;
				}
				if (heap_loops > 4000000) {
					std::cout << heap_loops << "   " << s_h[0].first << "   " << s_h[1].first<< "   " << s_h[2].first<< "   " << s_h[3].first << std::endl;
				}
				int new_action = s_h[0].second;
				//if (top_action == 0 || temp > V1[s])
				//{
				//	V1[s] = temp;
				//	policy[s] = top_action;
				//}
				if (updated_top_action_value > old+0.0000000000002) {
			    	std::cout << "ILLEGAL UPGRADE "<< std::endl;
					std::cout << "New val: " << std::setprecision(9) << updated_top_action_value <<"\n";
					std::cout << "Old val: " << std::setprecision(9) << old <<"\n";
					std::cout << "hatR: " << hatR[s][top_action] << "  confR: " << confR[s][top_action] << "\n";
					std::cout << " vsa: " << vsa[s][top_action] << "\n";
					std::cout << " NSA: " << Nsa[s][top_action] << "\n";
					//std::cout << " v2: " << sqrt(log(2.0 / r_delta) / (double) (2));
					//std::cout << " nsa: " << Nsa[s][top_action] <<" delta: " << r_delta <<  "\n";
					std::cout << std::endl;
					for (auto i: sorted_indices) {
						std::cout << V0[i] << " ";
					}
					std::cout << std::endl;
					std::cout << "Max proba sum: ";
					double sum_max_p = 0.0;
					for (int i = 0; i < max_p.size(); i++)
					{
						sum_max_p += max_p[i];
					}
					std::cout << sum_max_p << std::endl;

					//std::cout <<"old val: " << old << " new val: "  << updated_top_action_value << std::endl;
					//std::cout <<"old: " << top_action << " new: "  << new_action << std::endl;
				}
				
				if (top_action == new_action || temp_top_val == updated_pair.first) {
					/*if (heap_loops > nA || heap_loops < nA) {
						std::cout << heap_loops << std::endl;
					}*/
					break;
				}
				if (heap_loops > 4000000) {
					std::cout << heap_loops << "   " << updated_top_action_value << "   " << old << "   " << (updated_top_action_value==old) << "   " << new_action << "   " << top_action << "   " << (temp_top_val == updated_pair.first) << std::endl;
					
				}
			}
			V1[s] = s_h[0].first;
			/*if (cnt >= 119) {
				std::cout << cnt << " " << s << " " << s_h[0].first << "  " << s_h[0].second << std::endl;
			}*/
			/*std::cout << s_h[0].first << "  " << s_h[0].second << std::endl;
			std::cout << s_h[1].first << "  " << s_h[1].second << std::endl;
			std::cout << std::endl;*/
			policy[s] = s_h[0].second;
		}else{
			for (int a = 0; a < nA; a++)
			{
				max_proba(sorted_indices, s, a);
				//auto &[P_s_a, P_s_a_nonzero] = hatP[s][a];
				R_s_a = hatR[s][a] + confR[s][a] + gamma * sum_of_mult(max_p, V0);
				if (a == 0 || R_s_a > V1[s]) 
				{
					V1[s] = R_s_a;
					policy[s] = a;
				}
			}

		}

		}
		

		// V distance
		/*int dist = 0;
		for (int i = 0; i < nS; i++) 
		{
			dist += (V0[i]-V1[i])*(V0[i]-V1[i]);
		}
		dist = sqrt(dist);*/
		//abs_max_diff(V0, V1, nS);
		//if (abs_max_diff(V0, V1, nS) < _epsilon) 
		if (abs_max_diff(V0, V1, nS)-abs_min_diff(V0,V1, nS) < epsilon) 
		{
			return policy;
		} 
		else 
		{
			//for (int i = 0; i< nS; i++) {
			//	V0[i] =
			//}
			
			std::swap(V0,V1);
			//V0 = V1; //copy
			for (int i = 0; i < nS; i++)
			{
				//V1[i] = (gamma / (1.0 - gamma))*(1.0+sqrt(log(2.0 / r_delta))/2)+1.0+sqrt(log(2.0 / r_delta))/2; //(gamma / (1.0 - gamma))*1+1;	
			}
			//sorted indices
			/*std::cout << std::endl;
			std::cout << "cnt: " << cnt << "| ";
			for (auto i: V0) {
				std::cout << i << " ";
			}
			std::cout << std::endl;
			for (auto i: sorted_indices) {
				std::cout << i << " ";
			}
			std::cout << std::endl;*/
		
			iota(sorted_indices.begin(), sorted_indices.end(), 0);
			sort(sorted_indices.begin(), sorted_indices.end(), [&](int i,int j){return V0[i]<V0[j];} );
			
			/*std::cout << std::endl;
			for (auto i: sorted_indices) {
				std::cout << i << " ";
			}
			std::cout << std::endl;
			std::cout << std::endl;
			std::cout << std::endl;*/

		}
		if (max_iter == niter) {
			std::cout << "Early stop in swiftEVI: "<< abs_max_diff(V0, V1, nS) << "  " << _epsilon  << std::endl;
			
			return policy;
		}
	}
}

vector<int> UCLR::EVI()
{
	int max_iter = 2000;
	int niter = 0;
	//int nS = S;
	vector<int> sorted_indices(nS);
	
	// Fill the vector with indices
	//iota(sorted_indices.begin(), sorted_indices.end(), 0);
	vector<int> policy(nS, 0);
	std::vector<double> V0(nS);
	for (int i = 0; i < nS; i++)
	{
		
		
		//V0[i] = (gamma / (1.0 - gamma))*(1.0+sqrt(log(2.0 / r_delta)/2.0))+1.0+sqrt(log(2.0 / r_delta)/2.0); //(gamma / (1.0 - gamma))*(1.0+sqrt(log(2.0 / delta)/2.0))+1.0+sqrt(log(2.0 / delta)/2.0);//(gamma / (1.0 - gamma))*1+1;//1.0 / (1.0 - gamma);

		for (int a_index = 0; a_index < nA; a_index++)
		{
			//FIXED R
			double r_bound = (gamma / (1.0 - gamma))*r_max_known+Rsa[i][a_index];
			
			//Non-fixed R
			//double r_bound = (gamma / (1.0 - gamma))*(r_max)+hatR[i][a_index]+confR[i][a_index];
			
			if (r_bound > V0[i]) {
				V0[i] = r_bound;
			}
		}
	}

	iota(sorted_indices.begin(), sorted_indices.end(), 0);
	sort(sorted_indices.begin(), sorted_indices.end(), [&](int i,int j){return V0[i]<V0[j];} );
		
	// Initialize V1
	//TODO
	vector<double> V1(nS, 1.0); //(gamma / (1.0 - gamma))*(1.0+sqrt(log(2.0 / delta)/2.0))+1.0+sqrt(log(2.0 / delta)/2.0));//(gamma / (1.0 - gamma))*(1.0+sqrt(log(2.0 / delta)/2.0))+1.0+sqrt(log(2.0 / delta)/2.0)); // Initialize with ones
	
	//double _epsilon = epsilon * (1.0 - gamma) / (2.0 * gamma);
	double R_s_a=0;

	while (true)
	{
		niter++;
		//std::cout << niter << std::endl;
		for (int s = 0; s < nS; s++)
		{
			for (int a = 0; a < nA; a++)
			{
				/*for (int l = 0; l < nS; l++)
				{
					std::cout << sorted_indices[l] << " ";
				}
				std::cout << std::endl;*/
				max_proba(sorted_indices, s, a);
				//auto &[P_s_a, P_s_a_nonzero] = hatP[s][a];
				
				//Fixed R
				R_s_a = Rsa[s][a] + gamma * sum_of_mult(max_p, V0);
				
				//Non-fixed R
				//R_s_a = hatR[s][a]+confR[s][a] + gamma * sum_of_mult(max_p, V0);
				
				//R_s_a = min(hatR[s][a] + confR[s][a],1.0) + gamma * sum_of_mult(max_p, V0);
				
				/*if (cnt > 110) {
					std::cout << R_s_a << "  " << hatR[s][a] << "  " << confR[s][a] << std::endl;;
				}*/

				if (a == 0 || R_s_a > V1[s]) 
				{
					V1[s] = R_s_a;
					policy[s] = a;
				}
			}
			/*if (cnt > 110) {
				std::cout << cnt << " " << s << " "  << V1[s] << "  " << policy[s] << std::endl;
			}
			std::cout << std::endl;*/
		}
		/*std::cout << "cnt: " << cnt << "| ";
		for (auto i: V1) {
			std::cout << i << " " ;
		}
		std::cout << std::endl;*/
		
		// V distance
		/*int dist = 0;
		for (int i = 0; i < nS; i++) 
		{
			dist += (V0[i]-V1[i])*(V0[i]-V1[i]);
		}
		dist = sqrt(dist);*/
		//std::cout << dist << std::endl;
		//if (abs_max_diff(V0, V1, nS) < _epsilon) 
		if (abs_max_diff(V0, V1, nS)-abs_min_diff(V0,V1, nS) < epsilon) 
		{
			//std::cout << niter << std::endl;
			/*for (auto i: V1) {
				std::cout << i << " " ;
			}
			std::cout << std::endl;*/
			return policy;
		} 
		else 
		{
			//for (int i = 0; i< nS; i++) {
			//	V0[i] =
			//}
			std::swap(V0,V1);
			//V0 = V1; //copy
			//why? we dont need it the way you have  (a == 0 || R_s_a > V1[s]) 
			for (int i = 0; i < nS; i++)
			{
				//V1[i] = (gamma / (1.0 - gamma))*(1.0+sqrt(log(2.0 / delta))/2)+1.0+sqrt(log(2.0 / delta))/2;//(gamma / (1.0 - gamma))*1+1;//1.0 / (1.0 - gamma);
			}
			//sorted indices
			iota(sorted_indices.begin(), sorted_indices.end(), 0);
			sort(sorted_indices.begin(), sorted_indices.end(), [&](int i,int j){return V0[i]<V0[j];} );
		}
		if (max_iter == niter) {
			std::cout << "Early stop in EVI: "<< abs_max_diff(V0, V1, nS) << "  " << epsilon  << std::endl;
			
			return policy;
		}
	}
}

void MBIE::delete_MBIE() {
    for (int i = 0; i < nS; ++i) {
        delete[] Nsa[i];
        delete[] Rsa[i];
        delete[] hatR[i];
        delete[] confR[i];
        delete[] confP[i];
        for (int j = 0; j < nA; ++j) {
            delete[] Nsas[i][j];
            delete[] hatP[i][j];
        }
        delete[] Nsas[i];
        delete[] hatP[i];
    }

    delete[] Nsa;
    delete[] Rsa;
    delete[] hatR;
    delete[] confR;
    delete[] confP;
    delete[] StateSwift;
    delete[] Nsas;
    delete[] hatP;
}


MBIE::MBIE(S_type S, int _nA, double _gamma, double _epsilon, double _delta, int _m) {
	delta = _delta;
	m = _m;
	nA = _nA; //Assumes the same number of actions for all states
	nS = S;
	gamma = _gamma;
	epsilon = _epsilon;
	// max(A[0].size();)
	// or S*4;
	//delta = _delta / (2 * S * nA * m);
	hatP = new double **[S];
	Nsas = new int **[S];
	Nsa = new int *[S];
	Rsa = new double *[S];
	hatR = new double *[S];
	confR = new double *[S];
	confP = new double *[S];
	StateSwift= new int [S];
	cnt = 0;
	r_max = 0;
	r_max_known = 0;

	vector<int> policy(S, 0);

	
	current_s = 0;
	last_action = -1;
	
	for (int i = 0; i < S; ++i)
	{
		max_p.push_back(0.0);// (nS, 0.0);
		Nsa[i] = new int[nA];
		Rsa[i] = new double[nA];
		hatR[i] = new double[nA];
		confR[i] = new double[nA];
		confP[i] = new double[nA];
		StateSwift[i]=0;
		memset(Nsa[i], 0, sizeof(int) * nA);
		memset(Rsa[i], 0.0, sizeof(double) * nA);
		memset(hatR[i], 0.0, sizeof(double) * nA);
	}

	//
	//	self.confP = np.zeros((self.nS, self.nA))

	for (int i = 0; i < S; i++)
	{
		Nsas[i] = new int *[nA];
		hatP[i] = new double *[nA];
		for (int j = 0; j < nA; j++)
		{
			Nsas[i][j] = new int[S];
			hatP[i][j] = new double[S];
			memset(Nsas[i][j], 0, sizeof(int) * S);
			memset(hatP[i][j], 0.0, sizeof(double) * S);
		}
	}
}



std::tuple<int,std::vector<int>> MBIE::playbao(int state, double reward){
		//std::cout << state << " " << reward << std::endl;
	//If not first action
	if (last_action >= 0) 
	{	
		cnt++;
		Nsas[current_s][last_action][state] += 1;
		Rsa[current_s][last_action] += reward; //update with 0 if fixed
	}

	// conduct updatesresult[t] += 
	confidence();
	r_max_known = 0;
	r_max = 0;
	for (int s = 0; s < nS; s++)
	{
		for (int a = 0; a < nA; a++)
		{
			//Fixed known R
			if (r_max_known < Rsa[s][a]) {
				r_max_known = Rsa[s][a];
			}

			//Unkown R
			hatR[s][a] = Rsa[s][a]/(double)max(1, Nsa[s][a]);
			for (int s2 = 0; s2 < nS; s2++)
			{
				hatP[s][a][s2] = ((double) Nsas[s][a][s2])/max(1, Nsa[s][a]);		
			}
			if (r_max < hatR[s][a]+confR[s][a]) {
				r_max = hatR[s][a]+confR[s][a];
			}
		}
	}
	//Estimate equation 6
	policy = baoEVI();
	//Follow the most optimistic greedy policy
	int action = policy[state];

	//Update with choice
	Nsa[state][action] += 1;
	current_s = state;
	last_action = action;

	return {action, policy};
}

std::tuple<int,std::vector<int>> MBIE::playswift(int state, double reward) {
	//std::cout << state << " " << reward << std::endl;
	//If not first action
	if (last_action >= 0) 
	{
		cnt++;
		Nsas[current_s][last_action][state] += 1;
		Rsa[current_s][last_action] += reward; //Update with zero if fixed
	} 

	// conduct updates
	confidence();
	float Conf_Sum=0;
	r_max = 0;
	r_max_known = 0;
	for (int s = 0; s < nS; s++)
	{
		Conf_Sum=0;
		for (int a = 0; a < nA; a++)
		{
			//Fixed known R
			if (r_max_known < Rsa[s][a]) {
				r_max_known = Rsa[s][a];
			}

			//Unkown R
			hatR[s][a] = Rsa[s][a]/(double)max(1, Nsa[s][a]);
			
			Conf_Sum+=(confP[s][a]+confR[s][a]);
			for (int s2 = 0; s2 < nS; s2++)
			{
				hatP[s][a][s2] = ((double) Nsas[s][a][s2])/max(1, Nsa[s][a]);		
			}
			if (r_max < hatR[s][a]+confR[s][a]) {
				r_max = hatR[s][a]+confR[s][a];
			}
		}if (Conf_Sum/(2*nA)>1){
			//StateSwift[s]=1;
		}else{
			//StateSwift[s]=1;
		}
		
	}
	//Estimate equation 6
	//policy = baoEVI();
	policy = swiftEVI();
	//Follow the most optimistic greedy policy
	int action = policy[state];

	//Update with choice
	Nsa[state][action] += 1;
	current_s = state;
	last_action = action;

	return {action, policy};
}
std::tuple<int,std::vector<int>> MBIE::update_vals(int state, double reward) {
	if (last_action >= 0) 
	{	
		cnt++;
		Nsas[current_s][last_action][state] += 1;
		Rsa[current_s][last_action] += reward;
	}
	int action = policy[state];
	//Update with choice
	Nsa[state][action] += 1;
	current_s = state;
	last_action = action;

	return {action, policy};
}

std::tuple<int,std::vector<int>> MBIE::play(int state, double reward) {
	//std::cout << state << " " << reward << std::endl;
	//If not first action
	if (last_action >= 0) 
	{	
		cnt++;
		Nsas[current_s][last_action][state] += 1;
		Rsa[current_s][last_action] += reward; //update with 0 if fixed
	}

	// conduct updatesresult[t] += 
	confidence();
	r_max_known = 0;
	r_max = 0;
	for (int s = 0; s < nS; s++)
	{
		for (int a = 0; a < nA; a++)
		{
			//Fixed known R
			if (r_max_known < Rsa[s][a]) {
				r_max_known = Rsa[s][a];
			}

			//Unkown R
			hatR[s][a] = Rsa[s][a]/(double)max(1, Nsa[s][a]);
			for (int s2 = 0; s2 < nS; s2++)
			{
				hatP[s][a][s2] = ((double) Nsas[s][a][s2])/max(1, Nsa[s][a]);		
			}
			if (r_max < hatR[s][a]+confR[s][a]) {
				r_max = hatR[s][a]+confR[s][a];
			}
		}
	}
	//Estimate equation 6
	policy = EVI();
	//Follow the most optimistic greedy policy
	int action = policy[state];

	//Update with choice
	Nsa[state][action] += 1;
	current_s = state;
	last_action = action;

	return {action, policy};
}

void MBIE::confidence()
{
	
	for (int s = 0; s < nS; s++)
	{
		for (int a = 0; a < nA; a++)
		{
			double delta2 = delta / (2 * nS * nA * (double) max(1, Nsa[s][a]));
			confP[s][a] = sqrt((2.0 * (log(pow(2, nS) - 2) - log(delta2)) / (double) max(1, Nsa[s][a])));
			//std::cout << confR[s][a] << " " << Nsa[s][a] << " " << 2*Nsa[s][a] <<  std::endl;
			confR[s][a] = sqrt(log(2.0 / delta2) / (double) (2 * max(1, Nsa[s][a])));
			//std::cout << confR[s][a] << std::endl;
		}
	}
}

void MBIE::reset(S_type init)
{
	r_max = 0;
	for (int i = 0; i < nS; i++)
	{
		for (int j = 0; j < nA; j++)
		{
			Rsa[i][j] = 0.0;
			hatR[i][j] = 0.0;
			confR[i][j] = 0.0;
			confP[i][j] = 0.0;
			Nsa[i][j] = 0;
			for (int k = 0; k < nS; k++)
			{
				Nsas[i][j][k] = 0;
				hatP[i][j][k] = 0.0;
			}
		}
	}
	current_s = init;
	last_action = -1;
	cnt = 0;
}

void MBIE::max_proba(vector<int> sorted_indices, int s, int a)
{
	double min1 = min(1.0, hatP[s][a][sorted_indices[nS - 1]] + confP[s][a] / 2.0);
	
	/*#pragma omp parallel
	{   
    auto tid = omp_get_thread_num();
    auto chunksize = max_p.size() / omp_get_num_threads();
    auto begin = max_p.begin() + chunksize * tid;
    auto end = (tid == omp_get_num_threads() -1) ? max_p.end() : begin + chunksize;
    std::fill(begin, end, 0.0);
	}*/

	//
	/*for (int i =0; i<nS; i++) {
		max_p[i] = 0;
	}*/
	

	if (min1 == 1)
	{
		//std::fill(std::execution::par, max_p.begin(),max_p.end(),0.0);
		std::fill(max_p.begin(),max_p.end(),0.0);
		
		for (int i =0; i<nS; i++) {
			max_p[i] = 0;
		}
		max_p[sorted_indices[nS - 1]] = 1.0;
		
		
	}
	else
	{	
		//std::cout << "My fill" << std::endl;
		//parallel_fill(this,s,a);
		for (int i = 0; i < nS; i++){
			max_p[i] = hatP[s][a][i];
			//if (hatP[s][a][i] != 0) {
			//	std::cout << s << " " << i << "  " << hatP[s][a][i] << std::endl;
			//}
		}
		//Mega copy hack (that may or may not work)
		//std::copy(hatP[s][a], hatP[s][a]+nS,max_p.begin());
		//max_p.assign(*hatP[s][a], *hatP[s][a] + nS);
		//vector<double> max_p(hatP[s][a].begin(), hatP[s][a].end());
		
		max_p[sorted_indices[nS - 1]] += confP[s][a] / 2.0;
		//std::cout << hatP[s][a][sorted_indices[nS - 1]] << std::endl;
		//std::cout << std::endl;
		
		int l = 0;
		double sum_max_p = 0.0;
		for (int i = 0; i < max_p.size(); i++)
		{
			sum_max_p += max_p[i];
		}
		while (sum_max_p > 1.0)
		{
			max_p[sorted_indices[l]] = max(0.0, 1.0 - sum_max_p + max_p[sorted_indices[l]]);
			l++;

			// Recalculate the sum of max_p
			sum_max_p = 0.0;
			for (int i = 0; i < max_p.size(); i++)
			{
				sum_max_p += max_p[i];
			}
		}
		
	}
	//max_p has been set
	//for (auto i: max_p) {
	//	std::cout << i << " ";
	//}
	//std::cout << std::endl;
}

vector<int> MBIE::swiftEVI()
{
	int max_iter = 2000;
	int niter = 0;
	//int nS = S;
	vector<int> sorted_indices(nS);
	
	// Fill the vector with indices
	//iota(sorted_indices.begin(), sorted_indices.end(), 0);
	vector<int> policy(nS, 0);
	std::vector<double> V0(nS);
	for (int i = 0; i < nS; i++)
	{

		for (int a_index = 0; a_index < nA; a_index++)
		{
			//FIXED R
			double r_bound = (gamma / (1.0 - gamma))*r_max_known+Rsa[i][a_index];
			
			//Non-fixed R
			//double r_bound = (gamma / (1.0 - gamma))*(r_max)+hatR[i][a_index]+confR[i][a_index];
			
			if (r_bound > V0[i]) {
				V0[i] = r_bound;
			}
		}
	}

	iota(sorted_indices.begin(), sorted_indices.end(), 0);
	sort(sorted_indices.begin(), sorted_indices.end(), [&](int i,int j){return V0[i]<V0[j];} );
		

	// Initialize V1
	vector<double> V1(nS, 1.0); // Initialize with ones
	double _epsilon = epsilon * (1.0 - gamma) / (2.0 * gamma);
	double temp=0;

	// Heap init
	q_action_pair_type **s_heaps = new q_action_pair_type *[nS];
	for (int i = 0; i < nS; i++)
	{
		// s_heaps[i] = new q_action_pair_type[A[i].size()];
		s_heaps[i] = new q_action_pair_type[nA];
	}

	int *heap_size = new int[nS];

	for (int s = 0; s < nS; s++)
	{ 
		if (true) {//(StateSwift[s]==1){
		// Put the initial q(s,a) elements into the heap
		// fill each one with the maximum value of each action
		// vector<q_action_pair_type> s_h(A[s].size(),(R_max / (1 - gamma)));
		q_action_pair_type *s_h = s_heaps[s];

		/*for (int a_index = 0; a_index < nA; a_index++)
		{
			double r_bound = (gamma / (1.0 - gamma))*(1.0+confR[s][a_index])+1.0+confR[s][a_index];
			if (r_bound > V0[s]) {
				V0[s] = r_bound;
			}
		}*/
		// for (int a_index = 0; a_index < A[s].size(); a_index++){
		for (int a_index = 0; a_index < nA; a_index++)
		{
			// get the action of the index
			//int a = A[s][a_index];
			//(Currently a is always its index)

			// auto& [P_s_a, P_s_a_nonzero] = P[s][a];
			// use the even iteration, as this is the one used in the i = 1 iteration, that we want to pre-do
			// double q_1_s_a = R[s][a] + gamma * sum_of_mult_nonzero_only(P_s_a, V[0], P_s_a_nonzero);
			
			double q_1_s_a = V0[s]; //(gamma / (1.0 - gamma)) * r_star_max + r_star_values[s];

			//q_action_pair_type q_a_pair = make_pair(q_1_s_a, a_index);
			q_action_pair_type q_a_pair = make_pair(q_1_s_a, a_index);
			s_h[a_index] = q_a_pair;
		}

		// set the heap size
		heap_size[s] = nA;

		// make it a heap for this state s
		make_heap(s_h, s_h + heap_size[s], cmp_action_value_pairs);
	}}
	double R_s_a=0;
	while (true)
	{
		niter++;
		//std::cout << niter << std::endl;
		//std::cout << nA << std::endl;
		for (int s = 0; s < nS; s++)
		{
			if (true){//(StateSwift[s]==1){
			q_action_pair_type *s_h = s_heaps[s];
			//nt old_action = -1;
			//std::cout << s_h[0].first << "  " << s_h[0].second << std::endl;
			//std::cout << s_h[1].first << "  " << s_h[1].second << std::endl;
			//std::cout << std::endl;
			//std::cout << std::endl;
			int heap_loops = 0;
			while (true) {
				heap_loops++;

				//std::cout << "loop" << std::endl;
				// update the top value
				int top_action = s_h[0].second;
				double old = s_h[0].first;
		
				/*for (int l = 0; l < nS; l++)
				{
					std::cout << sorted_indices[l] << " ";
				}
				std::cout << std::endl;*/
				//std::cout << top_action << st
				max_proba(sorted_indices, s, top_action);
				//auto &[P_s_a, P_s_a_nonzero] = hatP[s][a];

				//double updated_top_action_value = std::min(hatR[s][top_action] + confR[s][top_action],1.0) + gamma * sum_of_mult(max_p, V0);
				
				//FIXED R
				double updated_top_action_value = Rsa[s][top_action]+ gamma * sum_of_mult(max_p, V0);
				
				//Non-fixed R
				//double updated_top_action_value = hatR[s][top_action] + confR[s][top_action] + gamma * sum_of_mult(max_p, V0);
				
				q_action_pair_type updated_pair = make_pair(updated_top_action_value, top_action);
				/*if (cnt >= 119) {
			    	std::cout << updated_top_action_value << "  " << hatR[s][top_action] << "  " << confR[s][top_action] << std::endl;;
				}*/
				//if (updated_top_action_value != updated_top_action_value) {
				//	std::cout << std::endlR_s_a;
				//} 
				pop_heap(s_h, s_h + heap_size[s], cmp_action_value_pairs);
				double temp_top_val = s_h[0].first;
				s_h[heap_size[s] - 1] = updated_pair;
				push_heap(s_h, s_h + heap_size[s], cmp_action_value_pairs);

				int new_action = s_h[0].second;
				//if (top_action == 0 || temp > V1[s])
				//{
				//	V1[s] = temp;
				//	policy[s] = top_action;
				//}
				if (updated_top_action_value > old+0.00000001) {
			    	std::cout << "ILLEGAL UPGRADE "<< std::endl;
					std::cout << "New val: " << std::setprecision(9) << updated_top_action_value <<"\n";
					std::cout << "Old val: " << std::setprecision(9) << old <<"\n";
					std::cout << "hatR: " << hatR[s][top_action] << "  confR: " << confR[s][top_action] << "\n";
					std::cout << " NSA: " << Nsa[s][top_action] << "\n";
					//std::cout << " v2: " << sqrt(log(2.0 / delta) / (double) (2));
					//std::cout << " nsa: " << Nsa[s][top_action] <<" delta: " << delta <<  "\n";
					std::cout << "Max proba sum: ";
					double sum_max_p = 0.0;
					for (int i = 0; i < max_p.size(); i++)
					{
						sum_max_p += max_p[i];
					}
					std::cout << sum_max_p << std::endl;

					//std::cout <<"old val: " << old << " new val: "  << updated_top_action_value << std::endl;
					//std::cout <<"old: " << top_action << " new: "  << new_action << std::endl;
				}
				
				if (top_action == new_action || temp_top_val == updated_pair.first) {
					/*if (heap_loops > nA || heap_loops < nA) {
						std::cout << heap_loops << std::endl;
					}*/
					break;
				}
			}
			V1[s] = s_h[0].first;
			/*if (cnt >= 119) {
				std::cout << cnt << " " << s << " " << s_h[0].first << "  " << s_h[0].second << std::endl;
			}*/
			/*std::cout << s_h[0].first << "  " << s_h[0].second << std::endl;
			std::cout << s_h[1].first << "  " << s_h[1].second << std::endl;
			std::cout << std::endl;*/
			policy[s] = s_h[0].second;
		}else{
			for (int a = 0; a < nA; a++)
			{
				max_proba(sorted_indices, s, a);
				//auto &[P_s_a, P_s_a_nonzero] = hatP[s][a];
				R_s_a = hatR[s][a] + confR[s][a] + gamma * sum_of_mult(max_p, V0);
				if (a == 0 || R_s_a > V1[s]) 
				{
					V1[s] = R_s_a;
					policy[s] = a;
				}
			}

		}

		}
		

		// V distance
		/*int dist = 0;
		for (int i = 0; i < nS; i++) 
		{
			dist += (V0[i]-V1[i])*(V0[i]-V1[i]);
		}
		dist = sqrt(dist);*/
		//abs_max_diff(V0, V1, nS);
		//if (abs_max_diff(V0, V1, nS) < _epsilon) 
		if (abs_max_diff(V0, V1, nS)-abs_min_diff(V0,V1, nS) < epsilon) 
		{
			return policy;
		} 
		else 
		{
			//for (int i = 0; i< nS; i++) {
			//	V0[i] =
			//}
			
			std::swap(V0,V1);
			//V0 = V1; //copy
			for (int i = 0; i < nS; i++)
			{
				//V1[i] = (gamma / (1.0 - gamma))*(1.0+sqrt(log(2.0 / delta))/2)+1.0+sqrt(log(2.0 / delta))/2; //(gamma / (1.0 - gamma))*1+1;	
			}
			//sorted indices
			/*std::cout << std::endl;
			std::cout << "cnt: " << cnt << "| ";
			for (auto i: V0) {
				std::cout << i << " ";
			}
			std::cout << std::endl;
			for (auto i: sorted_indices) {
				std::cout << i << " ";
			}
			std::cout << std::endl;*/
		
			iota(sorted_indices.begin(), sorted_indices.end(), 0);
			sort(sorted_indices.begin(), sorted_indices.end(), [&](int i,int j){return V0[i]<V0[j];} );
			
			/*std::cout << std::endl;
			for (auto i: sorted_indices) {
				std::cout << i << " ";
			}
			std::cout << std::endl;
			std::cout << std::endl;
			std::cout << std::endl;*/

		}
		if (max_iter == niter) {
			std::cout << "Early stop in swiftEVI: "<< abs_max_diff(V0, V1, nS) << "  " << _epsilon  << std::endl;
			
			return policy;
		}
	}
}
vector<int> MBIE::baoEVI(){
	int max_iter = 2000;
	int niter = 0;
	//int nS = S;
	vector<int> sorted_indices(nS);
	
	// Fill the vector with indices
	//iota(sorted_indices.begin(), sorted_indices.end(), 0);
	vector<int> policy(nS, 0);
	std::vector<double> V0(nS);
	for (int i = 0; i < nS; i++)
	{

		for (int a_index = 0; a_index < nA; a_index++)
		{
			//FIXED R
			double r_bound = (gamma / (1.0 - gamma))*r_max_known+Rsa[i][a_index];
			
			//Non-fixed R
			//double r_bound = (gamma / (1.0 - gamma))*(r_max)+hatR[i][a_index]+confR[i][a_index];
			
			if (r_bound > V0[i]) {
				V0[i] = r_bound;
			}
		}
	}

	iota(sorted_indices.begin(), sorted_indices.end(), 0);
	sort(sorted_indices.begin(), sorted_indices.end(), [&](int i,int j){return V0[i]<V0[j];} );

	vector<double> V1(nS, 1.0);//(gamma / (1.0 - gamma))*(1.0+sqrt(log(2.0 / delta)/2.0))+1.0+sqrt(log(2.0 / delta)/2.0)); // Initialize with ones
	double _epsilon = epsilon * (1.0 - gamma) / (2.0 * gamma);
	double R_s_a=0;
	double **Q_values_per_state = new double *[nS];
	for (int i = 0; i < nS; ++i)
	{
		// Q_values_per_state[i] = new double[A[i].size()];
		Q_values_per_state[i] = new double[nA];
	}
		for (int s = 0; s < nS; s++)
	{
		// pointers to the heaps of current state s
		double *Q_values_s = Q_values_per_state[s];
		for (int a = 0; a < nA; a++)
		{
			// for (int a = 0; a < (A[s].size()); a++){
			// Q_values_s[a] = (r_star_max / (1.0 - gamma));	//init with V_max as stated in BAO1 paper
			// Q_values_s[a] = (gamma / (1.0 - gamma)) * r_star_max + r_star_values[s]; //init with upper bound V_U
			Q_values_s[a] = V0[s];
			// Q_values_s[a]=1.0;
		}
	}
	//int niter = 0;
		while (true)
	{

		// Increment iteration counter
		niter++;

		// Record actions eliminated in this iteration over all states
		//vector<pair<int, int>> actions_eliminated_in_iteration;

		// begin timing of this iteration
		auto start_of_iteration = high_resolution_clock::now();

		// If iiteration is even, then (iteration & 1) is 0, and the one to change is V[0]
		

		// for all states in each iteration
		for (int s = 0; s < nS; s++)
		{
			// keep best actions here
			double *Q_values_s = Q_values_per_state[s];

			// start with delta value larger than epsilon such that we go into while loop at least once
			double inner_delta = epsilon + 1;

			while (!(inner_delta < epsilon))
			{

				// Find Max Q value
				// double Q_max = numeric_limits<double>::min();
				double Q_max = -100000;
				//best action is policy, so just get the policy?
				for (int a = 0; a < nA; a++)
				{
					// for (int a = 0; a < A[s].size(); a++){
					if (Q_values_s[a] > Q_max)
					{
						Q_max = Q_values_s[a];
					}
				}
				

				// best_actions: find those actions that are at most epsilon from largest action
				vector<int> best_actions;
				for (int a = 0; a < nA; a++)
				{
					// for (int a = 0; a < A[s].size(); a++){
					if (abs(Q_values_s[a] - Q_max) < epsilon)
					{
						best_actions.push_back(a);
					}
				}

				inner_delta = 0.00000;

				for (int a : best_actions)
				{
					double old_q = Q_values_s[a];

					// actually update this value Q(s,a)
					max_proba(sorted_indices, s, a);
					
					//FIXED R
					R_s_a = Rsa[s][a] + gamma * sum_of_mult(max_p, V0);

					//Non-fixed R
					//R_s_a = hatR[s][a] + confR[s][a] + gamma * sum_of_mult(max_p, V0);
					
					Q_values_s[a] = R_s_a;

					if (abs(old_q - Q_values_s[a]) > inner_delta)
					{
						inner_delta = abs(old_q - Q_values_s[a]);
					}
				}
			}

			// find new value of V_U[s]
			// V_U_current_iteration[s] = numeric_limits<double>::min();
			V1[s] = -100000;
			for (int a = 0; a < nA; a++)
			{
				// for (int a = 0; a <A[s].size(); a++){
				if (Q_values_s[a] > V1[s])
				{
					V1[s] = Q_values_s[a];
					policy[s] = a;
				}
			}
		}

		// Check if upper convergence criteria is met
		if (abs_max_diff(V0, V1, nS)-abs_min_diff(V0,V1, nS) < epsilon) 
		{
			//std::cout << niter << std::endl;
			return policy;
		} 
		else 
		{
			//for (int i = 0; i< nS; i++) {
			//	V0[i] =
			//}
			std::swap(V0,V1);
			//V0 = V1; //copy
			// no need
			for (int i = 0; i < nS; i++)
			{
				//V1[i] = (gamma / (1.0 - gamma))*(1.0+sqrt(log(2.0 / delta))/2)+1.0+sqrt(log(2.0 / delta))/2;//(gamma / (1.0 - gamma))*1+1;//1.0 / (1.0 - gamma);
			}
			//sorted indices
			iota(sorted_indices.begin(), sorted_indices.end(), 0);
			sort(sorted_indices.begin(), sorted_indices.end(), [&](int i,int j){return V0[i]<V0[j];} );
		}
		if (max_iter == niter) {
			std::cout << "Early stop in baoEVI: "<< abs_max_diff(V0, V1, nS) << "  " << _epsilon  << std::endl;
			
			return policy;
		}
	}
}
vector<int> MBIE::EVI()
{
	int max_iter = 2000;
	int niter = 0;
	//int nS = S;
	vector<int> sorted_indices(nS);
	
	// Fill the vector with indices
	//iota(sorted_indices.begin(), sorted_indices.end(), 0);
	vector<int> policy(nS, 0);
	std::vector<double> V0(nS);
	for (int i = 0; i < nS; i++)
	for (int i = 0; i < nS; i++)
	{

		for (int a_index = 0; a_index < nA; a_index++)
		{
			//FIXED R
			double r_bound = (gamma / (1.0 - gamma))*r_max_known+Rsa[i][a_index];
			
			//Non-fixed R
			//double r_bound = (gamma / (1.0 - gamma))*(r_max)+hatR[i][a_index]+confR[i][a_index];
			
			if (r_bound > V0[i]) {
				V0[i] = r_bound;
			}
		}
	}

	iota(sorted_indices.begin(), sorted_indices.end(), 0);
	sort(sorted_indices.begin(), sorted_indices.end(), [&](int i,int j){return V0[i]<V0[j];} );
		

	// Initialize V1
	vector<double> V1(nS, 1.0);//(gamma / (1.0 - gamma))*(1.0+sqrt(log(2.0 / delta)/2.0))+1.0+sqrt(log(2.0 / delta)/2.0)); // Initialize with ones
	
	double _epsilon = epsilon * (1.0 - gamma) / (2.0 * gamma);
	double R_s_a=0;

	while (true)
	{
		niter++;
		//std::cout << niter << std::endl;
		for (int s = 0; s < nS; s++)
		{
			for (int a = 0; a < nA; a++)
			{
				/*for (int l = 0; l < nS; l++)
				{
					std::cout << sorted_indices[l] << " ";
				}
				std::cout << std::endl;*/
				max_proba(sorted_indices, s, a);
				//auto &[P_s_a, P_s_a_nonzero] = hatP[s][a];
				
				//FIXED R
				R_s_a = Rsa[s][a] + gamma * sum_of_mult(max_p, V0);
				
				//Non-Fixed R
				//R_s_a = hatR[s][a] + confR[s][a] + gamma * sum_of_mult(max_p, V0);
				
				//R_s_a = min(hatR[s][a] + confR[s][a],1.0) + gamma * sum_of_mult(max_p, V0);
				
				/*if (cnt > 110) {
					std::cout << R_s_a << "  " << hatR[s][a] << "  " << confR[s][a] << std::endl;;
				}*/

				if (a == 0 || R_s_a > V1[s]) 
				{
					V1[s] = R_s_a;
					policy[s] = a;
				}
			}
			/*if (cnt > 110) {
				std::cout << cnt << " " << s << " "  << V1[s] << "  " << policy[s] << std::endl;
			}
			std::cout << std::endl;*/
		}
		/*std::cout << "cnt: " << cnt << "| ";
		for (auto i: V1) {
			std::cout << i << " " ;
		}
		std::cout << std::endl;*/
		
		// V distance
		/*int dist = 0;
		for (int i = 0; i < nS; i++) 
		{
			dist += (V0[i]-V1[i])*(V0[i]-V1[i]);
		}
		dist = sqrt(dist);*/
		//std::cout << dist << std::endl;
		//if (abs_max_diff(V0, V1, nS) < _epsilon) 
		if (abs_max_diff(V0, V1, nS)-abs_min_diff(V0,V1, nS) < epsilon) 
		{
			//std::cout << niter << std::endl;
			return policy;
		} 
		else 
		{
			//for (int i = 0; i< nS; i++) {
			//	V0[i] =
			//}
			std::swap(V0,V1);
			//V0 = V1; //copy
			//why? we dont need it the way you have  (a == 0 || R_s_a > V1[s]) 
			for (int i = 0; i < nS; i++)
			{
				//V1[i] = (gamma / (1.0 - gamma))*(1.0+sqrt(log(2.0 / delta))/2)+1.0+sqrt(log(2.0 / delta))/2;//(gamma / (1.0 - gamma))*1+1;//1.0 / (1.0 - gamma);
			}
			//sorted indices
			iota(sorted_indices.begin(), sorted_indices.end(), 0);
			sort(sorted_indices.begin(), sorted_indices.end(), [&](int i,int j){return V0[i]<V0[j];} );
		}
		if (max_iter == niter) {
			std::cout << "Early stop in EVI: "<< abs_max_diff(V0, V1, nS) << "  " << _epsilon  << std::endl;
			
			return policy;
		}
	}
}
/*V_type MBIE(S_type S, R_type R, A_type A, P_type P, double gamma, double epsilon, double delta, int m)
{
	delta = 0.05;
	m = 100;
	int nA = 4;
	// max(A[0].size();)
	// or S*4;
	delta = delta / (2 * S * nA * m);
	int s_state = 0;
	int **Nsa = NULL;
	float ***hatP = NULL;
	float **Rsa = NULL;
	int ***Nsas = NULL;
	float **hatR = NULL;
	float **confR = NULL;
	float **confP = NULL;
	hatP = new float **[S];
	Nsas = new int **[S];
	Nsa = new int *[S];
	Rsa = new float *[S];
	hatR = new float *[S];
	confR = new float *[S];
	confP = new float *[S];

	for (int i = 0; i < S; ++i)
	{
		Nsa[i] = new int[nA];
		Rsa[i] = new float[nA];
		hatR[i] = new float[nA];
		confR[i] = new float[nA];
		confP[i] = new float[nA];
		memset(Nsa[i], 0, sizeof(int) * nA);
		memset(Rsa[i], 0, sizeof(float) * nA);
		memset(hatR[i], 0, sizeof(float) * nA);
	}

	//
	//	self.confP = np.zeros((self.nS, self.nA))

	for (int i = 0; i < S; i++)
	{
		Nsas[i] = new int *[nA];
		hatP[i] = new float *[nA];
		for (int j = 0; j < nA; j++)
		{
			Nsas[i][j] = new int[S];
			hatP[i][j] = new float[S];
			memset(Nsas[i][j], 0, sizeof(int) * S);
			memset(hatP[i][j], 0, sizeof(float) * S);
		}
	}
}*/

V_type value_iterationGS(S_type S, R_type R, A_type A, P_type P, double gamma, double epsilon)
{

	// Find the maximum reward in the reward table
	auto [r_star_min, r_star_max, r_star_values] = find_all_r_values(R);

	// 2. Improved Lower Bound
	double **V = new double *[1];
	for (int i = 0; i < 1; ++i)
	{
		V[i] = new double[S];
	}
	// int siz=sqrt(S-1)-2;
	// int Xmax=siz+2;
	// gamma=1;
	for (int s = 0; s < S; s++)
	{
		/*
		int x_curr=s%Xmax;
			int y_curr=s/Xmax;
		double x1= sqrt( pow( abs(x_curr-siz),2)+pow(abs(y_curr-siz),2));
		V[0][s] = -x1*5-10;*/
		V[0][s] = (gamma / (1.0 - gamma)) * r_star_min + r_star_values[s];
	} // V[0][S-1] = 0;

	// record actions eliminated in each iteration, where a pair is (state, action)
	// push empty vector for 0-index. Iterations start with 1
	vector<vector<pair<int, int>>> actions_eliminated;
	actions_eliminated.push_back(vector<pair<int, int>>());

	// keep track of work done in each iteration in microseconds
	// start from iteration 1
	vector<microseconds> work_per_iteration(1);

	//policy
	vector<int> policy(S, 0);

	// keep count of number of iterations
	int iterations = 0;
	bool upper_convergence_criteria = false;
	const double convergence_bound_precomputed = (epsilon * (1.0 - gamma)) / gamma;
	//const double convergence_bound_precomputed = 0.0005;

	while (!upper_convergence_criteria)
	{
		upper_convergence_criteria = true;
		// Increment iteration counter i
		iterations++;
		// Record actions eliminated in this iteration over all states
		vector<pair<int, int>> actions_eliminated_in_iteration;

		// begin timing of this iteration
		auto start_of_iteration = high_resolution_clock::now();

		// If i is even, then (i & 1) is 0, and the one to change is V[0]
		double *V_current_iteration = V[0];

		// for all states in each iteration
		for (int s = 0; s < S; s++)
		{
			// TODO if non-negative rewards, 0 is a lower bound of the maximization. to be changed if we want negative rewards
			// V_current_iteration[s] = double(0);
			double oldV = V_current_iteration[s];
			// ranged for loop over all actions in the action set of state s
			for (auto a : A[s])
			{
				auto &[P_s_a, P_s_a_nonzero] = P[s][a];
				double R_s_a = R[s][a] + gamma * sum_of_mult_nonzero_only(P_s_a, V_current_iteration, P_s_a_nonzero);
				if (R_s_a > V_current_iteration[s])
				{
					V_current_iteration[s] = R_s_a;
					policy[s] = a;
				}
			}

			if (abs(oldV - V_current_iteration[s]) > convergence_bound_precomputed)
				upper_convergence_criteria = false;
		}

		// end timing of this iteration and record it in work vector
		auto end_of_iteration = high_resolution_clock::now();
		auto duration_of_iteration = duration_cast<microseconds>(end_of_iteration - start_of_iteration);
		work_per_iteration.push_back(duration_of_iteration);
		actions_eliminated.push_back(move(actions_eliminated_in_iteration));
	}
	vector<double> result(V[0], V[0] + S);
	V_type result_tuple = make_tuple(result, iterations, work_per_iteration, actions_eliminated);
/*
	std::cout << "V_star policy: ";
	for (int i: policy) {
		std::cout << i;
	}
	std::cout << std::endl;
*/
	// DEALLOCATE MEMORY
	for (int i = 0; i < 1; ++i)
	{
		delete[] V[i];
	}
	delete[] V;

	return result_tuple;
}

V_type value_iterationGSPS(S_type S, R_type R, A_type A, P_type P, double gamma, double epsilon)
{

	// Find the maximum reward in the reward table
	auto [r_star_min, r_star_max, r_star_values] = find_all_r_values(R);

	// 2. Improved Lower Bound
	double **V = new double *[1];
	for (int i = 0; i < 1; ++i)
	{
		V[i] = new double[S];
	}

	for (int s = 0; s < S; s++)
	{

		V[0][s] = (gamma / (1.0 - gamma)) * r_star_min + r_star_values[s];
	} 

	// record actions eliminated in each iteration, where a pair is (state, action)
	// push empty vector for 0-index. Iterations start with 1
	vector<vector<pair<int, int>>> actions_eliminated;
	actions_eliminated.push_back(vector<pair<int, int>>());
	double *reverseV = new double [S];
	// keep track of work done in each iteration in microseconds
	// start from iteration 1
	vector<microseconds> work_per_iteration(1);
	std::vector<std::vector<int>> predecessor (S);

	//std::priority_queue<std::pair<double, int>, std::vector<std::pair<double, int>>, ComparatorType> PriorityHeap(cmp);
	better_priority_queue::updatable_priority_queue<int, double> PriorityHeap;
	PriorityHeap = better_priority_queue::updatable_priority_queue<int, double>();
	//policy
	vector<int> policy(S, 0);

	// keep count of number of iterations
	int iterations = 0;
	bool upper_convergence_criteria = false;
	const double convergence_bound_precomputed = (epsilon * (1.0 - gamma)) / gamma;
	//const double convergence_bound_precomputed = 0.0005;
	
		double *V_current_iteration = V[0];

		// for all states in each iteration
		
		/*
		for (int s = 0; s < S; s++)
		{
			// TODO if non-negative rewards, 0 is a lower bound of the maximization. to be changed if we want negative rewards
			// V_current_iteration[s] = double(0);
			double oldV = V_current_iteration[s];
			// ranged for loop over all actions in the action set of state s
			for (auto a : A[s])
			{
				auto &[P_s_a, P_s_a_nonzero] = P[s][a];
					double cum_sum = double(0);
					int k = 0;
					for (int ks : P_s_a_nonzero)
				{
				// cum_sum += 	(V_one[s] * V_two[s]);
					cum_sum += (P_s_a[k] * V_current_iteration[ks]);
					k++;
					if (P_s_a[k]>0.1){
					if (predecessor[ks].size()==0)
						predecessor[ks].push_back(s);
					else if(predecessor[ks][predecessor[ks].size()-1]!=s)
						predecessor[ks].push_back(s);}
				}
				double R_s_a = R[s][a] + gamma *cum_sum ;
				if (R_s_a > V_current_iteration[s])
				{
					V_current_iteration[s]better_priority_queue::updatable_priority_queue<int, double> = R_s_a;
					policy[s] = a;
				}
			}
			PriorityHeap.push({oldV-V_current_iteration[s],s});
			reverseV[s]=oldV-V_current_iteration[s];
		}*/
		int s;
		double value;
		performIterationUPDprestep(S,A,R,P,gamma,V_current_iteration,PriorityHeap,policy,predecessor,reverseV);
		while (!PriorityHeap.empty()){
			s=PriorityHeap.top().key;
			//value=PriorityHeap.top().first;
			PriorityHeap.pop();
			//if(abs(value-reverseV[s])>convergence_bound_precomputed)//outdaded value in heap.
			//continue;

			double oldV = V_current_iteration[s];
			// ranged for loop over all actions in the action set of state s
			for (auto a : A[s])
			{
				auto &[P_s_a, P_s_a_nonzero] = P[s][a];
				double R_s_a = R[s][a] + gamma * sum_of_mult_nonzero_only(P_s_a, V_current_iteration, P_s_a_nonzero);
				if (R_s_a > V_current_iteration[s])
				{
					V_current_iteration[s] = R_s_a;
					policy[s] = a;
				}
			}
			if(abs(oldV-V_current_iteration[s])>convergence_bound_precomputed){
				PriorityHeap.push(s,V_current_iteration[s]-oldV);
				reverseV[s]=V_current_iteration[s]-oldV;
			}
			performIterationUPDPred(s,A,R,P,gamma,V_current_iteration,PriorityHeap,policy,predecessor,reverseV,convergence_bound_precomputed);
			/*
			for (auto sa: predecessor[s]){
				double oldV = V_current_iteration[sa];
			// ranged for loop over all actions in the action set of state s
			for (auto a : A[sa])
			{
				auto &[P_s_a, P_s_a_nonzero] = P[sa][a];
				double R_s_a = R[sa][a] + gamma * sum_of_mult_nonzero_only(P_s_a, V_current_iteration, P_s_a_nonzero);
				if (R_s_a > V_current_iteration[sa])
				{
					V_current_iteration[sa] = R_s_a;
					policy[sa] = a;
				}
			}
			if(abs(oldV-V_current_iteration[sa])>convergence_bound_precomputed){
				PriorityHeap.push({abs(oldV-V_current_iteration[sa]),sa});
				reverseV[sa]=abs(oldV-V_current_iteration[sa]);
			}
			}		*/	
		}

	vector<double> result(V[0], V[0] + S);
	V_type result_tuple = make_tuple(result, iterations, work_per_iteration, actions_eliminated);
/*
	std::cout << "V_star policy: ";
	for (int i: policy) {
		std::cout << i;
	}
	std::cout << std::endl;
*/
	// DEALLOCATE MEMORY
	for (int i = 0; i < 1; ++i)
	{
		delete[] V[i];
	}
	delete[] V;

	return result_tuple;
}




V_type value_iterationGSTM(S_type S, R_type R, A_type A, P_type P, double gamma, double epsilon, int D3)
{

	// Find the maximum reward in the reward table
	auto [r_star_min, r_star_max, r_star_values] = find_all_r_values(R);

	// 2. Improved Lower Bound
	double **V = new double *[1];
	for (int i = 0; i < 1; ++i)
	{
		V[i] = new double[S];
	}
	// int siz=sqrt(S-1)-2;
	// int Xmax=siz+2;
	// gamma=1;
	for (int s = 0; s < S; s++)
	{
		/*
		int x_curr=s%Xmax;
			int y_curr=s/Xmax;
		double x1= sqrt( pow( abs(x_curr-siz),2)+pow(abs(y_curr-siz),2));
		V[0][s] = -x1*5-10;*/
		if (D3 == 0)
			V[0][s] = (gamma / (1.0 - gamma)) * r_star_min + r_star_values[s];
		else
		{
			// int x_curr=s%Xmax;
			// int y_curr=s/Xmax;
			// double x1= sqrt( pow( abs(x_curr-siz),2)+pow(abs(y_curr-siz),2));
			V[0][s] = -500;
			// V[0][s] = -x1*5-10;
			gamma = 1;
		}
	}
	if (D3 != 0)
		V[0][S - 1] = 0;

	// record actions eliminated in each iteration, where a pair is (state, action)
	// push empty vector for 0-index. Iterations start with 1
	vector<vector<pair<int, int>>> actions_eliminated;
	actions_eliminated.push_back(vector<pair<int, int>>());

	// keep track of work done in each iteration in microseconds
	// start from iteration 1
	vector<microseconds> work_per_iteration(1);

	// keep count of number of iterations
	int iterations = 0;
	bool upper_convergence_criteria = false;
	// const double convergence_bound_precomputed = (epsilon * (1.0 - gamma)) / gamma;
	const double convergence_bound_precomputed = 0.0005;

	while (!upper_convergence_criteria)
	{
		upper_convergence_criteria = true;
		// Increment iteration counter i
		iterations++;
		// Record actions eliminated in this iteration over all states
		vector<pair<int, int>> actions_eliminated_in_iteration;

		// begin timing of this iteration
		auto start_of_iteration = high_resolution_clock::now();

		// If i is even, then (i & 1) is 0, and the one to change is V[0]
		double *V_current_iteration = V[0];

		// for all states in each iteration
		for (int s = 0; s < S; s++)
		{
			// TODO if non-negative rewards, 0 is a lower bound of the maximization. to be changed if we want negative rewards
			// V_current_iteration[s] = double(0);
			double oldV = V_current_iteration[s];
			// ranged for loop over all actions in the action set of state s
			for (auto a : A[s])
			{
				auto &[P_s_a, P_s_a_nonzero] = P[s][a];
				double R_s_a = R[s][a] + gamma * sum_of_mult_nonzero_only(P_s_a, V_current_iteration, P_s_a_nonzero);
				if (R_s_a > V_current_iteration[s])
				{
					V_current_iteration[s] = R_s_a;
				}
			}
			if (abs(oldV - V_current_iteration[s]) > convergence_bound_precomputed)
				upper_convergence_criteria = false;
		}

		// end timing of this iteration and record it in work vector
		auto end_of_iteration = high_resolution_clock::now();
		auto duration_of_iteration = duration_cast<microseconds>(end_of_iteration - start_of_iteration);
		work_per_iteration.push_back(duration_of_iteration);
		actions_eliminated.push_back(move(actions_eliminated_in_iteration));
	}
	vector<double> result(V[0], V[0] + S);
	V_type result_tuple = make_tuple(result, iterations, work_per_iteration, actions_eliminated);

	// DEALLOCATE MEMORY
	for (int i = 0; i < 1; ++i)
	{
		delete[] V[i];
	}
	delete[] V;

	return result_tuple;
}
