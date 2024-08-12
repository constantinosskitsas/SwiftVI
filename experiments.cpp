#include <algorithm>
#include <chrono>
#include <ctime>
#include <tuple>
#include <random>
#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <sstream>
#include <math.h>
#include <thread>
 #include <eigen3/Eigen/Dense> //apt-get install libeigen3-dev

#include "MDP_type_definitions.h"
#include "pretty_printing_MDP.h"
#include "MDP_generation.h"
#include "VI_algorithms_helper_methods.h"
#include "VI_algorithm.h"
#include "VIU_algorithm.h"
#include "BVI_algorithm.h"
#include "VIAE_algorithm.h"
#include "VIAE_algorithm_improved_bounds.h"
#include "VIAE_algorithm_old_bounds.h"
#include "VIAEH_algorithm.h"
#include "VIAEH_algorithm_no_pointers.h"
#include "VIAEH_algorithm_maxmin_heap.h"
#include "VIAEH_algorithm_lower_bound_approx.h"
#include "VIAEH_algorithm_lazy_update.h"
#include "VIAEH_algorithm_set.h"
#include "VIH_algorithm.h"
#include "experiments.h"
#include "stopping_criteria_plot.h"
#include "top_action_change_plot.h"
#include "VIH_actions_touched.h"
#include "BAO_algorithm.h"
#include "VIH_algorithm_custom_heaps.h"

using namespace std;
using namespace std::chrono;

void runUCRLgamma(MDP_type &mdp, int S, int _nA)
{
	std::default_random_engine generator;
	generator.seed(1337);

	int nS = S;
	int nA = _nA;
	double gamma = 0.99;
	double epsilon = 0.1;
	double delta = 0.05;
	//nt m = 1;

	int T = 1000000;
	int k = 0;
	int t = 0;
	//T=10000;
	int reps = 5; // plotting replicates
	bool make_plots = true;

	MDP_type MDP = mdp; //ErgodicRiverSwim(5); // GridWorld(5, 5, 1337);
	R_type R = get<0>(MDP);
	A_type A = get<1>(MDP);
	P_type P = get<2>(MDP);

	V_type V_star_return = value_iterationGS(nS, R, A, P, gamma, epsilon);
	vector<double> V_star = get<0>(V_star_return);
	
	std::vector<std::vector<double>> v_opt(reps, std::vector<double>(T, 0.0));
	std::vector<std::vector<double>> v_pol(reps, std::vector<double>(T, 0.0));

	std::vector<std::vector<double>> v_opt_e(reps, std::vector<double>(T, 0.0));
	std::vector<std::vector<double>> v_pol_e(reps, std::vector<double>(T, 0.0));

	double reward = 0;
	UCLR MB = UCLR(nS, nA, gamma, epsilon, delta);
	vector<double> step_vector(nS,0.0); 


	for (int rep = 0; rep < reps; rep++)
	{
		// Init game
		int state = 0;
		int prev_state;
		MB.reset(state);

		//Use known R
		/*for (int s = 0; s < nS; s++) {
			for (int a = 0; a < nA; a++) {
				MB.Rsa[s][a] = R[s][a]; 
			} 

		}*/
		t = 0;
		k = 0;
		reward = 0;
		vector<int> _policy(state, 0);
		int action;
		int prev_action;
		std::vector<int> policy;

		// Run game
		while (t < T)
		{
			
			MB.confidence();
			policy = MB.EVI();
			do {
				if (t%10000 == 0) {
					std::cout << "UCRL_GAMMA " << t << std::endl; 
				}
				//Act
				action = policy[state];

				reward = R[state][action]; 
				
				//We update reward trackers right away, as they are not used before next EVI.
				MB.Rsa[state][action] += reward;  

				auto &[P_s_a, P_s_a_nonzero] = P[state][action];

				prev_state = state;
				prev_action = action;


				/*##################### TRACKING ####################*/
				if (make_plots) {
					//Get V for current policy
					Eigen::MatrixXd P_pi(nS, nS);
					Eigen::VectorXd R_pi(nS);
					//P_pi.reserve(nS * nS);
					//R_pi.reserve(nS);

					for (int s = 0; s < nS; s++) {

						std::vector<double> p_row(nS,0.0);
						auto &[_P_s_a, _P_s_a_nonzero] = P[s][policy[s]];
						for (int i = 0; i < _P_s_a_nonzero.size(); i++) {
							p_row[_P_s_a_nonzero[i]] = _P_s_a[i];
						}
						P_pi.row(s) = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(p_row.data(),nS);
						R_pi(s) = R[s][policy[s]]; //Assume R map is full cover
					}
					Eigen::MatrixXd I = Eigen::MatrixXd::Identity(nS, nS);
					Eigen::VectorXd V_pol = (I - P_pi * gamma).inverse() * R_pi;
					
					v_opt[rep][t]=V_star[state];
					v_pol[rep][t]=V_pol[state];

					v_opt_e[rep][t]=accumulate(V_star.begin(), V_star.end(), 0.0)/nS;
					v_pol_e[rep][t]=accumulate(V_pol.begin(), V_pol.end(), 0.0)/nS;  
				}
				/*##################### END TRACKING ####################*/


				std::discrete_distribution<int> distribution(P_s_a.begin(), P_s_a.end());
				state = P_s_a_nonzero[distribution(generator)];
				
				MB.vsa[prev_state][prev_action] += 1;
				MB.vsas[prev_state][prev_action][state] += 1;
				t += 1;

				if (t%10000 == 0) {
					//MB.end_act(prev_state, prev_action, true);  
				}
			} while (!MB.end_act(prev_state, prev_action, false) && t < T);  //repeat end condition

			MB.update(prev_state, prev_action);

			// Delay
			for (int j = 0; j < MB.H; j++) {
				if (t == T) {
					break;
				}
				/*if (j==0) {
					std::cout << "UCRL_GAMMA ACTING" << t << "  H = " << MB.H << std::endl;
				}*/
				if (t%10000 == 0) {
					std::cout << "UCRL_GAMMA " << t << std::endl; 
				}
				action = policy[state];

				reward = R[state][action];

				//We update reward trackers right away, as they are not used before next EVI.
				MB.Rsa[state][action] += reward;  
				auto &[P_s_a, P_s_a_nonzero] = P[state][action];

				prev_state = state;
				prev_action = action;


				/*##################### TRACKING ####################*/
				if (make_plots) {
					//Get V for current policy
					Eigen::MatrixXd P_pi(nS, nS);
					Eigen::VectorXd R_pi(nS);
					//P_pi.reserve(nS * nS);
					//R_pi.reserve(nS);

					for (int s = 0; s < nS; s++) {

						std::vector<double> p_row(nS,0.0);
						auto &[_P_s_a, _P_s_a_nonzero] = P[s][policy[s]];
						for (int i = 0; i < _P_s_a_nonzero.size(); i++) {
							p_row[_P_s_a_nonzero[i]] = _P_s_a[i];
						}
						P_pi.row(s) = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(p_row.data(),nS);
						R_pi(s) = R[s][policy[s]]; //Assume R map is full cover
					}
					Eigen::MatrixXd I = Eigen::MatrixXd::Identity(nS, nS);
					Eigen::VectorXd V_pol = (I - P_pi * gamma).inverse() * R_pi;
					
					v_opt[rep][t]=V_star[state];
					v_pol[rep][t]=V_pol[state];

					v_opt_e[rep][t]=accumulate(V_star.begin(), V_star.end(), 0.0)/nS;
					v_pol_e[rep][t]=accumulate(V_pol.begin(), V_pol.end(), 0.0)/nS;  
				}
				/*##################### END TRACKING ####################*/


				std::discrete_distribution<int> distribution(P_s_a.begin(), P_s_a.end());
				state = P_s_a_nonzero[distribution(generator)];

				MB.vsa[prev_state][prev_action] += 1;
				MB.vsas[prev_state][prev_action][state] += 1;
				t += 1;

			}	
			
			k += 1; //Tracks episodes.
			_policy=policy;
			
		}

		std::cout << "UCRL_Gamma policy:  ";
		for (int i: _policy) {
			std::cout << i;
		}	
		 std::cout << std::endl;
	}
	if (make_plots) {
		//Export tracked data for plotting
		std::stringstream filename;
		filename << "pyplotfiles/ucrlg_v_opt_" << nS << "_" << nA <<".txt";
		ofstream topython;

		topython.open(filename.str());
		if (topython.is_open())
		{
			for (int r = 0; r < reps; r++) {
				topython << v_opt[r][0];
				for (int t = 1; t < T; t++) {
					topython << " " << v_opt[r][t];
				}
				topython << std::endl;
			}
			topython.close();
		}
		else
		{
			printf("opened file: fail\n");
		}
		std::stringstream filename1;
		filename1 << "pyplotfiles/ucrlg_v_pol_" << nS << "_" << nA <<".txt";
		topython.open(filename1.str());
		if (topython.is_open())
		{
			for (int r = 0; r < reps; r++) {
				topython << v_pol[r][0];
				for (int t = 1; t < T; t++) {
					topython << " " << v_pol[r][t];
				}
				topython << std::endl;
			}
			topython.close();
		}
		else
		{
			printf("opened file: fail\n");
		}
		std::stringstream filename2;
		filename2 << "pyplotfiles/ucrlg_v_pol_e" << nS << "_" << nA <<".txt";

		topython.open(filename2.str());
		if (topython.is_open())
		{
			for (int r = 0; r < reps; r++) {
				topython << v_pol_e[r][0];
				for (int t = 1; t < T; t++) {
					topython << " " << v_pol_e[r][t];
				}
				topython << std::endl;
			}
			topython.close();
		}
		else
		{
			printf("opened file: fail\n");
		}

		std::stringstream filename3;
		filename3 << "pyplotfiles/ucrlg_v_opt_e" << nS << "_" << nA <<".txt";
		//std::cout << filename3.str()<< std::endl;
		topython.open(filename3.str());
		if (topython.is_open())
		{
			for (int r = 0; r < reps; r++) {
				topython << v_opt_e[r][0];
				for (int t = 1; t < T; t++) {
					topython << " " << v_opt_e[r][t];
				}
				topython << std::endl;
			}
			topython.close();
		}
		else
		{
			printf("opened file: fail\n");
		}
	}
}


//I just copied the BAO function, feel bad doing this should we merge them?
//However if we keep only BAO and VIH then its kind of okay.
void runBaoMBIE(MDP_type &mdp, int S, int _nA)
{
	std::default_random_engine generator;
	generator.seed(1337);

	int nS = S;
	int nA = _nA;
	double gamma = 0.99;
	double epsilon = 0.1;
	double delta = 0.05;
	int m = 1;

	int T = 1000;
	//T=10000;
	int reps = 1; // replicates
	bool make_plots = false;

	MDP_type MDP = mdp; //ErgodicRiverSwim(5); // GridWorld(5, 5, 1337);
	R_type R = get<0>(MDP);
	A_type A = get<1>(MDP);
	P_type P = get<2>(MDP);

	V_type V_star_return = value_iterationGS(nS, R, A, P, gamma, epsilon);
	vector<double> V_star = get<0>(V_star_return);
	
	std::vector<std::vector<double>> v_opt(reps, std::vector<double>(T, 0.0));
	std::vector<std::vector<double>> v_pol(reps, std::vector<double>(T, 0.0));

	std::vector<std::vector<double>> v_opt_e(reps, std::vector<double>(T, 0.0));
	std::vector<std::vector<double>> v_pol_e(reps, std::vector<double>(T, 0.0));

	double reward = 0;
	MBIE MB = MBIE(nS, nA, gamma, epsilon, delta, m);
	vector<double> step_vector(nS,0.0); 
	for (int rep = 0; rep < reps; rep++)
	{
		// Init game
		int state = 0;
		MB.reset(state);
		reward = 0;
		vector<int> _policy(state, 0);
		int action;
		std::vector<int> policy;
		// Run game
		for (int t = 0; t < T; t++)
		{
			if (t%10000 == 0) {
				std::cout << "MBIEBAO " << t << std::endl;
			}
			
			if (t%1000 == 0) {
				/*if (t < T/12) {
					std::tie(action, policy) = MB.play(state, reward);
				} else {*/
					std::tie(action, policy) = MB.playbao(state, reward);
				//}
			} else {
				std::tie(action, policy) = MB.update_vals(state, reward);
			} 
			/*else {
				std::tie(action, policy) = MB.playswift(state, reward);
			}*/
			//std::cout << t << std::endl;
			// Run MBIE step
			//auto [action, policy] = MB.playswift(state, reward);
			// Get reward and next step from MDP
			reward = R[state][action];
			
			auto &[P_s_a, P_s_a_nonzero] = P[state][action];
/*
			if (make_plots) {
				//Get V for current policy
				Eigen::MatrixXd P_pi(nS, nS);
				Eigen::VectorXd R_pi(nS);
				//P_pi.reserve(nS * nS);
				//R_pi.reserve(nS);

				for (int s = 0; s < nS; s++) {

					std::vector<double> p_row(nS,0.0);
					auto &[_P_s_a, _P_s_a_nonzero] = P[s][policy[s]];
					for (int i = 0; i < _P_s_a_nonzero.size(); i++) {
						p_row[_P_s_a_nonzero[i]] = _P_s_a[i];
					}
					P_pi.row(s) = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(p_row.data(),nS);
					R_pi(s) = R[s][policy[s]]; //Assume R map is full cover
				}
				Eigen::MatrixXd I = Eigen::MatrixXd::Identity(nS, nS);
				Eigen::VectorXd V_pol = (I - P_pi * gamma).inverse() * R_pi;
				
				v_opt[rep][t]=V_star[state];
				v_pol[rep][t]=V_pol[state];

				v_opt_e[rep][t]=accumulate(V_star.begin(), V_star.end(), 0.0)/nS;
				v_pol_e[rep][t]=accumulate(V_pol.begin(), V_pol.end(), 0.0)/nS;  
			}*/


			std::discrete_distribution<int> distribution(P_s_a.begin(), P_s_a.end());
			state = P_s_a_nonzero[distribution(generator)];

			_policy=policy;
		}

		std::cout << "BaoMBIE policy:  ";
		for (int i: _policy) {
			std::cout << i;
		}	
		 std::cout << std::endl;
	}
	if (make_plots) {
		//Export tracked data for plotting
		std::stringstream filename;
		filename << "pyplotfiles/baombie_v_opt_" << nS << "_" << nA <<".txt";
		ofstream topython;

		topython.open(filename.str());
		if (topython.is_open())
		{
			for (int r = 0; r < reps; r++) {
				topython << v_opt[r][0];
				for (int t = 1; t < T; t++) {
					topython << " " << v_opt[r][t];
				}
				topython << std::endl;
			}
			topython.close();
		}
		else
		{
			printf("opened file: fail\n");
		}
		std::stringstream filename1;
		filename1 << "pyplotfiles/baombie_v_pol_" << nS << "_" << nA <<".txt";
		topython.open(filename1.str());
		if (topython.is_open())
		{
			for (int r = 0; r < reps; r++) {
				topython << v_pol[r][0];
				for (int t = 1; t < T; t++) {
					topython << " " << v_pol[r][t];
				}
				topython << std::endl;
			}
			topython.close();
		}
		else
		{
			printf("opened file: fail\n");
		}
		std::stringstream filename2;
		filename2 << "pyplotfiles/baombie_v_pol_e" << nS << "_" << nA <<".txt";

		topython.open(filename2.str());
		if (topython.is_open())
		{
			for (int r = 0; r < reps; r++) {
				topython << v_pol_e[r][0];
				for (int t = 1; t < T; t++) {
					topython << " " << v_pol_e[r][t];
				}
				topython << std::endl;
			}
			topython.close();
		}
		else
		{
			printf("opened file: fail\n");
		}

		std::stringstream filename3;
		filename3 << "pyplotfiles/baombie_v_opt_e" << nS << "_" << nA <<".txt";
		//std::cout << filename3.str()<< std::endl;
		topython.open(filename3.str());
		if (topython.is_open())
		{
			for (int r = 0; r < reps; r++) {
				topython << v_opt_e[r][0];
				for (int t = 1; t < T; t++) {
					topython << " " << v_opt_e[r][t];
				}
				topython << std::endl;
			}
			topython.close();
		}
		else
		{
			printf("opened file: fail\n");
		}
	}
}



void runSwiftMBIE(MDP_type &mdp, int S, int _nA)
{
	std::default_random_engine generator;
	generator.seed(1337);

	int nS = S;
	int nA = _nA;
	double gamma = 0.99;
	double epsilon = 0.1;
	double delta = 0.05;
	int m = 1;

	int T = 1000000;
	//T=10000;
	int reps = 5; // replicates
	bool make_plots = true;

	MDP_type MDP = mdp; //ErgodicRiverSwim(5); // GridWorld(5, 5, 1337);
	R_type R = get<0>(MDP);
	A_type A = get<1>(MDP);
	P_type P = get<2>(MDP);

	V_type V_star_return = value_iterationGS(nS, R, A, P, gamma, epsilon);
	vector<double> V_star = get<0>(V_star_return);
	
	std::vector<std::vector<double>> v_opt(reps, std::vector<double>(T, 0.0));
	std::vector<std::vector<double>> v_pol(reps, std::vector<double>(T, 0.0));

	std::vector<std::vector<double>> v_opt_e(reps, std::vector<double>(T, 0.0));
	std::vector<std::vector<double>> v_pol_e(reps, std::vector<double>(T, 0.0));

	double reward = 0;
	MBIE MB = MBIE(nS, nA, gamma, epsilon, delta, m);
	vector<double> step_vector(nS,0.0); 
	for (int rep = 0; rep < reps; rep++)
	{
		// Init game
		int state = 0;
		MB.reset(state);
		reward = 0;
		vector<int> _policy(state, 0);
		int action;
		std::vector<int> policy;
		// Run game
		for (int t = 0; t < T; t++)
		{
			if (t%10000 == 0) {
				std::cout << "MBIEH " << t << std::endl;
			}
			
			if (t%1000 == 0) {
				/*if (t < T/12) {
					std::tie(action, policy) = MB.play(state, reward);
				} else {*/
					std::tie(action, policy) = MB.playswift(state, reward);
				//}
			} else {
				std::tie(action, policy) = MB.update_vals(state, reward);
			}
			//std::cout << t << std::endl;
			// Run MBIE step
			//auto [action, policy] = MB.playswift(state, reward);
			// Get reward and next step from MDP
			reward = R[state][action];
			
			auto &[P_s_a, P_s_a_nonzero] = P[state][action];
			
			if (make_plots) {
				//Get V for current policy
				Eigen::MatrixXd P_pi(nS, nS);
				Eigen::VectorXd R_pi(nS);
				//P_pi.reserve(nS * nS);
				//R_pi.reserve(nS);

				for (int s = 0; s < nS; s++) {

					std::vector<double> p_row(nS,0.0);
					auto &[_P_s_a, _P_s_a_nonzero] = P[s][policy[s]];
					for (int i = 0; i < _P_s_a_nonzero.size(); i++) {
						p_row[_P_s_a_nonzero[i]] = _P_s_a[i];
					}
					P_pi.row(s) = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(p_row.data(),nS);
					R_pi(s) = R[s][policy[s]]; //Assume R map is full cover
				}
				Eigen::MatrixXd I = Eigen::MatrixXd::Identity(nS, nS);
				Eigen::VectorXd V_pol = (I - P_pi * gamma).inverse() * R_pi;
				
				v_opt[rep][t]=V_star[state];
				v_pol[rep][t]=V_pol[state];

				v_opt_e[rep][t]=accumulate(V_star.begin(), V_star.end(), 0.0)/nS;
				v_pol_e[rep][t]=accumulate(V_pol.begin(), V_pol.end(), 0.0)/nS;  
			}


			std::discrete_distribution<int> distribution(P_s_a.begin(), P_s_a.end());
			state = P_s_a_nonzero[distribution(generator)];

			_policy=policy;
		}

		std::cout << "SwiftMBIE policy:  ";
		for (int i: _policy) {
			std::cout << i;
		}	
		 std::cout << std::endl;
	}
	if (make_plots) {
		//Export tracked data for plotting
		std::stringstream filename;
		filename << "pyplotfiles/swiftmbie_v_opt_" << nS << "_" << nA <<".txt";
		ofstream topython;

		topython.open(filename.str());
		if (topython.is_open())
		{
			for (int r = 0; r < reps; r++) {
				topython << v_opt[r][0];
				for (int t = 1; t < T; t++) {
					topython << " " << v_opt[r][t];
				}
				topython << std::endl;
			}
			topython.close();
		}
		else
		{
			printf("opened file: fail\n");
		}
		std::stringstream filename1;
		filename1 << "pyplotfiles/swiftmbie_v_pol_" << nS << "_" << nA <<".txt";
		topython.open(filename1.str());
		if (topython.is_open())
		{
			for (int r = 0; r < reps; r++) {
				topython << v_pol[r][0];
				for (int t = 1; t < T; t++) {
					topython << " " << v_pol[r][t];
				}
				topython << std::endl;
			}
			topython.close();
		}
		else
		{
			printf("opened file: fail\n");
		}
		std::stringstream filename2;
		filename2 << "pyplotfiles/swiftmbie_v_pol_e" << nS << "_" << nA <<".txt";

		topython.open(filename2.str());
		if (topython.is_open())
		{
			for (int r = 0; r < reps; r++) {
				topython << v_pol_e[r][0];
				for (int t = 1; t < T; t++) {
					topython << " " << v_pol_e[r][t];
				}
				topython << std::endl;
			}
			topython.close();
		}
		else
		{
			printf("opened file: fail\n");
		}

		std::stringstream filename3;
		filename3 << "pyplotfiles/swiftmbie_v_opt_e" << nS << "_" << nA <<".txt";
		//std::cout << filename3.str()<< std::endl;
		topython.open(filename3.str());
		if (topython.is_open())
		{
			for (int r = 0; r < reps; r++) {
				topython << v_opt_e[r][0];
				for (int t = 1; t < T; t++) {
					topython << " " << v_opt_e[r][t];
				}
				topython << std::endl;
			}
			topython.close();
		}
		else
		{
			printf("opened file: fail\n");
		}
	}
}

void runMBIE(MDP_type &mdp, int S, int _nA)
{
	std::default_random_engine generator;
	generator.seed(1337);

	int nS = S;
	int nA = _nA;
	double gamma = 0.99;
	double epsilon = 0.1;
	double delta = 0.05;
	int m = 1;

	int T = 1000;
	//T=10000;
	int reps = 1; // replicates
	bool make_plots = false;
	MDP_type MDP = mdp; //ErgodicRiverSwim(5); // GridWorld(5, 5, 1337);
	R_type R = get<0>(MDP);
	A_type A = get<1>(MDP);
	P_type P = get<2>(MDP);

	V_type V_star_return = value_iterationGS(nS, R, A, P, gamma, epsilon);
	vector<double> V_star = get<0>(V_star_return);
	
	std::vector<std::vector<double>> v_opt(reps, std::vector<double>(T, 0.0));
	std::vector<std::vector<double>> v_pol(reps, std::vector<double>(T, 0.0));

	std::vector<std::vector<double>> v_opt_e(reps, std::vector<double>(T, 0.0));
	std::vector<std::vector<double>> v_pol_e(reps, std::vector<double>(T, 0.0));


	double reward = 0;
	MBIE MB = MBIE(nS, nA, gamma, epsilon, delta, m);
	//MBIE MBswift = MBIE(nS, nA, gamma, epsilon, delta, m);
	for (int rep = 0; rep < reps; rep++)
	{
		// Init game
		int state = 0;
		int swiftstate = 0;
		MB.reset(state);
		//MBswift.reset(swiftstate);
		reward = 0;
		//swiftreward = 0;

		vector<int> _policy(state, 0);
		vector<double> step_vector(nS,0.0);
		int action;
		std::vector<int> policy;
		// Run game
		for (int t = 0; t < T; t++)
		{
			if (t%10000 == 0) {
				std::cout << "MBIE " << t << std::endl;
			}
			//std::cout << t << std::endl;
			// Run MBIE step
			
			if (t%1000 == 0) {
				std::tie(action, policy) = MB.play(state, reward);
			} else {
				std::tie(action, policy) = MB.update_vals(state, reward);
			}
			//auto [action, policy] = MB.play(state, reward);
			// Get reward and next step from MDP
			reward = R[state][action];

			auto &[P_s_a, P_s_a_nonzero] = P[state][action];
			/*
			if (make_plots) {
				//Get V for current policy
				Eigen::MatrixXd P_pi(nS, nS);
				Eigen::VectorXd R_pi(nS);
				for (int s = 0; s < nS; s++) {

					std::vector<double> p_row(nS,0.0);
					auto &[_P_s_a, _P_s_a_nonzero] = P[s][policy[s]];
					for (int i = 0; i < _P_s_a_nonzero.size(); i++) {
						p_row[_P_s_a_nonzero[i]] = _P_s_a[i];
					}
					P_pi.row(s) = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(p_row.data(),nS);
					R_pi(s) = R[s][policy[s]]; //Assume R map is full cover
				}
				Eigen::MatrixXd I = Eigen::MatrixXd::Identity(nS, nS);
				Eigen::VectorXd V_pol = (I - P_pi * gamma).inverse() * R_pi;
				
				v_opt[rep][t]=V_star[state];
				v_pol[rep][t]=V_pol[state];

				v_opt_e[rep][t]=accumulate(V_star.begin(), V_star.end(), 0.0)/nS;
				v_pol_e[rep][t]=accumulate(V_pol.begin(), V_pol.end(), 0.0)/nS;  
			}*/

			std::discrete_distribution<int> distribution(P_s_a.begin(), P_s_a.end());
			state = P_s_a_nonzero[distribution(generator)];
			_policy=policy;
		}
		std::cout << "MBIE policy: ";
		for (int i: _policy) {
			std::cout << i;
		}	
		std::cout << std::endl;
		//std::cout << "\n##################################\n########################" << std::endl;
	}
	if (make_plots) {
		//Export tracked data for plotting
		std::stringstream filename;
		filename << "pyplotfiles//mbie_v_opt_" << nS << "_" << nA <<".txt";
		ofstream topython;

		topython.open(filename.str());
		if (topython.is_open())
		{
			for (int r = 0; r < reps; r++) {
				topython << v_opt[r][0];
				for (int t = 1; t < T; t++) {
					topython << " " << v_opt[r][t];
				}
				topython << std::endl;
			}
			topython.close();
		}
		else
		{
			printf("opened file: fail\n");
		}
		std::stringstream filename1;
		filename1 << "pyplotfiles//mbie_v_pol_" << nS << "_" << nA <<".txt";
		topython.open(filename1.str());
		if (topython.is_open())
		{
			for (int r = 0; r < reps; r++) {
				topython << v_pol[r][0];
				for (int t = 1; t < T; t++) {
					topython << " " << v_pol[r][t];
				}
				topython << std::endl;
			}
			topython.close();
		}
		else
		{
			printf("opened file: fail\n");
		}
		std::stringstream filename2;
		filename2 << "pyplotfiles//mbie_v_pol_e" << nS << "_" << nA <<".txt";
		
		topython.open(filename2.str());
		if (topython.is_open())
		{
			for (int r = 0; r < reps; r++) {
				topython << v_pol_e[r][0];
				for (int t = 1; t < T; t++) {
					topython << " " << v_pol_e[r][t];
				}
				topython << std::endl;
			}
			topython.close();
		}
		else
		{
			printf("opened file: fail\n");
		}

		std::stringstream filename3;
		filename3 << "pyplotfiles/mbie_v_opt_e" << nS << "_" << nA <<".txt";
		//std::cout << filename3.str()<< std::endl;
		topython.open(filename3.str());
		if (topython.is_open())
		{
			for (int r = 0; r < reps; r++) {
				topython << v_opt_e[r][0];
				for (int t = 1; t < T; t++) {
					topython << " " << v_opt_e[r][t];
				}
				topython << std::endl;
			}
			topython.close();
		}
		else
		{
			printf("opened file: fail\n");
		}
	}
	
}
void RLRS(string filename, int expnum, int States, int Actions, int SS, int StartP, int endP, int IncP, double epsilon, double gamma, double upper_reward, double non_zero_transition)
{
	MDP_type MDP;
	ostringstream string_stream;
	ostringstream avgstring_stream;
	ofstream output_stream;
	ofstream avgoutput_stream;
	string file_name_VI = "Skitsas//RLRS.txt";
	file_name_VI = "Skitsas//RLRS_S500_100A_50SS_E-00001_OF_ITER46.txt";
	string file_name_VIAVG = "Skitsas//avgRLRS.txt";
	file_name_VIAVG = "Skitsas//avgRLRS_S500_100A_50SS_E-00001_OF_ITER46.txt";
	string_stream << "Experiment ID: " << expnum << endl;
	avgstring_stream << "Experiment ID" << expnum << endl;
	string_stream << "MBVI MBVIH MBBAO" << endl;
	avgstring_stream << "MBVI MBVIH MBBAO" << endl;
	int repetitions = 1;
	int siIter = ((endP - StartP) / IncP) + 1;
	// int siIter= 5;
	std::vector<std::vector<float>> VI(4,std::vector<float>(siIter, 0));
	int k = 0;
	int S;
	for (int i = 0; i < 4; i++)
		for (int j = 0; j < siIter; j++)
			VI[i][j] = 0;

	for (int iters = 0; iters < repetitions; iters++)
	{
		k = 0;

		std::vector<std::thread> threads;
		for (int ite = StartP; ite <= endP; ite = ite + IncP)
		{
			//threads.push_back(std::thread([&,ite, k]() mutable { 
				std::cout <<"Repetition: " << iters <<"/"<< repetitions << "     Size: " << ite << "/" << endP << std::endl;
				int seed = time(0);
				/**MDP CHANGES**/
				int nA = 4;
				int FB = 1; //Gridworld block extension
				MDP = FixedGridWorld(ite,FB,true);//ErgodicRiverSwim(ite);//generate_random_MDP_normal_distributed_rewards(ite, nA, 0.5, 10, seed, 0.5, 0.05);//GridWorld(ite,ite,123, 0);//ErgodicRiverSwim(ite);//ErgodicRiverSwim(ite);//GridWorld(ite,ite,123, 0); //Maze(ite,ite,123);// (ite);
				S = ite*ite-5-FB*4; 
				/**MDP CHANGES**/
				R_type R = get<0>(MDP);
				A_type A = get<1>(MDP);
				P_type P = get<2>(MDP);
				int counter = 0;

				A_type A1 = copy_A(A);
				auto start_VI = high_resolution_clock::now();
				runMBIE(MDP, S, nA);
				auto stop_VI = high_resolution_clock::now();
				auto duration_VI = duration_cast<milliseconds>(stop_VI - start_VI);

				A_type A2 = copy_A(A);
				auto start_VIH = high_resolution_clock::now();
				runSwiftMBIE(MDP, S, nA);
				auto stop_VIH = high_resolution_clock::now();
				auto duration_VIH = duration_cast<milliseconds>(stop_VIH - start_VIH);

				A_type A3 = copy_A(A);
				auto start_BAO = high_resolution_clock::now();
				runBaoMBIE(MDP, S, nA);
				auto stop_BAO = high_resolution_clock::now();
				auto duration_BAO = duration_cast<milliseconds>(stop_BAO - start_BAO);

				A_type A4 = copy_A(A);
				auto start_UCRL_G = high_resolution_clock::now();
				runUCRLgamma(MDP, S, nA);
				auto stop_UCRL_G = high_resolution_clock::now();
				auto duration_UCRL_G= duration_cast<milliseconds>(stop_UCRL_G - start_UCRL_G);
				//std::cout << k << "  " << ite << std::endl;
				VI[0][k] += duration_VI.count();
				//std::cout << k << std::endl;
				VI[1][k] += duration_VIH.count();
				VI[2][k] += duration_BAO.count();
				VI[3][k] += duration_UCRL_G.count();
				//std::cout << k << std::endl;
				//std::cout << VI[0][k] << std::endl;
				string_stream << duration_VI.count() << " " << duration_VIH.count()<< " " << duration_BAO.count() << " " << duration_UCRL_G.count() <<endl;
			//}));
			k++;
			//std::cout << VI[0][k] << std::endl;
		}
		/*for (auto& th : threads) {
			th.join();
		}*/
		//std::cout << VI[0][k] << std::endl;
	}
	for (int k = 0; k < siIter; k++)
	{
		avgstring_stream << VI[0][k] / repetitions << " " << VI[1][k] / repetitions << " " << VI[2][k] / repetitions << " " << VI[3][k] / repetitions  <<  endl;
	}
	// WRITE ALL DATA TO THEIR RESPECTVIE FILES
	write_stringstream_to_file(string_stream, output_stream, file_name_VI);
	write_stringstream_to_file(avgstring_stream, avgoutput_stream, file_name_VIAVG);
}
void GSTM(string filename, int expnum, int States, int Actions, int SS, int StartP, int endP, int IncP, double epsilon, double gamma, double upper_reward, double non_zero_transition)
{
	MDP_type MDP;
	ostringstream string_stream;
	ostringstream avgstring_stream;
	ofstream output_stream;
	ofstream avgoutput_stream;

	// set the name of the file to write to
	string file_name_VI = "Skitsas//Ter_Maze2D.txt";
	string file_name_VIAVG = "Skitsas//avgTer_Maze2D.txt";
	if (expnum == 10)
	{
		file_name_VI = "Skitsas//Ter_Maze3D.txt";
		file_name_VIAVG = "Skitsas//avgTer_Maze3D.txt";
	}
	string_stream << "Experiment ID: " << expnum << endl;
	avgstring_stream << "Experiment ID" << expnum << endl;
	string_stream << "VI UVI VIH BVI VIAE VIAEH VIAEHLB BAO" << endl;
	avgstring_stream << "VI UVI VIH BVI VIAE VIAEH VIAEHLB BAO" << endl;

	double action_prob = 1.0;
	// A_num=100;
	// write meta data to all stringstreams as first in their respective files

	int repetitions = 10;
	int siIter = ((endP - StartP) / IncP) + 1;
	float VI[10][siIter];
	int k = 0;
	int S;
	for (int i = 0; i < 10; i++)
		for (int j = 0; j < siIter; j++)
			VI[i][j] = 0;
	for (int iters = 0; iters < repetitions; iters++)
	{
		k = 0;
		for (int ite = StartP; ite <= endP; ite = ite + IncP)
		{
			// printf("Beginning iteration %d  S2,  %d, A, S %d = %d\n",iters, S2,A_num,S);
			// auto MDP ;
			// GENERATE THE MDP
			int seed = time(0);
			// auto MDP = generate_random_MDP_exponential_distributed_rewards(S, A_num, 1.0, S, 0.02, seed);
			if (expnum == 9)
			{
				MDP = Maze(ite, ite, seed);
				States = ite * ite + 1;
				S = States;
			}
			else if (expnum == 10)
			{
				MDP = Maze3d(ite, ite, ite, seed);
				States = ite * ite * ite + 1;
				S = States;
			}
			else if (expnum == 11)
			{
				ite = 3;
				MDP = GridWorld(ite, ite, seed, 0);
				States = ite * ite; //+ 1;
				S = States;
			}
			// auto MDP = generate_random_MDP_with_variable_parameters(S, A_num, action_prob, non_zero_transition, upper_reward, seed);
			R_type R = get<0>(MDP);
			A_type A = get<1>(MDP);
			P_type P = get<2>(MDP);
			int counter = 0;
			for (int i = 0; i < S; i++)
			{
				cout << "State " << i << endl;
				for (auto a : A[i])
				{
					cout << "Action " << a << endl;
					auto &[P_s_a, P_s_a_nonzero] = P[i][a];
					int k = 0;
					for (int s : P_s_a_nonzero)
					{
						cout << "SS state :" << s << " Prob :" << P_s_a[k] << endl;
						k++;
					}
				}
			}
			return;
			A_type A1 = copy_A(A);
			auto start_VI = high_resolution_clock::now();
			V_type V_approx_solution_tuple = value_iterationGSTM(States, R, A1, P, gamma, epsilon, expnum);
			auto stop_VI = high_resolution_clock::now();
			auto duration_VI = duration_cast<milliseconds>(stop_VI - start_VI);

			A_type A6 = copy_A(A);
			auto start_VIU = high_resolution_clock::now();
			V_type V_approx_solution_upper_tuple = value_iteration_upperGSTM(States, R, A6, P, gamma, epsilon, expnum);
			auto stop_VIU = high_resolution_clock::now();
			auto duration_VIU = duration_cast<milliseconds>(stop_VIU - start_VIU);

			A_type A2 = copy_A(A);
			auto start_VIH = high_resolution_clock::now();
			V_type V_heap_approx_tuple = value_iteration_with_heapGSTM(States, R, A2, P, gamma, epsilon, expnum);
			auto stop_VIH = high_resolution_clock::now();
			auto duration_VIH = duration_cast<milliseconds>(stop_VIH - start_VIH);

			// BVI
			A_type A3 = copy_A(A);
			auto start_BVI = high_resolution_clock::now();
			V_type V_bounded_approx_solution_tuple = bounded_value_iterationGSTM(States, R, A3, P, gamma, epsilon, expnum);
			auto stop_BVI = high_resolution_clock::now();
			auto duration_BVI = duration_cast<milliseconds>(stop_BVI - start_BVI);

			A_type A4 = copy_A(A);
			auto start_VIAE = high_resolution_clock::now();
			V_type V_AE_approx_solution_tuple = value_iteration_action_eliminationGSTM(States, R, A4, P, gamma, epsilon, expnum);
			auto stop_VIAE = high_resolution_clock::now();
			auto duration_VIAE = duration_cast<milliseconds>(stop_VIAE - start_VIAE);

			A_type A5 = copy_A(A);
			auto start_VIAEH = high_resolution_clock::now();
			V_type V_AE_H_approx_solution_tuple = value_iteration_action_elimination_heapsGSTM(States, R, A5, P, gamma, epsilon, expnum);
			auto stop_VIAEH = high_resolution_clock::now();
			auto duration_VIAEH = duration_cast<milliseconds>(stop_VIAEH - start_VIAEH);

			A_type A8 = copy_A(A);
			auto start_VIAEHL = high_resolution_clock::now();
			V_type VIAEHL_approx_solution_tuple = value_iteration_action_elimination_heaps_lower_bound_approxGSTM(States, R, A8, P, gamma, epsilon, expnum);
			auto stop_VIAEHL = high_resolution_clock::now();
			auto duration_VIAEHL = duration_cast<milliseconds>(stop_VIAEHL - start_VIAEHL);
			// BAO
			A_type A9 = copy_A(A);
			auto start_BAO = high_resolution_clock::now();
			V_type BAO_approx_solution_tuple = value_iteration_BAOGSTM(States, R, A9, P, gamma, epsilon, expnum);
			auto stop_BAO = high_resolution_clock::now();
			auto duration_BAO = duration_cast<milliseconds>(stop_BAO - start_BAO);

			vector<double> BAO_approx_solution = get<0>(BAO_approx_solution_tuple);
			vector<double> VIAEHL_approx_solution = get<0>(VIAEHL_approx_solution_tuple);
			vector<double> V_AE_H_approx_solution = get<0>(V_AE_H_approx_solution_tuple);
			vector<double> V_AE_approx_solution = get<0>(V_AE_approx_solution_tuple);
			vector<double> V_bounded_approx_solution = get<0>(V_bounded_approx_solution_tuple);
			vector<double> V_heap_approx = get<0>(V_heap_approx_tuple);
			vector<double> V_approx_solution = get<0>(V_approx_solution_tuple);
			vector<double> V_approx_solution_upper = get<0>(V_approx_solution_upper_tuple);
			if (abs_max_diff_vectors(V_approx_solution, BAO_approx_solution) > (2 * epsilon))
			{
				printf("DIFFERENCE1\n");
			}
			if (abs_max_diff_vectors(V_approx_solution, V_heap_approx) > (2 * epsilon))
			{
				printf("DIFFERENCE2\n");
			}

			if (abs_max_diff_vectors(V_approx_solution, V_AE_H_approx_solution) > (2 * epsilon))
			{
				printf("DIFFERENCE3\n");
			}
			if (abs_max_diff_vectors(V_approx_solution, V_approx_solution_upper) > (2 * epsilon))
			{
				printf("DIFFERENCE4\n");
			}
			if (abs_max_diff_vectors(V_approx_solution, V_bounded_approx_solution) > (2 * epsilon))
			{
				printf("DIFFERENCE5\n");
			}
			if (abs_max_diff_vectors(V_approx_solution, VIAEHL_approx_solution) > (2 * epsilon))
			{
				printf("DIFFERENCE6\n");
			}
			if (abs_max_diff_vectors(V_approx_solution, V_AE_approx_solution) > (2 * epsilon))
			{
				printf("DIFFERENCE7\n");
			}

			VI[0][k] += duration_VI.count();
			VI[1][k] += duration_VIU.count();
			VI[2][k] += duration_VIH.count();
			VI[3][k] += duration_BVI.count();
			VI[4][k] += duration_VIAE.count();
			VI[5][k] += duration_VIAEH.count();
			VI[6][k] += duration_VIAEHL.count();
			VI[7][k] += duration_BAO.count();
			string_stream << duration_VI.count() << " " << duration_VIU.count() << " " << duration_VIH.count() << " " << duration_BVI.count() << " ";
			string_stream << duration_VIAE.count() << " " << duration_VIAEH.count() << " " << duration_VIAEHL.count() << " " << duration_BAO.count() << endl;
			k++;
		}
	}
	for (int k = 0; k < siIter; k++)
	{
		avgstring_stream << VI[0][k] / repetitions << " " << VI[1][k] / repetitions << " " << VI[2][k] / repetitions << " " << VI[3][k] / repetitions << " " << VI[4][k] / repetitions << " " << VI[5][k] / repetitions << " ";
		avgstring_stream << VI[6][k] / repetitions << " " << VI[7][k] / repetitions << endl;
	}
	// WRITE ALL DATA TO THEIR RESPECTVIE FILES
	write_stringstream_to_file(string_stream, output_stream, file_name_VI);
	write_stringstream_to_file(avgstring_stream, avgoutput_stream, file_name_VIAVG);
}

void REXP(string filename, int expnum, int States, int Actions, int SS, int StartP, int endP, int IncP, double epsilon, double gamma, double upper_reward, double non_zero_transition)
{
	auto duration_VI = milliseconds(0);
	V_type V_approx_solution_tuple;
	auto duration_VIU = milliseconds(0);
	V_type V_approx_solution_upper_tuple;
	auto duration_BVI = milliseconds(0);
	V_type V_bounded_approx_solution_tuple;
	auto duration_VIAE = milliseconds(0);
	V_type V_AE_approx_solution_tuple;
	auto duration_VIAEH = milliseconds(0);
	V_type V_AE_H_approx_solution_tuple;
	auto duration_VIH = milliseconds(0);
	V_type V_heap_approx_tuple;
	auto duration_VIAEHL = milliseconds(0);
	V_type VIAEHL_approx_solution_tuple;
	auto duration_BAO = milliseconds(0);
	V_type BAO_approx_solution_tuple;
	int k = 0;
	int repetitions = 10;
	ostringstream string_stream;
	ostringstream avgstring_stream;
	ofstream output_stream;
	ofstream avgoutput_stream;
	string SE[9] = {"0", "0", "0", "S", "SBO", "A", "ABO", "SS", "SSBO"};
	int S_finishing_value = endP;
	double action_prob = 1.0;
	int NE = int((endP - StartP) / IncP) + 1;
	bool BO = false;
	if (expnum == 3 || expnum == 5 || expnum == 7)
	{
		BO = true;
	}
	float VI[10][NE];
	int seed = time(0);
	for (int i = 0; i < 10; i++)
		for (int j = 0; j < NE; j++)
			VI[i][j] = 0;
	MDP_type MDP;
	string file_name_VI = "Skitsas//RandomGraphs_50S_100A_50SS_OF" + SE[expnum] + ".txt";
	string file_name_VIAVG = "Skitsas//AVG_RandomGraphs_50S_100A_50SS_OF" + SE[expnum] + ".txt";
	string_stream << "Exp_ID: " << expnum << " States: " << States << " Actions: " << Actions << " S_states: " << SS << " Start_P: " << StartP << " endP: " << endP << " IncP: " << IncP << endl;
	avgstring_stream << "Exp_ID: " << expnum << " States: " << States << " Actions: " << Actions << " S_states: " << SS << " Start_P: " << StartP << " endP: " << endP << " IncP: " << IncP << endl;
	if (BO)
	{
		string_stream << "VI UVI BVI VIAE VIAEH ";
		avgstring_stream << "VI UVI BVI VIAE VIAEH ";
	}

	string_stream << "VIH VIAEHLB BAO" << endl;
	avgstring_stream << "VIH VIAEHLB BAO" << endl;

	for (int iters = 0; iters < repetitions; iters++)
	{
		k = 0;
		for (int ite = StartP; ite <= endP; ite = ite + IncP)
		{
			if (expnum == 3 || expnum == 4)
			{
				States = ite;
				MDP = generate_random_MDP_normal_distributed_rewards(ite, Actions, action_prob, SS, seed, 1000, 10);
			}
			else if (expnum == 5 || expnum == 6)
				MDP = generate_random_MDP_normal_distributed_rewards(States, ite, action_prob, SS, seed, 1000, 10);
			else
				MDP = generate_random_MDP_normal_distributed_rewards(States, Actions, action_prob, ite, seed, 1000, 10);
			R_type R = get<0>(MDP);
			A_type A = get<1>(MDP);
			P_type P = get<2>(MDP);
			// Print the values of P
			// std::cout << "Doubles:" << std::endl;
			// typedef <<pair<vector<double>, vector<int> > > > P_type;
			double sum = 0;
			/* for (int i=0;i<P.size();i++) {
				 for (int j=0;j<P[i].size();j++) {
					 sum=0;
					 cout<<"i,j"<<i<<","<<j<<endl;
					 for (int k=0;k<P[i][j].second.size();k++){
						 std::cout << P[i][j].first[k] << ", "<<P[i][j].second[k]<<endl;
						 sum+=(P[i][j].first[k]);
				 }
				 if(sum!=1)
				 cout<<"monaxa etsi tha teliwsoume emis "<<sum<<endl;
				 }
				 cout <<"Break" <<endl;
			 }
			 cout<<"hi"<<endl;*/
			cout << States << endl;
			int counter = 0;
			if (BO)
			{
				A_type A1 = copy_A(A);
				auto start_VI = high_resolution_clock::now();
				V_approx_solution_tuple = value_iterationGS(States, R, A1, P, gamma, epsilon);
				auto stop_VI = high_resolution_clock::now();
				duration_VI = duration_cast<milliseconds>(stop_VI - start_VI);
				// VIU testing
				A_type A6 = copy_A(A);
				auto start_VIU = high_resolution_clock::now();
				V_approx_solution_upper_tuple = value_iteration_upperGS(States, R, A6, P, gamma, epsilon);
				auto stop_VIU = high_resolution_clock::now();
				duration_VIU = duration_cast<milliseconds>(stop_VIU - start_VIU);

				A_type A3 = copy_A(A);
				auto start_BVI = high_resolution_clock::now();
				V_bounded_approx_solution_tuple = bounded_value_iterationGS(States, R, A3, P, gamma, epsilon);
				auto stop_BVI = high_resolution_clock::now();
				duration_BVI = duration_cast<milliseconds>(stop_BVI - start_BVI);
			}
			A_type A4 = copy_A(A);
			auto start_VIAE = high_resolution_clock::now();
			V_AE_approx_solution_tuple = value_iteration_action_eliminationGS(States, R, A4, P, gamma, epsilon);
			auto stop_VIAE = high_resolution_clock::now();
			duration_VIAE = duration_cast<milliseconds>(stop_VIAE - start_VIAE);

			A_type A5 = copy_A(A);
			auto start_VIAEH = high_resolution_clock::now();
			V_AE_H_approx_solution_tuple = value_iteration_action_elimination_heapsGS(States, R, A5, P, gamma, epsilon);
			auto stop_VIAEH = high_resolution_clock::now();
			duration_VIAEH = duration_cast<milliseconds>(stop_VIAEH - start_VIAEH);
			// VIAEHL

			A_type A2 = copy_A(A);
			auto start_VIH = high_resolution_clock::now();
			V_heap_approx_tuple = value_iteration_with_heapGS(States, R, A2, P, gamma, epsilon);
			auto stop_VIH = high_resolution_clock::now();
			duration_VIH = duration_cast<milliseconds>(stop_VIH - start_VIH);

			A_type A8 = copy_A(A);
			auto start_VIAEHL = high_resolution_clock::now();
			VIAEHL_approx_solution_tuple = value_iteration_action_elimination_heaps_lower_bound_approxGS(States, R, A8, P, gamma, epsilon);
			auto stop_VIAEHL = high_resolution_clock::now();
			duration_VIAEHL = duration_cast<milliseconds>(stop_VIAEHL - start_VIAEHL);
			// BAO
			A_type A9 = copy_A(A);
			auto start_BAO = high_resolution_clock::now();
			BAO_approx_solution_tuple = value_iteration_BAOGS(States, R, A9, P, gamma, epsilon);
			auto stop_BAO = high_resolution_clock::now();
			duration_BAO = duration_cast<milliseconds>(stop_BAO - start_BAO);

			// They should in theory all be epsilon from true value, and therefore, at most be 2 * epsilon from each other
			vector<double> BAO_approx_solution = get<0>(BAO_approx_solution_tuple);
			vector<double> VIAEHL_approx_solution = get<0>(VIAEHL_approx_solution_tuple);
			vector<double> V_heap_approx = get<0>(V_heap_approx_tuple);
			if (BO)
			{
				vector<double> V_AE_H_approx_solution = get<0>(V_AE_H_approx_solution_tuple);
				vector<double> V_AE_approx_solution = get<0>(V_AE_approx_solution_tuple);
				vector<double> V_bounded_approx_solution = get<0>(V_bounded_approx_solution_tuple);
				vector<double> V_approx_solution = get<0>(V_approx_solution_tuple);
				vector<double> V_approx_solution_upper = get<0>(V_approx_solution_upper_tuple);

				if (abs_max_diff_vectors(V_approx_solution, BAO_approx_solution) > (2 * epsilon))
				{
					printf("DIFFERENCE1\n");
				}
				if (abs_max_diff_vectors(V_approx_solution, V_heap_approx) > (2 * epsilon))
				{
					printf("DIFFERENCE2\n");
				}

				if (abs_max_diff_vectors(V_approx_solution, V_AE_H_approx_solution) > (2 * epsilon))
				{
					printf("DIFFERENCE3\n");
				}
				if (abs_max_diff_vectors(V_approx_solution, V_approx_solution_upper) > (2 * epsilon))
				{
					printf("DIFFERENCE4\n");
				}
				if (abs_max_diff_vectors(V_approx_solution, V_bounded_approx_solution) > (2 * epsilon))
				{
					printf("DIFFERENCE5\n");
				}
				if (abs_max_diff_vectors(V_approx_solution, VIAEHL_approx_solution) > (2 * epsilon))
				{
					printf("DIFFERENCE6\n");
				}
				if (abs_max_diff_vectors(V_approx_solution, V_AE_approx_solution) > (2 * epsilon))
				{
					printf("DIFFERENCE7\n");
				}
				if (abs_max_diff_vectors(V_approx_solution, BAO_approx_solution) > (2 * epsilon))
				{
					printf("DIFFERENCE8\n");
				}
				VI[0][k] += duration_VI.count();
				VI[1][k] += duration_VIU.count();

				VI[3][k] += duration_BVI.count();
				VI[4][k] += duration_VIAE.count();
				VI[5][k] += duration_VIAEH.count();
				string_stream << duration_VI.count() << " " << duration_VIU.count() << " " << duration_BVI.count() << " " << duration_VIAE.count() << " " << duration_VIAEH.count() << " ";
			}
			VI[2][k] += duration_VIH.count();
			VI[6][k] += duration_VIAEHL.count();
			VI[7][k] += duration_BAO.count();

			string_stream << duration_VIH.count() << " " << duration_VIAEHL.count() << " " << duration_BAO.count() << endl;
			k++;
		}
	}
	for (int k = 0; k < NE; k++)
	{
		if (BO)
		{
			avgstring_stream << VI[0][k] / repetitions << " " << VI[1][k] / repetitions << " " << VI[3][k] / repetitions << " " << VI[4][k] / repetitions << " " << VI[5][k] / repetitions << " ";
		}
		avgstring_stream << VI[2][k] / repetitions << " " << VI[6][k] / repetitions << " " << VI[7][k] / repetitions << endl;
	}
	// WRITE ALL DATA TO THEIR RESPECTVIE FILES
	write_stringstream_to_file(string_stream, output_stream, file_name_VI);
	write_stringstream_to_file(avgstring_stream, avgoutput_stream, file_name_VIAVG);
}

void VMS(int NOFexp, double epsilon, double gamma)
{
	string seed1 = "X";
	string seed11[5] = {"1", "2", "3", "4", "5"};
	int Rseed = 5;
	int S_starting_value = 50;
	double action_prob = 1.0;
	S_type S = 0;
	string SE[5] = {"10S", "20S", "30S", "40S", "50S"};
	string AE[5] = {"10A", "20A", "30A", "40A", "50A"};
	MDP_type MDP;
	float VI[10][5];
	for (int i = 0; i < 10; i++)
		for (int j = 0; j < 5; j++)
			VI[i][j] = 0;
	int k;
	ostringstream string_stream;
	ostringstream avgstring_stream;
	ofstream output_stream;
	ofstream avgoutput_stream;
	string file_name_VI = "Skitsas//VMS.txt";
	string file_name_VIAVG = "Skitsas//VMSavg.txt";
	if (NOFexp != 1)
	{
		string file_name_VI = "Skitsas//VMA.txt";
		string file_name_VIAVG = "Skitsas//VMAavg.txt";
	}
	// int seed = time(0);
	auto timeS = high_resolution_clock::now();
	string_stream << "Experiment ID: " << NOFexp << endl;
	avgstring_stream << "Experiment ID" << NOFexp << endl;
	string_stream << "VI UVI VIH BVI VIAE VIAEH VIAEHLB BAO" << endl;
	avgstring_stream << "VI UVI VIH BVI VIAE VIAEH VIAEHLB BAO" << endl;
	for (int iter = 0; iter < 5; iter++)
	{
		seed1 = seed11[iter];
		k = 0;
		for (int temp = 0; temp < 5; temp = temp + 1)
		{
			// GENERATE THE MDP
			if (NOFexp == 1)
			{
				MDP = readMDPS(seed1, SE[temp]);
				S = (temp + 1) * 100;
			}
			else
			{
				MDP = readMDPS(seed1, AE[temp]);
				S = 500;
			}
			R_type R = get<0>(MDP);
			A_type A = get<1>(MDP);
			P_type P = get<2>(MDP);

			A_type A1 = copy_A(A);
			auto start_VI = high_resolution_clock::now();
			V_type V_approx_solution_tuple = value_iterationGS(S, R, A1, P, gamma, epsilon);
			auto stop_VI = high_resolution_clock::now();
			auto duration_VI = duration_cast<milliseconds>(stop_VI - start_VI);

			// VIU testing
			A_type A6 = copy_A(A);
			auto start_VIU = high_resolution_clock::now();
			V_type V_approx_solution_upper_tuple = value_iteration_upperGS(S, R, A6, P, gamma, epsilon);
			auto stop_VIU = high_resolution_clock::now();
			auto duration_VIU = duration_cast<milliseconds>(stop_VIU - start_VIU);

			// VIH testing
			A_type A2 = copy_A(A);
			auto start_VIH = high_resolution_clock::now();
			V_type V_heap_approx_tuple = value_iteration_with_heapGS(S, R, A2, P, gamma, epsilon);
			auto stop_VIH = high_resolution_clock::now();
			auto duration_VIH = duration_cast<milliseconds>(stop_VIH - start_VIH);

			A_type A3 = copy_A(A);
			auto start_BVI = high_resolution_clock::now();
			V_type V_bounded_approx_solution_tuple = bounded_value_iterationGS(S, R, A3, P, gamma, epsilon);
			auto stop_BVI = high_resolution_clock::now();
			auto duration_BVI = duration_cast<milliseconds>(stop_BVI - start_BVI);

			A_type A4 = copy_A(A);
			auto start_VIAE = high_resolution_clock::now();
			V_type V_AE_approx_solution_tuple = value_iteration_action_eliminationGS(S, R, A4, P, gamma, epsilon);
			auto stop_VIAE = high_resolution_clock::now();
			auto duration_VIAE = duration_cast<milliseconds>(stop_VIAE - start_VIAE);

			A_type A5 = copy_A(A);
			auto start_VIAEH = high_resolution_clock::now();
			V_type V_AE_H_approx_solution_tuple = value_iteration_action_elimination_heapsGS(S, R, A5, P, gamma, epsilon);
			auto stop_VIAEH = high_resolution_clock::now();
			auto duration_VIAEH = duration_cast<milliseconds>(stop_VIAEH - start_VIAEH);

			A_type A8 = copy_A(A);
			auto start_VIAEHL = high_resolution_clock::now();
			V_type VIAEHL_approx_solution_tuple = value_iteration_action_elimination_heaps_lower_bound_approxGS(S, R, A8, P, gamma, epsilon);
			auto stop_VIAEHL = high_resolution_clock::now();
			auto duration_VIAEHL = duration_cast<milliseconds>(stop_VIAEHL - start_VIAEHL);

			A_type A9 = copy_A(A);
			auto start_BAO = high_resolution_clock::now();
			V_type BAO_approx_solution_tuple = value_iteration_BAOGS(S, R, A9, P, gamma, epsilon);
			auto stop_BAO = high_resolution_clock::now();
			auto duration_BAO = duration_cast<milliseconds>(stop_BAO - start_BAO);

			// They should in theory all be epsilon from true value, and therefore, at most be 2 * epsilon from each other
			// testing
			vector<double> BAO_approx_solution = get<0>(BAO_approx_solution_tuple);
			vector<double> VIAEHL_approx_solution = get<0>(VIAEHL_approx_solution_tuple);
			vector<double> V_AE_H_approx_solution = get<0>(V_AE_H_approx_solution_tuple);
			vector<double> V_AE_approx_solution = get<0>(V_AE_approx_solution_tuple);
			vector<double> V_bounded_approx_solution = get<0>(V_bounded_approx_solution_tuple);
			vector<double> V_heap_approx = get<0>(V_heap_approx_tuple);
			vector<double> V_approx_solution = get<0>(V_approx_solution_tuple);
			vector<double> V_approx_solution_upper = get<0>(V_approx_solution_upper_tuple);

			if (abs_max_diff_vectors(V_approx_solution, BAO_approx_solution) > (2 * epsilon))
			{
				printf("DIFFERENCE1a\n");
			}
			if (abs_max_diff_vectors(V_approx_solution, VIAEHL_approx_solution) > (2 * epsilon))
			{
				printf("DIFFERENCE1b\n");
			}
			if (abs_max_diff_vectors(V_approx_solution, V_AE_H_approx_solution) > (2 * epsilon))
			{
				printf("DIFFERENCE1c\n");
			}
			if (abs_max_diff_vectors(V_approx_solution, V_AE_approx_solution) > (2 * epsilon))
			{
				printf("DIFFERENCE1d\n");
			}
			if (abs_max_diff_vectors(V_approx_solution, V_bounded_approx_solution) > (2 * epsilon))
			{
				printf("DIFFERENCE1e\n");
			}
			if (abs_max_diff_vectors(V_approx_solution, V_heap_approx) > (2 * epsilon))
			{
				printf("DIFFERENCE1f\n");
			}
			if (abs_max_diff_vectors(V_approx_solution, V_approx_solution) > (2 * epsilon))
			{
				printf("DIFFERENCE1g\n");
			}
			if (abs_max_diff_vectors(V_approx_solution, V_approx_solution_upper) > (2 * epsilon))
			{
				printf("DIFFERENCE1g\n");
			}

			VI[0][temp] += duration_VI.count();
			VI[1][temp] += duration_VIU.count();
			VI[2][temp] += duration_VIH.count();
			VI[3][temp] += duration_BVI.count();
			VI[4][temp] += duration_VIAE.count();
			VI[5][temp] += duration_VIAEH.count();
			VI[6][temp] += duration_VIAEHL.count();
			VI[7][temp] += duration_BAO.count();
			string_stream << duration_VI.count() << " " << duration_VIU.count() << " " << duration_VIH.count() << " " << duration_BVI.count() << " ";
			string_stream << duration_VIAE.count() << " " << duration_VIAEH.count() << " " << duration_VIAEHL.count() << " " << duration_BAO.count() << endl;

			// VI[8][temp]+=duration_BAOSK.count();
			// k++;
		}
	}
	for (int k = 0; k < 5; k++)
	{
		avgstring_stream << VI[0][k] / 5 << " " << VI[1][k] / 5 << " " << VI[2][k] / 5 << " " << VI[3][k] / 5 << " ";
		avgstring_stream << VI[4][k] / 5 << " " << VI[5][k] / 5 << " " << VI[6][k] / 5 << " " << VI[7][k] / 5 << endl;
	}

	// WRITE ALL DATA TO THEIR RESPECTVIE FILES
	write_stringstream_to_file(string_stream, output_stream, file_name_VI);
	write_stringstream_to_file(avgstring_stream, avgoutput_stream, file_name_VIAVG);
}

// GENERAL HELPER METHOD TO WRITE DATA TO FILE
void write_stringstream_to_file(ostringstream &string_stream, ofstream &output_stream, string file_name)
{
	output_stream.open(file_name);
	if (output_stream.is_open())
	{
		// printf("opened file: success\n");
		output_stream << string_stream.str();
		output_stream.close();
	}
	else
	{
		printf("opened file: fail\n");
	}
}

// VARYING ACTION PROBABILITY EXPERIMENT
void write_meta_data_to_dat_file_action_prob(ostringstream &string_stream, int S, int A_num, double epsilon, double gamma, double upper_reward, double non_zero_transition, double action_prob_starting_value, double action_prob_finishing_value, double action_prob_increment)
{
	time_t time_now = time(0);
	string_stream << "# META DATA" << endl;
	string_stream << "# " << endl;
	string_stream << "# "
				  << "experiment run at: " << ctime(&time_now);
	string_stream << "# " << endl;
	string_stream << "# "
				  << "gamma = " << gamma << endl;
	string_stream << "# "
				  << "epsilon = " << epsilon << endl;
	string_stream << "# "
				  << "S = " << S << endl;
	string_stream << "# "
				  << "A = " << A_num << endl;
	string_stream << "# "
				  << "non_zero_transition = " << non_zero_transition << endl;
	string_stream << "# "
				  << "upper_reward = " << upper_reward << endl;
	string_stream << "# " << endl;
	string_stream << "# VARIABLE IN THIS EXPERIMENT" << endl;
	string_stream << "# "
				  << "action_prob is varied from " << action_prob_starting_value << " to " << action_prob_finishing_value << " with " << action_prob_increment << " increment" << endl;
	string_stream << "# " << endl;
	string_stream << "# ACTUAL DATA" << endl;
	string_stream << "# action_prob | microseconds" << endl;
}

void create_data_tables_action_prob(string filename, int S, int A_num, double epsilon, double gamma, double upper_reward, double non_zero_transition)
{

	// the stringstreams to create the test for the files
	ostringstream stringstream_BVI;
	ostringstream stringstream_VI;
	ostringstream stringstream_VIU;
	ostringstream stringstream_VIH;
	ostringstream stringstream_VIAE;
	ostringstream stringstream_VIAEH;

	// the file output objects
	ofstream output_stream_BVI;
	ofstream output_stream_VI;
	ofstream output_stream_VIU;
	ofstream output_stream_VIH;
	ofstream output_stream_VIAE;
	ofstream output_stream_VIAEH;

	// set the name of the file to write to
	string file_name_BVI = "data_tables/action_prob/" + filename + "_BVI.dat";
	string file_name_VI = "data_tables/action_prob/" + filename + "_VI.dat";
	string file_name_VIU = "data_tables/action_prob/" + filename + "_VIU.dat";
	string file_name_VIH = "data_tables/action_prob/" + filename + "_VIH.dat";
	string file_name_VIAE = "data_tables/action_prob/" + filename + "_VIAE.dat";
	string file_name_VIAEH = "data_tables/action_prob/" + filename + "_VIAEH.dat";

	double action_prob_starting_value = 0.10;
	double action_prob_finishing_value = 1.0;
	double action_prob_increment = 0.05;

	// write meta data to all stringstreams as first in their respective files
	write_meta_data_to_dat_file_action_prob(stringstream_VI, S, A_num, epsilon, gamma, upper_reward, non_zero_transition, action_prob_starting_value, action_prob_finishing_value, action_prob_increment);
	write_meta_data_to_dat_file_action_prob(stringstream_VIU, S, A_num, epsilon, gamma, upper_reward, non_zero_transition, action_prob_starting_value, action_prob_finishing_value, action_prob_increment);
	write_meta_data_to_dat_file_action_prob(stringstream_VIH, S, A_num, epsilon, gamma, upper_reward, non_zero_transition, action_prob_starting_value, action_prob_finishing_value, action_prob_increment);
	write_meta_data_to_dat_file_action_prob(stringstream_BVI, S, A_num, epsilon, gamma, upper_reward, non_zero_transition, action_prob_starting_value, action_prob_finishing_value, action_prob_increment);
	write_meta_data_to_dat_file_action_prob(stringstream_VIAE, S, A_num, epsilon, gamma, upper_reward, non_zero_transition, action_prob_starting_value, action_prob_finishing_value, action_prob_increment);
	write_meta_data_to_dat_file_action_prob(stringstream_VIAEH, S, A_num, epsilon, gamma, upper_reward, non_zero_transition, action_prob_starting_value, action_prob_finishing_value, action_prob_increment);

	for (double action_prob = action_prob_starting_value; action_prob <= action_prob_finishing_value; action_prob = action_prob + action_prob_increment)
	{

		// status message of experiment
		printf("Beginning iteration action_prob = %f\n", action_prob);

		// GENERATE THE MDP FROM CURRENT TIME SEED
		int seed = time(0);
		auto MDP = generate_random_MDP_with_variable_parameters(S, A_num, action_prob, non_zero_transition, upper_reward, seed);
		R_type R = get<0>(MDP);
		A_type A = get<1>(MDP);
		P_type P = get<2>(MDP);

		// VI testing
		// TODO: make method that takes arguments and performs the writing of data, takes filename, MDP etc. as argument
		A_type A1 = copy_A(A);
		auto start_VI = high_resolution_clock::now();

		V_type V_approx_solution_tuple = value_iteration(S, R, A1, P, gamma, epsilon);
		vector<double> V_approx_solution = get<0>(V_approx_solution_tuple);

		auto stop_VI = high_resolution_clock::now();
		auto duration_VI = duration_cast<microseconds>(stop_VI - start_VI);

		stringstream_VI << to_string(action_prob) << " " << duration_VI.count() << endl;

		// VIU testing
		A_type A6 = copy_A(A);
		auto start_VIU = high_resolution_clock::now();

		V_type V_approx_solution_upper_tuple = value_iteration_upper(S, R, A6, P, gamma, epsilon);
		vector<double> V_approx_solution_upper = get<0>(V_approx_solution_upper_tuple);

		auto stop_VIU = high_resolution_clock::now();
		auto duration_VIU = duration_cast<microseconds>(stop_VIU - start_VIU);

		stringstream_VIU << to_string(action_prob) << " " << duration_VIU.count() << endl;

		// VIH testing
		A_type A2 = copy_A(A);
		auto start_VIH = high_resolution_clock::now();

		V_type V_heap_approx_tuple = value_iteration_with_heap(S, R, A2, P, gamma, epsilon);
		vector<double> V_heap_approx = get<0>(V_heap_approx_tuple);

		auto stop_VIH = high_resolution_clock::now();
		auto duration_VIH = duration_cast<microseconds>(stop_VIH - start_VIH);

		stringstream_VIH << to_string(action_prob) << " " << duration_VIH.count() << endl;

		// BVI
		A_type A3 = copy_A(A);
		auto start_BVI = high_resolution_clock::now();

		V_type V_bounded_approx_solution_tuple = bounded_value_iteration(S, R, A3, P, gamma, epsilon);
		vector<double> V_bounded_approx_solution = get<0>(V_bounded_approx_solution_tuple);

		auto stop_BVI = high_resolution_clock::now();
		auto duration_BVI = duration_cast<microseconds>(stop_BVI - start_BVI);

		stringstream_BVI << to_string(action_prob) << " " << duration_BVI.count() << endl;

		// VIAE
		A_type A4 = copy_A(A);
		auto start_VIAE = high_resolution_clock::now();

		V_type V_AE_approx_solution_tuple = value_iteration_action_elimination(S, R, A4, P, gamma, epsilon);
		vector<double> V_AE_approx_solution = get<0>(V_AE_approx_solution_tuple);

		auto stop_VIAE = high_resolution_clock::now();
		auto duration_VIAE = duration_cast<microseconds>(stop_VIAE - start_VIAE);

		stringstream_VIAE << to_string(action_prob) << " " << duration_VIAE.count() << endl;

		// VIAEH
		A_type A5 = copy_A(A);
		auto start_VIAEH = high_resolution_clock::now();

		V_type V_AE_H_approx_solution_tuple = value_iteration_action_elimination_heaps(S, R, A5, P, gamma, epsilon);
		vector<double> V_AE_H_approx_solution = get<0>(V_AE_H_approx_solution_tuple);

		auto stop_VIAEH = high_resolution_clock::now();
		auto duration_VIAEH = duration_cast<microseconds>(stop_VIAEH - start_VIAEH);

		stringstream_VIAEH << to_string(action_prob) << " " << duration_VIAEH.count() << endl;

		// They should in theory all be epsilon from true value, and therefore, at most be 2 * epsilon from each other
		if (abs_max_diff_vectors(V_approx_solution, V_heap_approx) > (2 * epsilon))
		{
			printf("DIFFERENCE\n");
		}
		if (abs_max_diff_vectors(V_AE_approx_solution, V_AE_H_approx_solution) > (2 * epsilon))
		{
			printf("DIFFERENCE\n");
		}
		if (abs_max_diff_vectors(V_bounded_approx_solution, V_approx_solution_upper) > (2 * epsilon))
		{
			printf("DIFFERENCE\n");
		}
	}

	// WRITE ALL DATA TO THEIR RESPECTVIE FILES
	write_stringstream_to_file(stringstream_VI, output_stream_VI, file_name_VI);
	write_stringstream_to_file(stringstream_VIU, output_stream_VIU, file_name_VIU);
	write_stringstream_to_file(stringstream_BVI, output_stream_BVI, file_name_BVI);
	write_stringstream_to_file(stringstream_VIH, output_stream_VIH, file_name_VIH);
	write_stringstream_to_file(stringstream_VIAE, output_stream_VIAE, file_name_VIAE);
	write_stringstream_to_file(stringstream_VIAEH, output_stream_VIAEH, file_name_VIAEH);
}

// VARYING NUMBER OF STATES EXPERIMENTS
void write_meta_data_to_dat_file_number_of_states(ostringstream &string_stream, int A_num, double epsilon, double gamma, double upper_reward, double non_zero_transition, double action_prob, int S_starting_value, int S_finishing_value, int S_increment)
{
	time_t time_now = time(0);
	string_stream << "# META DATA" << endl;
	string_stream << "# " << endl;
	string_stream << "# "
				  << "experiment run at: " << ctime(&time_now);
	string_stream << "# " << endl;
	string_stream << "# "
				  << "gamma = " << gamma << endl;
	string_stream << "# "
				  << "epsilon = " << epsilon << endl;
	string_stream << "# "
				  << "A = " << A_num << endl;
	string_stream << "# "
				  << "non_zero_transition = " << non_zero_transition << endl;
	string_stream << "# "
				  << "upper_reward = " << upper_reward << endl;
	string_stream << "# "
				  << "action_prob = " << action_prob << endl;
	string_stream << "# " << endl;
	string_stream << "# VARIABLE IN THIS EXPERIMENT" << endl;
	string_stream << "# "
				  << "Number of states, S, is varied from " << S_starting_value << " to " << S_finishing_value << " with " << S_increment << " increment" << endl;
	string_stream << "# " << endl;
	string_stream << "# ACTUAL DATA" << endl;
	string_stream << "# number of states | microseconds" << endl;
}

void create_data_tables_number_of_states(string filename, int S_max, int A_num, double epsilon, double gamma, double upper_reward, double non_zero_transition)
{

	// the stringstreams to create the test for the files
	ostringstream stringstream_BVI;
	ostringstream stringstream_VI;
	ostringstream stringstream_VIU;
	ostringstream stringstream_VIH;
	ostringstream stringstream_VIHN;
	ostringstream stringstream_VIAE;
	ostringstream stringstream_VIAEH;
	ostringstream stringstream_VIAEHL;
	ostringstream stringstream_BAO;
	ostringstream stringstream_BAON;

	// the file output objects
	ofstream output_stream_BVI;
	ofstream output_stream_VI;
	ofstream output_stream_VIU;
	ofstream output_stream_VIH;
	ofstream output_stream_VIHN;
	ofstream output_stream_VIAE;
	ofstream output_stream_VIAEH;
	ofstream output_stream_VIAEHL;
	ofstream output_stream_BAO;
	ofstream output_stream_BAON;

	// set the name of the file to write to
	string file_name_BVI = "data_tables/number_of_states/" + filename + "BVI.dat";
	string file_name_VI = "data_tables/number_of_states/" + filename + "VI.dat";
	string file_name_VIU = "data_tables/number_of_states/" + filename + "VIU.dat";
	string file_name_VIH = "data_tables/number_of_states/" + filename + "VI.Tdat";
	string file_name_VIHN = "data_tables/number_of_states/" + filename + "SaaS_VIHNM.dat";
	string file_name_VIAE = "data_tables/number_of_states/" + filename + "VIAE.dat";
	string file_name_VIAEH = "data_tables/number_of_states/" + filename + "VIAEH.dat";
	string file_name_VIAEHL = "data_tables/number_of_states/" + filename + "VIAEHL.dat";
	string file_name_BAO = "data_tables/number_of_states/" + filename + "BAO.dat";
	string file_name_BAON = "data_tables/number_of_states/" + filename + "BAON.dat";

	// The varying parameters
	int S_starting_value = 50;
	int S_finishing_value = S_max;
	int S_increment = 50;
	S_finishing_value = 250;
	// hardcoded parameter
	double action_prob = 1.0;

	// write meta data to all stringstreams as first in their respective files

	write_meta_data_to_dat_file_number_of_states(stringstream_VI, A_num, epsilon, gamma, upper_reward, non_zero_transition, action_prob, S_starting_value, S_finishing_value, S_increment);
	write_meta_data_to_dat_file_number_of_states(stringstream_VIU, A_num, epsilon, gamma, upper_reward, non_zero_transition, action_prob, S_starting_value, S_finishing_value, S_increment);
	write_meta_data_to_dat_file_number_of_states(stringstream_BVI, A_num, epsilon, gamma, upper_reward, non_zero_transition, action_prob, S_starting_value, S_finishing_value, S_increment);
	write_meta_data_to_dat_file_number_of_states(stringstream_VIH, A_num, epsilon, gamma, upper_reward, non_zero_transition, action_prob, S_starting_value, S_finishing_value, S_increment);
	write_meta_data_to_dat_file_number_of_states(stringstream_VIAE, A_num, epsilon, gamma, upper_reward, non_zero_transition, action_prob, S_starting_value, S_finishing_value, S_increment);
	write_meta_data_to_dat_file_number_of_states(stringstream_VIAEH, A_num, epsilon, gamma, upper_reward, non_zero_transition, action_prob, S_starting_value, S_finishing_value, S_increment);
	float VI[10][10];
	int S2 = 0;
	int k = 0;
	int S = 1000;

	for (int iters = 0; iters < 5; iters++)
	{
		k = 0;
		for (int xs = S_starting_value; xs <= S_finishing_value; xs = xs + S_increment)
		{
			printf("Beginning iteration %d xs = %d\n", iters, xs);
			// int S2=S/20;
			// GENERATE THE MDP
			int seed = time(0);
			// auto MDP = generate_random_MDP_exponential_distributed_rewards(S, A_num, 1.0, S/2, 0.02, seed);
			// auto MDP = generate_random_MDP_normal_distributed_rewards(S, A_num, action_prob, S2, seed, 1000, 10);
			// auto MDP = generate_random_MDP_with_variable_parameters(S, A_num, action_prob, non_zero_transition, upper_reward, seed);
			// auto MDP = RiverSwim(S);
			// int xs=10;
			// xs=100;
			auto MDP = Maze(xs, xs, seed);
			S = xs * xs + 1;

			R_type R = get<0>(MDP);
			A_type A = get<1>(MDP);
			P_type P = get<2>(MDP);
			int counter = 0;
			for (int t = 0; t < S; t++)
			{
				for (auto a : A[t])
				{
					auto &[P_s_a, P_s_a_nonzero] = P[t][a];
					for (int k : P_s_a_nonzero)
					{
						// cout<<"State"<< t<<" probability "<<P_s_a[counter]<<" to "<<P_s_a_nonzero[counter]<<" state "<< R[t][a]  <<" Reward"<<a<<" Action num"<<endl;
						counter++;
					}
					counter = 0;
				}
			}
			// cout<<"MDP"<<endl;
			// gamma=1;
			// VI testing
			// TODO: make method that takes arguments and performs the writing of data, takes filename, MDP etc. as argument
			A_type A1 = copy_A(A);
			auto start_VI = high_resolution_clock::now();
			// cout<<getValue()<<endl;
			V_type V_approx_solution_tuple = value_iteration(S, R, A1, P, gamma, epsilon);
			vector<double> V_approx_solution = get<0>(V_approx_solution_tuple);

			auto stop_VI = high_resolution_clock::now();
			auto duration_VI = duration_cast<milliseconds>(stop_VI - start_VI);
			cout << "VI, " << duration_VI.count() << endl;
			stringstream_VI << to_string(S) << " " << duration_VI.count() << endl;
			VI[0][k] += duration_VI.count();
			// VIU testing
			A_type A6 = copy_A(A);
			auto start_VIU = high_resolution_clock::now();

			// for (int k;k<V_approx_solution.size();k++)
			// cout<<"V[k]= "<<k/4<<" , "<<k%4<<" , "<< k<<" , "<<V_approx_solution[k]<<endl;
			V_type V_approx_solution_upper_tuple = value_iteration_upper(S, R, A6, P, gamma, epsilon);
			vector<double> V_approx_solution_upper = get<0>(V_approx_solution_upper_tuple);
			// vector<double> V_heap_approx = get<0>(V_approx_solution_upper_tuple);
			auto stop_VIU = high_resolution_clock::now();
			auto duration_VIU = duration_cast<milliseconds>(stop_VIU - start_VIU);
			VI[1][k] += duration_VIU.count();
			stringstream_VIU << to_string(S) << " " << duration_VIU.count() << endl;
			cout << "VIU" << duration_VIU.count() << endl;
			// VIH testing
			A_type A2 = copy_A(A);
			auto start_VIH = high_resolution_clock::now();

			V_type V_heap_approx_tuple = value_iteration_with_heap(S, R, A2, P, gamma, epsilon);
			vector<double> V_heap_approx = get<0>(V_heap_approx_tuple);
			auto stop_VIH = high_resolution_clock::now();
			auto duration_VIH = duration_cast<milliseconds>(stop_VIH - start_VIH);
			/*
				int siz=xs-2;
				bool hi=false;

			for (int i=0;i<V_heap_approx.size()-1;i++){
				int x_curr=i%xs;
				int y_curr=i/xs;
				int xa1= abs(x_curr-siz);
				int ya1= abs(y_curr-siz);
				double x2=0;
			if (xa1>ya1)
				x2=xa1;
			else
				x2=ya1;
				double x1= sqrt( pow( abs(x_curr-siz),2)+pow(abs(y_curr-siz),2));
				if((-xs*5)-10>V_approx_solution[i])
					cout<<"oupsL "<<V_approx_solution[i]<<" "<< -x1*5+10<<"S"<<i<< y_curr<<x_curr<<endl;
				if(-x2<V_heap_approx[i]){
					cout<<"oupsU "<<V_approx_solution[i]<<" "<< -x2<<"S"<<i<<" " <<y_curr<<" "<<x_curr<<endl;
					hi=true;}
					}
					if (hi){
						for (int t = 0; t < S; t++) {
					for (auto a : A[t]) {
							auto& [P_s_a, P_s_a_nonzero] = P[t][a];
								for (int k : P_s_a_nonzero) {
									cout<<"State"<< t<<" probability "<<P_s_a[counter]<<" to "<<P_s_a_nonzero[counter]<<" state "<< R[t][a]  <<" Reward"<<a<<" Action num"<<endl;
									counter++;
									}
									counter=0;

							}
					}
					}*/

			VI[2][k] += duration_VIH.count();
			cout << "VIH, " << duration_VIH.count() << endl;
			stringstream_VIH << to_string(S) << " " << duration_VIH.count() << endl;
			// A_type A12 = copy_A(A);
			auto start_VIHN = high_resolution_clock::now();

			// V_type V_heap_approx_tupleN = value_iteration_VIH_custom(S, R, A12, P, gamma, epsilon);
			// vector<double> V_heap_approxN = get<0>(V_heap_approx_tuple);

			auto stop_VIHN = high_resolution_clock::now();
			auto duration_VIHN = duration_cast<milliseconds>(stop_VIHN - start_VIHN);

			// stringstream_VIHN << to_string(A_num) << " " << duration_VIHN.count() << endl;
			VI[9][k] += duration_VIHN.count();

			// BVI
			A_type A3 = copy_A(A);
			auto start_BVI = high_resolution_clock::now();

			V_type V_bounded_approx_solution_tuple = bounded_value_iteration(S, R, A3, P, gamma, epsilon);
			vector<double> V_bounded_approx_solution = get<0>(V_bounded_approx_solution_tuple);

			auto stop_BVI = high_resolution_clock::now();
			auto duration_BVI = duration_cast<milliseconds>(stop_BVI - start_BVI);

			stringstream_BVI << to_string(S) << " " << duration_BVI.count() << endl;
			VI[3][k] += duration_BVI.count();
			cout << "BVI, " << duration_BVI.count() << endl;
			// VIAE
			A_type A4 = copy_A(A);
			auto start_VIAE = high_resolution_clock::now();

			V_type V_AE_approx_solution_tuple = value_iteration_action_elimination(S, R, A4, P, gamma, epsilon);
			vector<double> V_AE_approx_solution = get<0>(V_AE_approx_solution_tuple);

			auto stop_VIAE = high_resolution_clock::now();
			auto duration_VIAE = duration_cast<milliseconds>(stop_VIAE - start_VIAE);

			stringstream_VIAE << to_string(S) << " " << duration_VIAE.count() << endl;
			VI[4][k] += duration_VIAE.count();
			cout << "VIAE, " << duration_VIAE.count() << endl;
			// VIAEH
			A_type A5 = copy_A(A);
			auto start_VIAEH = high_resolution_clock::now();

			V_type V_AE_H_approx_solution_tuple = value_iteration_action_elimination_heaps(S, R, A5, P, gamma, epsilon);
			vector<double> V_AE_H_approx_solution = get<0>(V_AE_H_approx_solution_tuple);

			auto stop_VIAEH = high_resolution_clock::now();
			auto duration_VIAEH = duration_cast<milliseconds>(stop_VIAEH - start_VIAEH);
			cout << "VIAEH, " << duration_VIAEH.count() << endl;
			stringstream_VIAEH << to_string(S) << " " << duration_VIAEH.count() << endl;
			VI[5][k] += duration_VIAEH.count();
			// VIAEHL
			A_type A8 = copy_A(A);
			auto start_VIAEHL = high_resolution_clock::now();
			V_type VIAEHL_approx_solution_tuple = value_iteration_action_elimination_heaps_lower_bound_approxA(S, R, A8, P, gamma, epsilon);
			vector<double> VIAEHL_approx_solution = get<0>(VIAEHL_approx_solution_tuple);

			auto stop_VIAEHL = high_resolution_clock::now();

			auto duration_VIAEHL = duration_cast<milliseconds>(stop_VIAEHL - start_VIAEHL);
			// cout<<"VIaeHl"<<endl;
			stringstream_VIAEHL << to_string(S) << " " << duration_VIAEHL.count() << endl;
			VI[6][k] += duration_VIAEHL.count();
			cout << "VIEHL, " << duration_VIAEHL.count() << endl;
			// BAO
			A_type A9 = copy_A(A);
			auto start_BAO = high_resolution_clock::now();

			V_type BAO_approx_solution_tuple = value_iteration_BAO(S, R, A9, P, gamma, epsilon);

			vector<double> BAO_approx_solution = get<0>(BAO_approx_solution_tuple);

			auto stop_BAO = high_resolution_clock::now();
			auto duration_BAO = duration_cast<milliseconds>(stop_BAO - start_BAO);

			stringstream_BAO << to_string(S) << " " << duration_BAO.count() << endl;
			VI[7][k] += duration_BAO.count();
			cout << "Bao, " << duration_BAO.count() << endl;
			A_type A10 = copy_A(A);
			auto start_BAOSK = high_resolution_clock::now();

			V_type BAO_approx_solution_tuple1 = value_iteration_BAOSK(S, R, A10, P, gamma, epsilon);
			vector<double> BAO_approx_solution1 = get<0>(BAO_approx_solution_tuple1);
			auto stop_BAOSK = high_resolution_clock::now();
			// cout<<BAO_approx_solution[5500]<<"Final Bao 0"<<endl;
			// cout<<VIAEHL_approx_solution[5500]<<"Final VIAEHL 0"<<endl;
			// cout<<BAO_approx_solution[9696]<<"Final Bao 0"<<endl;
			// cout<<VIAEHL_approx_solution[9696]<<"Final VIAEHL 0"<<endl;
			auto duration_BAOSK = duration_cast<milliseconds>(stop_BAOSK - start_BAOSK);
			stringstream_BAON << to_string(S) << " " << duration_BAOSK.count() << endl;
			VI[8][k] += duration_BAOSK.count();
			cout << "BAOSK," << duration_BAOSK.count() << endl;
			// They should in theory all be epsilon from true value, and therefore, at most be 2 * epsilon from each other

			if (abs_max_diff_vectors(V_approx_solution, V_heap_approx) > (2 * epsilon))
			{
				stringstream_VIHN << "DIFFERENCE1" << endl;
				printf("DIFFERENCE1\n");
			}
			if (abs_max_diff_vectors(V_AE_approx_solution, V_AE_H_approx_solution) > (2 * epsilon))
			{
				printf("DIFFERENCE2\n");
				stringstream_VIHN << "DIFFERENCE1" << endl;
			}
			if (abs_max_diff_vectors(V_bounded_approx_solution, V_approx_solution_upper) > (2 * epsilon))
			{
				printf("DIFFERENCE3\n");
				stringstream_VIHN << "DIFFERENCE1" << endl;
			}
			if (abs_max_diff_vectors(VIAEHL_approx_solution, BAO_approx_solution1) > (2 * epsilon))
			{
				printf("DIFFERENCE4a\n");
				stringstream_VIHN << "DIFFERENCE1" << endl;
			}
			if (abs_max_diff_vectors(V_approx_solution, BAO_approx_solution) > (2 * epsilon))
			{
				printf("DIFFERENCE5a\n");
				stringstream_VIHN << "DIFFERENCE1" << endl;
			}
			if (abs_max_diff_vectors(V_approx_solution, BAO_approx_solution1) > (2 * epsilon))
			{
				printf("DIFFERENCE6\n");
				stringstream_VIHN << "DIFFERENCE1" << endl;
			}
			if (abs_max_diff_vectors(V_approx_solution, V_AE_approx_solution) > (2 * epsilon))
			{
				printf("DIFFERENCE1VI\n");
				stringstream_VIHN << "DIFFERENCE1" << endl;
			}
			if (abs_max_diff_vectors(V_approx_solution, V_AE_H_approx_solution) > (2 * epsilon))
			{
				printf("DIFFERENCE2VI\n");
				stringstream_VIHN << "DIFFERENCE1" << endl;
			}
			if (abs_max_diff_vectors(V_approx_solution, V_approx_solution_upper) > (2 * epsilon))
			{
				printf("DIFFERENCE3\n");
				stringstream_VIHN << "DIFFERENCE1" << endl;
			}
			if (abs_max_diff_vectors(VIAEHL_approx_solution, V_approx_solution) > (2 * epsilon))
			{
				printf("DIFFERENCE4\n");
				stringstream_VIHN << "DIFFERENCE1" << endl;
			}
			if (abs_max_diff_vectors(V_approx_solution, V_bounded_approx_solution) > (2 * epsilon))
			{
				printf("DIFFERENCE5\n");
				stringstream_VIHN << "DIFFERENCE1" << endl;
			}
			if (abs_max_diff_vectors(V_approx_solution, V_heap_approx) > (2 * epsilon))
			{
				printf("DIFFERENCE6\n");
				stringstream_VIHN << "DIFFERENCE1" << endl;
			}
			k++;
		}
	}
	for (int k = 0; k < 10; k++)
	{
		stringstream_VI << VI[0][k] / 5 << endl;
		stringstream_VIU << VI[1][k] / 5 << endl;
		stringstream_VIH << VI[2][k] / 5 << endl;
		// stringstream_VIHN <<  VI[9][k]/5 << endl;
		stringstream_BVI << VI[3][k] / 5 << endl;
		stringstream_VIAE << VI[4][k] / 5 << endl;
		stringstream_VIAEH << VI[5][k] / 5 << endl;
		stringstream_VIAEHL << VI[6][k] / 5 << endl;
		stringstream_BAO << VI[7][k] / 5 << endl;
		stringstream_BAON << VI[8][k] / 5 << endl;
		// cout<<"writeA"<<endl;
	}
	// WRITE ALL DATA TO THEIR RESPECTVIE FILES
	write_stringstream_to_file(stringstream_VI, output_stream_VI, file_name_VI);
	write_stringstream_to_file(stringstream_VIU, output_stream_VIU, file_name_VIU);
	write_stringstream_to_file(stringstream_BVI, output_stream_BVI, file_name_BVI);
	write_stringstream_to_file(stringstream_VIH, output_stream_VIH, file_name_VIH);
	write_stringstream_to_file(stringstream_VIHN, output_stream_VIHN, file_name_VIHN);
	write_stringstream_to_file(stringstream_VIAE, output_stream_VIAE, file_name_VIAE);
	write_stringstream_to_file(stringstream_VIAEH, output_stream_VIAEH, file_name_VIAEH);
	write_stringstream_to_file(stringstream_VIAEHL, output_stream_VIAEHL, file_name_VIAEHL);
	write_stringstream_to_file(stringstream_BAO, output_stream_BAO, file_name_BAO);
	write_stringstream_to_file(stringstream_BAON, output_stream_BAON, file_name_BAON);

}

// VARYING NUMBER OF BOTH STATES AND ACTIONS
void write_meta_data_to_dat_file_number_of_states_and_actions(ostringstream &string_stream, double epsilon, double gamma, double upper_reward, double non_zero_transition, double action_prob, int A_S_starting_value, int A_S_finishing_value, int A_S_increment)
{
	time_t time_now = time(0);
	string_stream << "# META DATA" << endl;
	string_stream << "# " << endl;
	string_stream << "# "
				  << "experiment run at: " << ctime(&time_now);
	string_stream << "# " << endl;
	string_stream << "# "
				  << "gamma = " << gamma << endl;
	string_stream << "# "
				  << "epsilon = " << epsilon << endl;
	string_stream << "# "
				  << "non_zero_transition = " << non_zero_transition << endl;
	string_stream << "# "
				  << "upper_reward = " << upper_reward << endl;
	string_stream << "# "
				  << "action_prob = " << action_prob << endl;
	string_stream << "# " << endl;
	string_stream << "# VARIABLE IN THIS EXPERIMENT" << endl;
	string_stream << "# "
				  << "Number of actions and states, A_S, is varied from " << A_S_starting_value << " to " << A_S_finishing_value << " with " << A_S_increment << " increment" << endl;
	string_stream << "# " << endl;
	string_stream << "# ACTUAL DATA" << endl;
	string_stream << "# number of actions and states | microseconds" << endl;
}

void create_data_tables_number_of_states_and_actions(string filename, int A_S_max, double epsilon, double gamma, double upper_reward, double non_zero_transition)
{

	// the stringstreams to create the test for the files
	ostringstream stringstream_BVI;
	ostringstream stringstream_VI;
	ostringstream stringstream_VIU;
	ostringstream stringstream_VIH;
	ostringstream stringstream_VIAE;
	ostringstream stringstream_VIAEH;

	// the file output objects
	ofstream output_stream_BVI;
	ofstream output_stream_VI;
	ofstream output_stream_VIU;
	ofstream output_stream_VIH;
	ofstream output_stream_VIAE;
	ofstream output_stream_VIAEH;

	// set the name of the file to write to
	string file_name_BVI = "data_tables/number_of_states_and_actions/" + filename + "_BVI.dat";
	string file_name_VI = "data_tables/number_of_states_and_actions/" + filename + "_VI.dat";
	string file_name_VIU = "data_tables/number_of_states_and_actions/" + filename + "_VIU.dat";
	string file_name_VIH = "data_tables/number_of_states_and_actions/" + filename + "_VIH.dat";
	string file_name_VIAE = "data_tables/number_of_states_and_actions/" + filename + "_VIAE.dat";
	string file_name_VIAEH = "data_tables/number_of_states_and_actions/" + filename + "_VIAEH.dat";

	// The varying parameters
	int A_S_starting_value = 50;
	int A_S_finishing_value = A_S_max;
	int A_S_increment = 50;

	// hardcoded parameter
	double action_prob = 1.0;

	write_meta_data_to_dat_file_number_of_states_and_actions(stringstream_VI, epsilon, gamma, upper_reward, non_zero_transition, action_prob, A_S_starting_value, A_S_finishing_value, A_S_increment);
	write_meta_data_to_dat_file_number_of_states_and_actions(stringstream_VIU, epsilon, gamma, upper_reward, non_zero_transition, action_prob, A_S_starting_value, A_S_finishing_value, A_S_increment);
	write_meta_data_to_dat_file_number_of_states_and_actions(stringstream_VIH, epsilon, gamma, upper_reward, non_zero_transition, action_prob, A_S_starting_value, A_S_finishing_value, A_S_increment);
	write_meta_data_to_dat_file_number_of_states_and_actions(stringstream_BVI, epsilon, gamma, upper_reward, non_zero_transition, action_prob, A_S_starting_value, A_S_finishing_value, A_S_increment);
	write_meta_data_to_dat_file_number_of_states_and_actions(stringstream_VIAE, epsilon, gamma, upper_reward, non_zero_transition, action_prob, A_S_starting_value, A_S_finishing_value, A_S_increment);
	write_meta_data_to_dat_file_number_of_states_and_actions(stringstream_VIAEH, epsilon, gamma, upper_reward, non_zero_transition, action_prob, A_S_starting_value, A_S_finishing_value, A_S_increment);

	for (int A_S = A_S_starting_value; A_S <= A_S_finishing_value; A_S = A_S + A_S_increment)
	{

		printf("Beginning iteration A_S = %d\n", A_S);

		// GENERATE THE MDP
		int seed = time(0);
		auto MDP = generate_random_MDP_with_variable_parameters(A_S, A_S, action_prob, non_zero_transition, upper_reward, seed);
		R_type R = get<0>(MDP);
		A_type A = get<1>(MDP);
		P_type P = get<2>(MDP);

		// VI testing
		// TODO: make method that takes arguments and performs the writing of data, takes filename, MDP etc. as argument
		A_type A1 = copy_A(A);
		auto start_VI = high_resolution_clock::now();

		V_type V_approx_solution_tuple = value_iteration(A_S, R, A1, P, gamma, epsilon);
		vector<double> V_approx_solution = get<0>(V_approx_solution_tuple);

		auto stop_VI = high_resolution_clock::now();
		auto duration_VI = duration_cast<microseconds>(stop_VI - start_VI);

		stringstream_VI << to_string(A_S) << " " << duration_VI.count() << endl;

		// VIU testing
		A_type A6 = copy_A(A);
		auto start_VIU = high_resolution_clock::now();

		V_type V_approx_solution_upper_tuple = value_iteration_upper(A_S, R, A6, P, gamma, epsilon);
		vector<double> V_approx_solution_upper = get<0>(V_approx_solution_upper_tuple);

		auto stop_VIU = high_resolution_clock::now();
		auto duration_VIU = duration_cast<microseconds>(stop_VIU - start_VIU);

		stringstream_VIU << to_string(A_S) << " " << duration_VIU.count() << endl;

		// VIH testing
		A_type A2 = copy_A(A);
		auto start_VIH = high_resolution_clock::now();

		V_type V_heap_approx_tuple = value_iteration_with_heap(A_S, R, A2, P, gamma, epsilon);
		vector<double> V_heap_approx = get<0>(V_heap_approx_tuple);

		auto stop_VIH = high_resolution_clock::now();
		auto duration_VIH = duration_cast<microseconds>(stop_VIH - start_VIH);

		stringstream_VIH << to_string(A_S) << " " << duration_VIH.count() << endl;

		// BVI
		A_type A3 = copy_A(A);
		auto start_BVI = high_resolution_clock::now();

		V_type V_bounded_approx_solution_tuple = bounded_value_iteration(A_S, R, A3, P, gamma, epsilon);
		vector<double> V_bounded_approx_solution = get<0>(V_bounded_approx_solution_tuple);

		auto stop_BVI = high_resolution_clock::now();
		auto duration_BVI = duration_cast<microseconds>(stop_BVI - start_BVI);

		stringstream_BVI << to_string(A_S) << " " << duration_BVI.count() << endl;

		// VIAE
		A_type A4 = copy_A(A);
		auto start_VIAE = high_resolution_clock::now();

		V_type V_AE_approx_solution_tuple = value_iteration_action_elimination(A_S, R, A4, P, gamma, epsilon);
		vector<double> V_AE_approx_solution = get<0>(V_AE_approx_solution_tuple);

		auto stop_VIAE = high_resolution_clock::now();
		auto duration_VIAE = duration_cast<microseconds>(stop_VIAE - start_VIAE);

		stringstream_VIAE << to_string(A_S) << " " << duration_VIAE.count() << endl;

		// VIAEH
		A_type A5 = copy_A(A);
		auto start_VIAEH = high_resolution_clock::now();

		V_type V_AE_H_approx_solution_tuple = value_iteration_action_elimination_heaps(A_S, R, A5, P, gamma, epsilon);
		vector<double> V_AE_H_approx_solution = get<0>(V_AE_H_approx_solution_tuple);

		auto stop_VIAEH = high_resolution_clock::now();
		auto duration_VIAEH = duration_cast<microseconds>(stop_VIAEH - start_VIAEH);

		stringstream_VIAEH << to_string(A_S) << " " << duration_VIAEH.count() << endl;

		// They should in theory all be epsilon from true value, and therefore, at most be 2 * epsilon from each other
		if (abs_max_diff_vectors(V_approx_solution, V_heap_approx) > (2 * epsilon))
		{
			printf("DIFFERENCE\n");
		}
		if (abs_max_diff_vectors(V_AE_approx_solution, V_AE_H_approx_solution) > (2 * epsilon))
		{
			printf("DIFFERENCE\n");
		}
		if (abs_max_diff_vectors(V_bounded_approx_solution, V_approx_solution_upper) > (2 * epsilon))
		{
			printf("DIFFERENCE\n");
		}
	}

	// WRITE ALL DATA TO THEIR RESPECTVIE FILES
	write_stringstream_to_file(stringstream_VI, output_stream_VI, file_name_VI);
	write_stringstream_to_file(stringstream_VIU, output_stream_VIU, file_name_VIU);
	write_stringstream_to_file(stringstream_BVI, output_stream_BVI, file_name_BVI);
	write_stringstream_to_file(stringstream_VIH, output_stream_VIH, file_name_VIH);
	write_stringstream_to_file(stringstream_VIAE, output_stream_VIAE, file_name_VIAE);
	write_stringstream_to_file(stringstream_VIAEH, output_stream_VIAEH, file_name_VIAEH);
}

// VARYING NUMBER OF ACTIONS
void write_meta_data_to_dat_file_number_of_actions(ostringstream &string_stream, int S, double epsilon, double gamma, double upper_reward, double non_zero_transition, double action_prob, int A_starting_value, int A_finishing_value, int A_increment)
{
	time_t time_now = time(0);
	string_stream << "# META DATA" << endl;
	string_stream << "# " << endl;
	string_stream << "# "
				  << "experiment run at: " << ctime(&time_now);
	string_stream << "# " << endl;
	string_stream << "# "
				  << "gamma = " << gamma << endl;
	string_stream << "# "
				  << "epsilon = " << epsilon << endl;
	string_stream << "# "
				  << "S = " << S << endl;
	string_stream << "# "
				  << "non_zero_transition = " << non_zero_transition << endl;
	string_stream << "# "
				  << "upper_reward = " << upper_reward << endl;
	string_stream << "# "
				  << "action_prob = " << action_prob << endl;
	string_stream << "# " << endl;
	string_stream << "# VARIABLE IN THIS EXPERIMENT" << endl;
	string_stream << "# "
				  << "Number of actions, A, is varied from " << A_starting_value << " to " << A_finishing_value << " with " << A_increment << " increment" << endl;
	string_stream << "# " << endl;
	string_stream << "# ACTUAL DATA" << endl;
	string_stream << "# number of actions | microseconds" << endl;
}

void create_data_tables_number_of_actions(string filename, int S, int A_max, double epsilon, double gamma, double upper_reward, double non_zero_transition)
{

	// the stringstreams to create the test for the files
	ostringstream stringstream_BVI;
	ostringstream stringstream_VI;
	ostringstream stringstream_VIU;
	ostringstream stringstream_VIH;
	ostringstream stringstream_VIHN;
	ostringstream stringstream_VIAE;
	ostringstream stringstream_VIAEH;
	ostringstream stringstream_VIAEHL;
	ostringstream stringstream_BAO;
	ostringstream stringstream_BAON;

	// the file output objects
	ofstream output_stream_BVI;
	ofstream output_stream_VI;
	ofstream output_stream_VIU;
	ofstream output_stream_VIH;
	ofstream output_stream_VIHN;
	ofstream output_stream_VIAE;
	ofstream output_stream_VIAEH;
	ofstream output_stream_VIAEHL;
	ofstream output_stream_BAO;
	ofstream output_stream_BAON;

	// set the name of the file to write to
	string file_name_BVI = "data_tables/number_of_actions/" + filename + "_BVI.dat";
	string file_name_VI = "data_tables/number_of_actions/" + filename + "_VI.dat";
	string file_name_VIU = "data_tables/number_of_actions/" + filename + "_VIU.dat";
	string file_name_VIH = "data_tables/number_of_actions/" + filename + "S_10_VIH.dat";
	string file_name_VIHN = "data_tables/number_of_actions/" + filename + "_VIHN.dat";
	string file_name_VIAE = "data_tables/number_of_actions/" + filename + "_VIAE.dat";
	string file_name_VIAEH = "data_tables/number_of_actions/" + filename + "_VIAEH.dat";
	string file_name_VIAEHL = "data_tables/number_of_actions/" + filename + "S_10_VIAEHL.dat";
	string file_name_BAO = "data_tables/number_of_actions/" + filename + "_BAO.dat";
	string file_name_BAON = "data_tables/number_of_actions/" + filename + "S_10_BAON.dat";

	// The varying parameters
	int A_starting_value = 50;
	int A_finishing_value = A_max;
	int A_increment = 50;
	int S2 = S / 10;
	// hardcoded parameter
	double action_prob = 1.0;

	// write meta data to all stringstreams as first in their respective files
	write_meta_data_to_dat_file_number_of_actions(stringstream_VI, S, epsilon, gamma, upper_reward, non_zero_transition, action_prob, A_starting_value, A_finishing_value, A_increment);
	write_meta_data_to_dat_file_number_of_actions(stringstream_VIU, S, epsilon, gamma, upper_reward, non_zero_transition, action_prob, A_starting_value, A_finishing_value, A_increment);
	write_meta_data_to_dat_file_number_of_actions(stringstream_VIH, S, epsilon, gamma, upper_reward, non_zero_transition, action_prob, A_starting_value, A_finishing_value, A_increment);
	write_meta_data_to_dat_file_number_of_actions(stringstream_BVI, S, epsilon, gamma, upper_reward, non_zero_transition, action_prob, A_starting_value, A_finishing_value, A_increment);
	write_meta_data_to_dat_file_number_of_actions(stringstream_VIAE, S, epsilon, gamma, upper_reward, non_zero_transition, action_prob, A_starting_value, A_finishing_value, A_increment);
	write_meta_data_to_dat_file_number_of_actions(stringstream_VIAEH, S, epsilon, gamma, upper_reward, non_zero_transition, action_prob, A_starting_value, A_finishing_value, A_increment);
	float VI[10][50];
	int k = 0;
	for (int iters = 0; iters < 5; iters++)
	{
		k = 0;
		for (int A_num = A_starting_value; A_num <= A_finishing_value; A_num = A_num + A_increment)
		{

			printf("Beginning iteration A_num = %d\n", A_num);

			// auto MDP = generate_random_MDP_exponential_distributed_rewards(S, A_num, action_prob, non_zero_transition, 0.02, seed);
			// auto MDP = generate_random_MDP_with_variable_parameters(S, A_num, action_prob, non_zero_transition, 1000, seed);
			// GENERATE THE MDP
			int seed = time(0);
			auto MDP = generate_random_MDP_normal_distributed_rewards(S, A_num, 1.0, S2, seed, 1000, 10);
			// auto MDP = generate_random_MDP_exponential_distributed_rewards(S, A_num, 1.0, S, 0.02, seed);
			// auto MDP = generate_random_MDP_with_variable_parameters(S, A_num, 1.0, non_zero_transition, upper_reward, seed);
			R_type R = get<0>(MDP);
			A_type A = get<1>(MDP);
			P_type P = get<2>(MDP);

			// VI testing
			// TODO: make method that takes arguments and performs the writing of data, takes filename, MDP etc. as argument
			A_type A1 = copy_A(A);
			auto start_VI = high_resolution_clock::now();

			// V_type V_approx_solution_tuple = value_iteration(S, R, A1, P, gamma, epsilon);
			// vector<double> V_approx_solution = get<0>(V_approx_solution_tuple);

			auto stop_VI = high_resolution_clock::now();
			auto duration_VI = duration_cast<milliseconds>(stop_VI - start_VI);

			stringstream_VI << to_string(A_num) << " " << duration_VI.count() << endl;

			// VIU testing
			A_type A6 = copy_A(A);
			auto start_VIU = high_resolution_clock::now();

			////V_type V_approx_solution_upper_tuple = value_iteration_upper(S, R, A6, P, gamma, epsilon);
			// vector<double> V_approx_solution_upper = get<0>(V_approx_solution_upper_tuple);

			auto stop_VIU = high_resolution_clock::now();
			auto duration_VIU = duration_cast<milliseconds>(stop_VIU - start_VIU);

			stringstream_VIU << to_string(A_num) << " " << duration_VIU.count() << endl;

			// VIH testing
			A_type A2 = copy_A(A);
			auto start_VIH = high_resolution_clock::now();

			V_type V_heap_approx_tuple = value_iteration_with_heap(S, R, A2, P, gamma, epsilon);
			vector<double> V_heap_approx = get<0>(V_heap_approx_tuple);

			auto stop_VIH = high_resolution_clock::now();
			auto duration_VIH = duration_cast<milliseconds>(stop_VIH - start_VIH);

			stringstream_VIH << to_string(A_num) << " " << duration_VIH.count() << endl;
			A_type A12 = copy_A(A);
			auto start_VIHN = high_resolution_clock::now();

			// V_type V_heap_approx_tupleN = value_iteration_VIH_custom(S, R, A12, P, gamma, epsilon);
			// vector<double> V_heap_approxN = get<0>(V_heap_approx_tuple);

			auto stop_VIHN = high_resolution_clock::now();
			auto duration_VIHN = duration_cast<milliseconds>(stop_VIHN - start_VIHN);

			stringstream_VIHN << to_string(A_num) << " " << duration_VIHN.count() << endl;

			// BVI
			A_type A3 = copy_A(A);
			auto start_BVI = high_resolution_clock::now();

			// V_type V_bounded_approx_solution_tuple = bounded_value_iteration(S, R, A3, P, gamma, epsilon);
			// vector<double> V_bounded_approx_solution = get<0>(V_bounded_approx_solution_tuple);

			auto stop_BVI = high_resolution_clock::now();
			auto duration_BVI = duration_cast<milliseconds>(stop_BVI - start_BVI);

			stringstream_BVI << to_string(A_num) << " " << duration_BVI.count() << endl;

			// VIAE
			A_type A4 = copy_A(A);
			auto start_VIAE = high_resolution_clock::now();

			// V_type V_AE_approx_solution_tuple = value_iteration_action_elimination(S, R, A4, P, gamma, epsilon);
			// vector<double> V_AE_approx_solution = get<0>(V_AE_approx_solution_tuple);

			auto stop_VIAE = high_resolution_clock::now();
			auto duration_VIAE = duration_cast<milliseconds>(stop_VIAE - start_VIAE);

			stringstream_VIAE << to_string(A_num) << " " << duration_VIAE.count() << endl;

			// VIAEH
			A_type A5 = copy_A(A);
			auto start_VIAEH = high_resolution_clock::now();

			// V_type V_AE_H_approx_solution_tuple = value_iteration_action_elimination_heaps(S, R, A5, P, gamma, epsilon);
			// vector<double> V_AE_H_approx_solution = get<0>(V_AE_H_approx_solution_tuple);

			auto stop_VIAEH = high_resolution_clock::now();
			auto duration_VIAEH = duration_cast<milliseconds>(stop_VIAEH - start_VIAEH);

			stringstream_VIAEH << to_string(A_num) << " " << duration_VIAEH.count() << endl;

			// VIAEHL
			A_type A9 = copy_A(A);
			auto start_VIAEHL = high_resolution_clock::now();

			V_type VIAEHL_approx_solution_tuple = value_iteration_action_elimination_heaps_lower_bound_approx(S, R, A9, P, gamma, epsilon);
			vector<double> VIAEHL_approx_solution = get<0>(VIAEHL_approx_solution_tuple);

			auto stop_VIAEHL = high_resolution_clock::now();
			auto duration_VIAEHL = duration_cast<milliseconds>(stop_VIAEHL - start_VIAEHL);

			stringstream_VIAEHL << to_string(A_num) << " " << duration_VIAEHL.count() << endl;

			// BAO
			A_type A8 = copy_A(A);
			auto start_BAO = high_resolution_clock::now();

			// V_type BAO_approx_solution_tuple = value_iteration_BAO(S, R, A8, P, gamma, epsilon);
			// vector<double> BAO_approx_solution = get<0>(BAO_approx_solution_tuple);

			auto stop_BAO = high_resolution_clock::now();
			auto duration_BAO = duration_cast<milliseconds>(stop_BAO - start_BAO);

			stringstream_BAO << to_string(A_num) << " " << duration_BAO.count() << endl;

			A_type A10 = copy_A(A);
			auto start_BAOSK = high_resolution_clock::now();

			V_type BAO_approx_solution_tuple1 = value_iteration_BAOSK(S, R, A10, P, gamma, epsilon);
			auto stop_BAOSK = high_resolution_clock::now();
			vector<double> BAO_approx_solution1 = get<0>(BAO_approx_solution_tuple1);
			auto duration_BAOSK = duration_cast<milliseconds>(stop_BAOSK - start_BAOSK);
			// cout<<"CheckpointBAO"<<endl;
			stringstream_BAON << to_string(A_num) << " " << duration_BAOSK.count() << endl;

			VI[0][k] += duration_VI.count();
			VI[1][k] += duration_VIU.count();
			VI[2][k] += duration_VIH.count();
			VI[3][k] += duration_BVI.count();
			VI[4][k] += duration_VIAE.count();
			VI[5][k] += duration_VIAEH.count();
			VI[6][k] += duration_VIAEHL.count();
			VI[7][k] += duration_BAO.count();
			VI[8][k] += duration_BAOSK.count();
			VI[9][k] += duration_VIHN.count();
			// They should in theory all be epsilon from true value, and therefore, at most be 2 * epsilon from each other
			/*if (abs_max_diff_vectors(V_heap_approxN, V_heap_approx) > (2 * epsilon)){
					printf("DIFFERENCE\n");
			}
			if (abs_max_diff_vectors(V_heap_approx, VIAEHL_approx_solution) > (2 * epsilon)){
					printf("DIFFERENCE\n");
			}
			if (abs_max_diff_vectors(VIAEHL_approx_solution, BAO_approx_solution1) > (2 * epsilon)){
					printf("DIFFERENCE\n");
			}
			//if (abs_max_diff_vectors(VIAEHL_approx_solution	, BAO_approx_solution) > (2 * epsilon)){
			//		printf("DIFFERENCE\n");
			//}*/
			k++;
		}
	}
	for (int k = 0; k < 50; k++)
	{
		// stringstream_VI  << VI[0][k]/5<< endl;
		// stringstream_VIU <<   VI[1][k]/5 << endl;
		stringstream_VIH << VI[2][k] / 5 << endl;
		// stringstream_BVI <<  VI[3][k]/5 << endl;
		// stringstream_VIAE <<  VI[4][k]/5 << endl;
		// stringstream_VIAEH <<  VI[5][k]/5 << endl;
		stringstream_VIAEHL << VI[6][k] / 5 << endl;
		// stringstream_BAO <<  VI[7][k]/5 << endl;
		stringstream_BAON << VI[8][k] / 5 << endl;
		// stringstream_VIHN <<  VI[9][k]/5 << endl;
	}
	// WRITE ALL DATA TO THEIR RESPECTVIE FILES
	// write_stringstream_to_file(stringstream_VI, output_stream_VI, file_name_VI);
	// write_stringstream_to_file(stringstream_VIU, output_stream_VIU, file_name_VIU);
	// write_stringstream_to_file(stringstream_BVI, output_stream_BVI, file_name_BVI);
	write_stringstream_to_file(stringstream_VIH, output_stream_VIH, file_name_VIH);
	// write_stringstream_to_file(stringstream_VIAE, output_stream_VIAE, file_name_VIAE);
	// write_stringstream_to_file(stringstream_VIAEH, output_stream_VIAEH, file_name_VIAEH);
	write_stringstream_to_file(stringstream_VIAEHL, output_stream_VIAEHL, file_name_VIAEHL);
	// write_stringstream_to_file(stringstream_BAO, output_stream_BAO, file_name_BAO);
	write_stringstream_to_file(stringstream_BAON, output_stream_BAON, file_name_BAON);
	// write_stringstream_to_file(stringstream_VIHN, output_stream_VIHN, file_name_VIHN);
}

// VARYING REWARD PROB
void write_meta_data_to_dat_file_reward_prob(ostringstream &string_stream, int S, int A_num, double epsilon, double gamma, double reward_factor, double upper_reward, double non_zero_transition, double action_prob, double reward_prob_starting_value, double reward_prob_finishing_value, double reward_prob_increment)
{
	time_t time_now = time(0);
	string_stream << "# META DATA" << endl;
	string_stream << "# " << endl;
	string_stream << "# "
				  << "experiment run at: " << ctime(&time_now);
	string_stream << "# " << endl;
	string_stream << "# "
				  << "gamma = " << gamma << endl;
	string_stream << "# "
				  << "epsilon = " << epsilon << endl;
	string_stream << "# "
				  << "S = " << S << endl;
	string_stream << "# "
				  << "A_num = " << A_num << endl;
	string_stream << "# "
				  << "non_zero_transition = " << non_zero_transition << endl;
	string_stream << "# "
				  << "reward_factor = " << reward_factor << endl;
	string_stream << "# "
				  << "upper_reward = " << upper_reward << endl;
	string_stream << "# "
				  << "action_prob = " << action_prob << endl;
	string_stream << "# " << endl;
	string_stream << "# VARIABLE IN THIS EXPERIMENT" << endl;
	string_stream << "# "
				  << "The probability that a reward is multiplied by reward_factor after initial sampling, reward_prob, is varied from " << reward_prob_starting_value << " to " << reward_prob_finishing_value << " with " << reward_prob_increment << " increment" << endl;
	string_stream << "# " << endl;
	string_stream << "# ACTUAL DATA" << endl;
	string_stream << "# reward_prob | microseconds" << endl;
}

void create_data_tables_rewards(string filename, int S, int A_num, double epsilon, double gamma, double reward_factor, double reward_prob, double upper_reward, double action_prob, double non_zero_transition)
{

	// the stringstreams to create the test for the files
	ostringstream stringstream_BVI;
	ostringstream stringstream_VI;
	ostringstream stringstream_VIU;
	ostringstream stringstream_VIH;
	ostringstream stringstream_VIAE;
	ostringstream stringstream_VIAEH;

	// the file output objects
	ofstream output_stream_BVI;
	ofstream output_stream_VI;
	ofstream output_stream_VIU;
	ofstream output_stream_VIH;
	ofstream output_stream_VIAE;
	ofstream output_stream_VIAEH;

	// set the name of the file to write to
	string file_name_BVI = "data_tables/reward_dist/" + filename + "_BVI.dat";
	string file_name_VI = "data_tables/reward_dist/" + filename + "_VI.dat";
	string file_name_VIU = "data_tables/reward_dist/" + filename + "_VIU.dat";
	string file_name_VIH = "data_tables/reward_dist/" + filename + "_VIH.dat";
	string file_name_VIAE = "data_tables/reward_dist/" + filename + "_VIAE.dat";
	string file_name_VIAEH = "data_tables/reward_dist/" + filename + "_VIAEH.dat";

	// The varying parameters
	double reward_prob_starting_value = 0.0;
	double reward_prob_finishing_value = 1.0;
	double reward_prob_increment = 0.01;

	write_meta_data_to_dat_file_reward_prob(stringstream_VI, S, A_num, epsilon, gamma, reward_factor, upper_reward, non_zero_transition, action_prob, reward_prob_starting_value, reward_prob_finishing_value, reward_prob_increment);
	write_meta_data_to_dat_file_reward_prob(stringstream_VIU, S, A_num, epsilon, gamma, reward_factor, upper_reward, non_zero_transition, action_prob, reward_prob_starting_value, reward_prob_finishing_value, reward_prob_increment);
	write_meta_data_to_dat_file_reward_prob(stringstream_VIH, S, A_num, epsilon, gamma, reward_factor, upper_reward, non_zero_transition, action_prob, reward_prob_starting_value, reward_prob_finishing_value, reward_prob_increment);
	write_meta_data_to_dat_file_reward_prob(stringstream_BVI, S, A_num, epsilon, gamma, reward_factor, upper_reward, non_zero_transition, action_prob, reward_prob_starting_value, reward_prob_finishing_value, reward_prob_increment);
	write_meta_data_to_dat_file_reward_prob(stringstream_VIAE, S, A_num, epsilon, gamma, reward_factor, upper_reward, non_zero_transition, action_prob, reward_prob_starting_value, reward_prob_finishing_value, reward_prob_increment);
	write_meta_data_to_dat_file_reward_prob(stringstream_VIAEH, S, A_num, epsilon, gamma, reward_factor, upper_reward, non_zero_transition, action_prob, reward_prob_starting_value, reward_prob_finishing_value, reward_prob_increment);

	for (double reward_prob = reward_prob_starting_value; reward_prob <= reward_prob_finishing_value; reward_prob = reward_prob + reward_prob_increment)
	{

		printf("Beginning iteration reward_prob = %f\n", reward_prob);

		// GENERATE THE MDP
		int seed = time(0);
		auto MDP = generate_random_MDP_with_variable_parameters_and_reward(S, A_num, action_prob, non_zero_transition, reward_factor, reward_prob, upper_reward, seed);
		R_type R = get<0>(MDP);
		A_type A = get<1>(MDP);
		P_type P = get<2>(MDP);

		// VI testing
		// TODO: make method that takes arguments and performs the writing of data, takes filename, MDP etc. as argument
		A_type A1 = copy_A(A);
		auto start_VI = high_resolution_clock::now();

		V_type V_approx_solution_tuple = value_iteration(S, R, A1, P, gamma, epsilon);
		vector<double> V_approx_solution = get<0>(V_approx_solution_tuple);

		auto stop_VI = high_resolution_clock::now();
		auto duration_VI = duration_cast<microseconds>(stop_VI - start_VI);

		stringstream_VI << to_string(reward_prob) << " " << duration_VI.count() << endl;

		// VIU testing
		A_type A6 = copy_A(A);
		auto start_VIU = high_resolution_clock::now();

		V_type V_approx_solution_upper_tuple = value_iteration_upper(S, R, A6, P, gamma, epsilon);
		vector<double> V_approx_solution_upper = get<0>(V_approx_solution_upper_tuple);

		auto stop_VIU = high_resolution_clock::now();
		auto duration_VIU = duration_cast<microseconds>(stop_VIU - start_VIU);

		stringstream_VIU << to_string(reward_prob) << " " << duration_VIU.count() << endl;

		// VIH testing
		A_type A2 = copy_A(A);
		auto start_VIH = high_resolution_clock::now();

		V_type V_heap_approx_tuple = value_iteration_with_heap(S, R, A2, P, gamma, epsilon);
		vector<double> V_heap_approx = get<0>(V_heap_approx_tuple);

		auto stop_VIH = high_resolution_clock::now();
		auto duration_VIH = duration_cast<microseconds>(stop_VIH - start_VIH);

		stringstream_VIH << to_string(reward_prob) << " " << duration_VIH.count() << endl;

		// BVI
		A_type A3 = copy_A(A);
		auto start_BVI = high_resolution_clock::now();

		V_type V_bounded_approx_solution_tuple = bounded_value_iteration(S, R, A3, P, gamma, epsilon);
		vector<double> V_bounded_approx_solution = get<0>(V_bounded_approx_solution_tuple);

		auto stop_BVI = high_resolution_clock::now();
		auto duration_BVI = duration_cast<microseconds>(stop_BVI - start_BVI);

		stringstream_BVI << to_string(reward_prob) << " " << duration_BVI.count() << endl;

		// VIAE
		A_type A4 = copy_A(A);
		auto start_VIAE = high_resolution_clock::now();

		V_type V_AE_approx_solution_tuple = value_iteration_action_elimination(S, R, A4, P, gamma, epsilon);
		vector<double> V_AE_approx_solution = get<0>(V_AE_approx_solution_tuple);

		auto stop_VIAE = high_resolution_clock::now();
		auto duration_VIAE = duration_cast<microseconds>(stop_VIAE - start_VIAE);

		stringstream_VIAE << to_string(reward_prob) << " " << duration_VIAE.count() << endl;

		// VIAEH
		A_type A5 = copy_A(A);
		auto start_VIAEH = high_resolution_clock::now();

		V_type V_AE_H_approx_solution_tuple = value_iteration_action_elimination_heaps(S, R, A5, P, gamma, epsilon);
		vector<double> V_AE_H_approx_solution = get<0>(V_AE_H_approx_solution_tuple);

		auto stop_VIAEH = high_resolution_clock::now();
		auto duration_VIAEH = duration_cast<microseconds>(stop_VIAEH - start_VIAEH);

		stringstream_VIAEH << to_string(reward_prob) << " " << duration_VIAEH.count() << endl;

		// They should in theory all be epsilon from true value, and therefore, at most be 2 * epsilon from each other
		if (abs_max_diff_vectors(V_approx_solution, V_heap_approx) > (2 * epsilon))
		{
			printf("DIFFERENCE\n");
		}
		if (abs_max_diff_vectors(V_AE_approx_solution, V_AE_H_approx_solution) > (2 * epsilon))
		{
			printf("DIFFERENCE\n");
		}
		if (abs_max_diff_vectors(V_bounded_approx_solution, V_approx_solution_upper) > (2 * epsilon))
		{
			printf("DIFFERENCE\n");
		}
	}

	// WRITE ALL DATA TO THEIR RESPECTVIE FILES
	write_stringstream_to_file(stringstream_VI, output_stream_VI, file_name_VI);
	write_stringstream_to_file(stringstream_VIU, output_stream_VIU, file_name_VIU);
	write_stringstream_to_file(stringstream_BVI, output_stream_BVI, file_name_BVI);
	write_stringstream_to_file(stringstream_VIH, output_stream_VIH, file_name_VIH);
	write_stringstream_to_file(stringstream_VIAE, output_stream_VIAE, file_name_VIAE);
	write_stringstream_to_file(stringstream_VIAEH, output_stream_VIAEH, file_name_VIAEH);
}

// VARYING TRANSITION PROB
void write_meta_data_to_dat_file_transition_prob(ostringstream &string_stream, int S, int A_num, double epsilon, double gamma, double upper_reward, double action_prob, double transition_prob_starting_value, double transition_prob_finishing_value, double transition_prob_increment)
{
	time_t time_now = time(0);
	string_stream << "# META DATA" << endl;
	string_stream << "# " << endl;
	string_stream << "# "
				  << "experiment run at: " << ctime(&time_now);
	string_stream << "# " << endl;
	string_stream << "# "
				  << "gamma = " << gamma << endl;
	string_stream << "# "
				  << "epsilon = " << epsilon << endl;
	string_stream << "# "
				  << "S = " << S << endl;
	string_stream << "# "
				  << "A_num = " << A_num << endl;
	string_stream << "# "
				  << "upper_reward = " << upper_reward << endl;
	string_stream << "# "
				  << "action_prob = " << action_prob << endl;
	string_stream << "# " << endl;
	string_stream << "# VARIABLE IN THIS EXPERIMENT" << endl;
	string_stream << "# "
				  << "The probability that a state has a non-zero transition probability, transition_prob, is varied from " << transition_prob_starting_value << " to " << transition_prob_finishing_value << " with " << transition_prob_increment << " increment" << endl;
	string_stream << "# " << endl;
	string_stream << "# ACTUAL DATA" << endl;
	string_stream << "# transition_prob | microseconds" << endl;
}

void create_data_tables_transition_prob(string filename, int S, int A_num, double epsilon, double gamma, double upper_reward)
{

	// the stringstreams to create the test for the files
	ostringstream stringstream_BVI;
	ostringstream stringstream_VI;
	ostringstream stringstream_VIU;
	ostringstream stringstream_VIH;
	ostringstream stringstream_VIAE;
	ostringstream stringstream_VIAEH;

	// the file output objects
	ofstream output_stream_BVI;
	ofstream output_stream_VI;
	ofstream output_stream_VIU;
	ofstream output_stream_VIH;
	ofstream output_stream_VIAE;
	ofstream output_stream_VIAEH;

	// set the name of the file to write to
	string file_name_BVI = "data_tables/transition_prob/" + filename + "_BVI.dat";
	string file_name_VI = "data_tables/transition_prob/" + filename + "_VI.dat";
	string file_name_VIU = "data_tables/transition_prob/" + filename + "_VIU.dat";
	string file_name_VIH = "data_tables/transition_prob/" + filename + "_VIH.dat";
	string file_name_VIAE = "data_tables/transition_prob/" + filename + "_VIAE.dat";
	string file_name_VIAEH = "data_tables/transition_prob/" + filename + "_VIAEH.dat";

	// The varying parameters
	double transition_prob_starting_value = 0.1;
	double transition_prob_finishing_value = 1.0;
	double transition_prob_increment = 0.05;

	// hardcoded parameter
	double action_prob = 1.0;

	write_meta_data_to_dat_file_transition_prob(stringstream_VI, S, A_num, epsilon, gamma, upper_reward, action_prob, transition_prob_starting_value, transition_prob_finishing_value, transition_prob_increment);
	write_meta_data_to_dat_file_transition_prob(stringstream_VIU, S, A_num, epsilon, gamma, upper_reward, action_prob, transition_prob_starting_value, transition_prob_finishing_value, transition_prob_increment);
	write_meta_data_to_dat_file_transition_prob(stringstream_VIH, S, A_num, epsilon, gamma, upper_reward, action_prob, transition_prob_starting_value, transition_prob_finishing_value, transition_prob_increment);
	write_meta_data_to_dat_file_transition_prob(stringstream_BVI, S, A_num, epsilon, gamma, upper_reward, action_prob, transition_prob_starting_value, transition_prob_finishing_value, transition_prob_increment);
	write_meta_data_to_dat_file_transition_prob(stringstream_VIAE, S, A_num, epsilon, gamma, upper_reward, action_prob, transition_prob_starting_value, transition_prob_finishing_value, transition_prob_increment);
	write_meta_data_to_dat_file_transition_prob(stringstream_VIAEH, S, A_num, epsilon, gamma, upper_reward, action_prob, transition_prob_starting_value, transition_prob_finishing_value, transition_prob_increment);

	for (double transition_prob = transition_prob_starting_value; transition_prob <= transition_prob_finishing_value; transition_prob = transition_prob + transition_prob_increment)
	{

		printf("Beginning iteration transition_prob = %f\n", transition_prob);

		// GENERATE THE MDP
		int seed = time(0);
		auto MDP = generate_random_MDP_with_variable_parameters(S, A_num, action_prob, transition_prob, upper_reward, seed);
		R_type R = get<0>(MDP);
		A_type A = get<1>(MDP);
		P_type P = get<2>(MDP);

		// VI testing
		// TODO: make method that takes arguments and performs the writing of data, takes filename, MDP etc. as argument
		A_type A1 = copy_A(A);
		auto start_VI = high_resolution_clock::now();

		V_type V_approx_solution_tuple = value_iteration(S, R, A1, P, gamma, epsilon);
		vector<double> V_approx_solution = get<0>(V_approx_solution_tuple);

		auto stop_VI = high_resolution_clock::now();
		auto duration_VI = duration_cast<microseconds>(stop_VI - start_VI);

		stringstream_VI << to_string(transition_prob) << " " << duration_VI.count() << endl;

		// VIU testing
		A_type A6 = copy_A(A);
		auto start_VIU = high_resolution_clock::now();

		V_type V_approx_solution_upper_tuple = value_iteration_upper(S, R, A6, P, gamma, epsilon);
		vector<double> V_approx_solution_upper = get<0>(V_approx_solution_upper_tuple);

		auto stop_VIU = high_resolution_clock::now();
		auto duration_VIU = duration_cast<microseconds>(stop_VIU - start_VIU);

		stringstream_VIU << to_string(transition_prob) << " " << duration_VIU.count() << endl;

		// VIH testing
		A_type A2 = copy_A(A);
		auto start_VIH = high_resolution_clock::now();

		V_type V_heap_approx_tuple = value_iteration_with_heap(S, R, A2, P, gamma, epsilon);
		vector<double> V_heap_approx = get<0>(V_heap_approx_tuple);

		auto stop_VIH = high_resolution_clock::now();
		auto duration_VIH = duration_cast<microseconds>(stop_VIH - start_VIH);

		stringstream_VIH << to_string(transition_prob) << " " << duration_VIH.count() << endl;

		// BVI
		A_type A3 = copy_A(A);
		auto start_BVI = high_resolution_clock::now();

		V_type V_bounded_approx_solution_tuple = bounded_value_iteration(S, R, A3, P, gamma, epsilon);
		vector<double> V_bounded_approx_solution = get<0>(V_bounded_approx_solution_tuple);

		auto stop_BVI = high_resolution_clock::now();
		auto duration_BVI = duration_cast<microseconds>(stop_BVI - start_BVI);

		stringstream_BVI << to_string(transition_prob) << " " << duration_BVI.count() << endl;

		// VIAE
		A_type A4 = copy_A(A);
		auto start_VIAE = high_resolution_clock::now();

		V_type V_AE_approx_solution_tuple = value_iteration_action_elimination(S, R, A4, P, gamma, epsilon);
		vector<double> V_AE_approx_solution = get<0>(V_AE_approx_solution_tuple);

		auto stop_VIAE = high_resolution_clock::now();
		auto duration_VIAE = duration_cast<microseconds>(stop_VIAE - start_VIAE);

		stringstream_VIAE << to_string(transition_prob) << " " << duration_VIAE.count() << endl;

		// VIAEH
		A_type A5 = copy_A(A);
		auto start_VIAEH = high_resolution_clock::now();

		V_type V_AE_H_approx_solution_tuple = value_iteration_action_elimination_heaps(S, R, A5, P, gamma, epsilon);
		vector<double> V_AE_H_approx_solution = get<0>(V_AE_H_approx_solution_tuple);

		auto stop_VIAEH = high_resolution_clock::now();
		auto duration_VIAEH = duration_cast<microseconds>(stop_VIAEH - start_VIAEH);

		stringstream_VIAEH << to_string(transition_prob) << " " << duration_VIAEH.count() << endl;

		// They should in theory all be epsilon from true value, and therefore, at most be 2 * epsilon from each other
		if (abs_max_diff_vectors(V_approx_solution, V_heap_approx) > (2 * epsilon))
		{
			printf("DIFFERENCE\n");
		}
		if (abs_max_diff_vectors(V_AE_approx_solution, V_AE_H_approx_solution) > (2 * epsilon))
		{
			printf("DIFFERENCE\n");
		}
		if (abs_max_diff_vectors(V_bounded_approx_solution, V_approx_solution_upper) > (2 * epsilon))
		{
			printf("DIFFERENCE\n");
		}
	}

	// WRITE ALL DATA TO THEIR RESPECTVIE FILES
	write_stringstream_to_file(stringstream_VI, output_stream_VI, file_name_VI);
	write_stringstream_to_file(stringstream_VIU, output_stream_VIU, file_name_VIU);
	write_stringstream_to_file(stringstream_BVI, output_stream_BVI, file_name_BVI);
	write_stringstream_to_file(stringstream_VIH, output_stream_VIH, file_name_VIH);
	write_stringstream_to_file(stringstream_VIAE, output_stream_VIAE, file_name_VIAE);
	write_stringstream_to_file(stringstream_VIAEH, output_stream_VIAEH, file_name_VIAEH);
}

// NUMBER OF STATES - ITERATIONS PLOT

void write_meta_data_to_dat_file_number_of_states_iterations(ostringstream &string_stream, int A_num, double epsilon, double gamma, double upper_reward, double non_zero_transition, double action_prob, int S_starting_value, int S_finishing_value, int S_increment)
{
	time_t time_now = time(0);
	string_stream << "# META DATA" << endl;
	string_stream << "# " << endl;
	string_stream << "# "
				  << "experiment run at: " << ctime(&time_now);
	string_stream << "# " << endl;
	string_stream << "# "
				  << "gamma = " << gamma << endl;
	string_stream << "# "
				  << "epsilon = " << epsilon << endl;
	string_stream << "# "
				  << "A = " << A_num << endl;
	string_stream << "# "
				  << "non_zero_transition = " << non_zero_transition << endl;
	string_stream << "# "
				  << "upper_reward = " << upper_reward << endl;
	string_stream << "# "
				  << "action_prob = " << action_prob << endl;
	string_stream << "# " << endl;
	string_stream << "# VARIABLE IN THIS EXPERIMENT" << endl;
	string_stream << "# "
				  << "Number of states, S, is varied from " << S_starting_value << " to " << S_finishing_value << " with " << S_increment << " increment" << endl;
	string_stream << "# " << endl;
	string_stream << "# ACTUAL DATA" << endl;
	string_stream << "# number of states | iterations" << endl;
}

void create_data_tables_number_of_states_iterations(string filename, int S_max, int A_num, double epsilon, double gamma, double upper_reward, double non_zero_transition)
{

	// the stringstreams to create the test for the files
	ostringstream stringstream_BVI;
	ostringstream stringstream_VI;
	ostringstream stringstream_VIU;
	ostringstream stringstream_VIH;
	ostringstream stringstream_VIAE;
	ostringstream stringstream_VIAEH;

	// the file output objects
	ofstream output_stream_BVI;
	ofstream output_stream_VI;
	ofstream output_stream_VIU;
	ofstream output_stream_VIH;
	ofstream output_stream_VIAE;
	ofstream output_stream_VIAEH;

	// set the name of the file to write to
	string file_name_BVI = "data_tables/number_of_states_iterations/" + filename + "_BVI.dat";
	string file_name_VI = "data_tables/number_of_states_iterations/" + filename + "_VI.dat";
	string file_name_VIU = "data_tables/number_of_states_iterations/" + filename + "_VIU.dat";
	string file_name_VIH = "data_tables/number_of_states_iterations/" + filename + "_VIH.dat";
	string file_name_VIAE = "data_tables/number_of_states_iterations/" + filename + "_VIAE.dat";
	string file_name_VIAEH = "data_tables/number_of_states_iterations/" + filename + "_VIAEH.dat";

	// The varying parameters
	int S_starting_value = 50;
	int S_finishing_value = S_max;
	int S_increment = 50;

	// hardcoded parameter
	double action_prob = 1.0;

	// write meta data to all stringstreams as first in their respective files

	write_meta_data_to_dat_file_number_of_states_iterations(stringstream_VI, A_num, epsilon, gamma, upper_reward, non_zero_transition, action_prob, S_starting_value, S_finishing_value, S_increment);
	write_meta_data_to_dat_file_number_of_states_iterations(stringstream_VIU, A_num, epsilon, gamma, upper_reward, non_zero_transition, action_prob, S_starting_value, S_finishing_value, S_increment);
	write_meta_data_to_dat_file_number_of_states_iterations(stringstream_BVI, A_num, epsilon, gamma, upper_reward, non_zero_transition, action_prob, S_starting_value, S_finishing_value, S_increment);
	write_meta_data_to_dat_file_number_of_states_iterations(stringstream_VIH, A_num, epsilon, gamma, upper_reward, non_zero_transition, action_prob, S_starting_value, S_finishing_value, S_increment);
	write_meta_data_to_dat_file_number_of_states_iterations(stringstream_VIAE, A_num, epsilon, gamma, upper_reward, non_zero_transition, action_prob, S_starting_value, S_finishing_value, S_increment);
	write_meta_data_to_dat_file_number_of_states_iterations(stringstream_VIAEH, A_num, epsilon, gamma, upper_reward, non_zero_transition, action_prob, S_starting_value, S_finishing_value, S_increment);

	for (int S = S_starting_value; S <= S_finishing_value; S = S + S_increment)
	{

		printf("\nBeginning iteration S = %d\n", S);

		// GENERATE THE MDP
		int seed = time(0);
		printf("\nseed: %d\n", seed);
		auto MDP = generate_random_MDP_with_variable_parameters(S, A_num, action_prob, non_zero_transition, upper_reward, seed);
		R_type R = get<0>(MDP);
		A_type A = get<1>(MDP);
		P_type P = get<2>(MDP);

		// iterations test printing
		float R_max = find_max_R(R);
		float R_min = find_min_R(R);
		float iterations_bound = log(R_max / ((1.0 - gamma) * epsilon)) / log(1.0 / gamma);
		printf("R_max is %f\n", R_max);
		printf("R_min is %f\n", R_min);
		printf("upper bound is %f\n", R_max / (1.0 - gamma));
		printf("lower bound is %f\n", R_min / (1.0 - gamma));
		printf("iterations bound: %f\n", iterations_bound);

		// VI testing
		// TODO: make method that takes arguments and performs the writing of data, takes filename, MDP etc. as argument
		A_type A1 = copy_A(A);
		auto start_VI = high_resolution_clock::now();

		V_type V_approx_solution_tuple = value_iteration(S, R, A1, P, gamma, epsilon);
		vector<double> V_approx_solution = get<0>(V_approx_solution_tuple);
		int V_approx_solution_iterations = get<1>(V_approx_solution_tuple);

		printf("VI iterations: %d\n", V_approx_solution_iterations);

		auto stop_VI = high_resolution_clock::now();
		auto duration_VI = duration_cast<microseconds>(stop_VI - start_VI);

		stringstream_VI << to_string(S) << " " << V_approx_solution_iterations << endl;

		// VIU testing
		A_type A6 = copy_A(A);
		auto start_VIU = high_resolution_clock::now();

		V_type V_approx_solution_upper_tuple = value_iteration_upper(S, R, A6, P, gamma, epsilon);
		vector<double> V_approx_solution_upper = get<0>(V_approx_solution_upper_tuple);
		int V_approx_solution_upper_iterations = get<1>(V_approx_solution_upper_tuple);

		printf("VIU iterations: %d\n", V_approx_solution_upper_iterations);

		auto stop_VIU = high_resolution_clock::now();
		auto duration_VIU = duration_cast<microseconds>(stop_VIU - start_VIU);

		stringstream_VIU << to_string(S) << " " << V_approx_solution_upper_iterations << endl;

		// VIH testing
		A_type A2 = copy_A(A);
		auto start_VIH = high_resolution_clock::now();

		V_type V_heap_approx_tuple = value_iteration_with_heap(S, R, A2, P, gamma, epsilon);
		vector<double> V_heap_approx = get<0>(V_heap_approx_tuple);
		int V_heap_approx_iterations = get<1>(V_heap_approx_tuple);

		printf("VIH iterations: %d\n", V_heap_approx_iterations);

		auto stop_VIH = high_resolution_clock::now();
		auto duration_VIH = duration_cast<microseconds>(stop_VIH - start_VIH);

		stringstream_VIH << to_string(S) << " " << V_heap_approx_iterations << endl;

		// BVI
		A_type A3 = copy_A(A);
		auto start_BVI = high_resolution_clock::now();

		V_type V_bounded_approx_solution_tuple = bounded_value_iteration(S, R, A3, P, gamma, epsilon);
		vector<double> V_bounded_approx_solution = get<0>(V_bounded_approx_solution_tuple);
		int V_bounded_approx_solution_iterations = get<1>(V_bounded_approx_solution_tuple);

		printf("BVI iterations: %d\n", V_bounded_approx_solution_iterations);

		auto stop_BVI = high_resolution_clock::now();
		auto duration_BVI = duration_cast<microseconds>(stop_BVI - start_BVI);

		stringstream_BVI << to_string(S) << " " << V_bounded_approx_solution_iterations << endl;

		// VIAE
		A_type A4 = copy_A(A);
		auto start_VIAE = high_resolution_clock::now();

		V_type V_AE_approx_solution_tuple = value_iteration_action_elimination(S, R, A4, P, gamma, epsilon);
		vector<double> V_AE_approx_solution = get<0>(V_AE_approx_solution_tuple);
		int V_AE_approx_solution_iterations = get<1>(V_AE_approx_solution_tuple);

		printf("VIAE iterations: %d\n", V_AE_approx_solution_iterations);

		auto stop_VIAE = high_resolution_clock::now();
		auto duration_VIAE = duration_cast<microseconds>(stop_VIAE - start_VIAE);

		stringstream_VIAE << to_string(S) << " " << V_AE_approx_solution_iterations << endl;

		// VIAEH
		A_type A5 = copy_A(A);
		auto start_VIAEH = high_resolution_clock::now();

		V_type V_AE_H_approx_solution_tuple = value_iteration_action_elimination_heaps(S, R, A5, P, gamma, epsilon);
		vector<double> V_AE_H_approx_solution = get<0>(V_AE_H_approx_solution_tuple);
		int V_AE_H_approx_solution_iterations = get<1>(V_AE_H_approx_solution_tuple);

		printf("VIAEH iterations: %d\n", V_AE_H_approx_solution_iterations);

		auto stop_VIAEH = high_resolution_clock::now();
		auto duration_VIAEH = duration_cast<microseconds>(stop_VIAEH - start_VIAEH);

		stringstream_VIAEH << to_string(S) << " " << V_AE_H_approx_solution_iterations << endl;

		// They should in theory all be epsilon from true value, and therefore, at most be 2 * epsilon from each other
		if (abs_max_diff_vectors(V_approx_solution, V_heap_approx) > (2 * epsilon))
		{
			printf("DIFFERENCE\n");
		}
		if (abs_max_diff_vectors(V_AE_approx_solution, V_AE_H_approx_solution) > (2 * epsilon))
		{
			printf("DIFFERENCE\n");
		}
		if (abs_max_diff_vectors(V_bounded_approx_solution, V_approx_solution_upper) > (2 * epsilon))
		{
			printf("DIFFERENCE\n");
		}
	}

	// WRITE ALL DATA TO THEIR RESPECTVIE FILES
	write_stringstream_to_file(stringstream_VI, output_stream_VI, file_name_VI);
	write_stringstream_to_file(stringstream_VIU, output_stream_VIU, file_name_VIU);
	write_stringstream_to_file(stringstream_BVI, output_stream_BVI, file_name_BVI);
	write_stringstream_to_file(stringstream_VIH, output_stream_VIH, file_name_VIH);
	write_stringstream_to_file(stringstream_VIAE, output_stream_VIAE, file_name_VIAE);
	write_stringstream_to_file(stringstream_VIAEH, output_stream_VIAEH, file_name_VIAEH);
}

// VARYING number of non-zero transition states
void write_meta_data_to_dat_file_number_of_transitions(ostringstream &string_stream, int S, int A_num, double epsilon, double gamma, double upper_reward, double action_prob, int number_of_transitions_starting_value, int number_of_transitions_finishing_value, int number_of_transitions_increment)
{
	time_t time_now = time(0);
	string_stream << "# META DATA" << endl;
	string_stream << "# " << endl;
	string_stream << "# "
				  << "experiment run at: " << ctime(&time_now);
	string_stream << "# " << endl;
	string_stream << "# "
				  << "gamma = " << gamma << endl;
	string_stream << "# "
				  << "epsilon = " << epsilon << endl;
	string_stream << "# "
				  << "S = " << S << endl;
	string_stream << "# "
				  << "A = " << A_num << endl;
	string_stream << "# "
				  << "action_prob = " << action_prob << endl;
	string_stream << "# "
				  << "upper_reward = " << upper_reward << endl;
	string_stream << "# " << endl;
	string_stream << "# VARIABLE IN THIS EXPERIMENT" << endl;
	string_stream << "# "
				  << "number of transitions is varied from " << number_of_transitions_starting_value << " to " << number_of_transitions_finishing_value << " with " << number_of_transitions_increment << " increment" << endl;
	string_stream << "# " << endl;
	string_stream << "# ACTUAL DATA" << endl;
	string_stream << "# number of transitions | microseconds" << endl;
}

void create_data_tables_number_of_transitions(string filename, int S, int A_num, double epsilon, double gamma, double upper_reward, double action_prob, int max_transitions)
{

	// the stringstreams to create the test for the files
	ostringstream stringstream_BVI;
	ostringstream stringstream_VI;
	ostringstream stringstream_VIU;
	ostringstream stringstream_VIH;
	ostringstream stringstream_VIAE;
	ostringstream stringstream_VIAEH;

	// the file output objects
	ofstream output_stream_BVI;
	ofstream output_stream_VI;
	ofstream output_stream_VIU;
	ofstream output_stream_VIH;
	ofstream output_stream_VIAE;
	ofstream output_stream_VIAEH;

	// set the name of the file to write to
	string file_name_BVI = "data_tables/number_of_transitions/" + filename + "_BVI.dat";
	string file_name_VI = "data_tables/number_of_transitions/" + filename + "_VI.dat";
	string file_name_VIU = "data_tables/number_of_transitions/" + filename + "_VIU.dat";
	string file_name_VIH = "data_tables/number_of_transitions/" + filename + "_VIH.dat";
	string file_name_VIAE = "data_tables/number_of_transitions/" + filename + "_VIAE.dat";
	string file_name_VIAEH = "data_tables/number_of_transitions/" + filename + "_VIAEH.dat";

	int number_of_transitions_starting_value = 10;
	int number_of_transitions_finishing_value = max_transitions;
	int number_of_transitions_increment = 10;

	// write meta data to all stringstreams as first in their respective files
	write_meta_data_to_dat_file_number_of_transitions(stringstream_VI, S, A_num, epsilon, gamma, upper_reward, action_prob, number_of_transitions_starting_value, number_of_transitions_finishing_value, number_of_transitions_increment);
	write_meta_data_to_dat_file_number_of_transitions(stringstream_VIU, S, A_num, epsilon, gamma, upper_reward, action_prob, number_of_transitions_starting_value, number_of_transitions_finishing_value, number_of_transitions_increment);
	write_meta_data_to_dat_file_number_of_transitions(stringstream_VIH, S, A_num, epsilon, gamma, upper_reward, action_prob, number_of_transitions_starting_value, number_of_transitions_finishing_value, number_of_transitions_increment);
	write_meta_data_to_dat_file_number_of_transitions(stringstream_BVI, S, A_num, epsilon, gamma, upper_reward, action_prob, number_of_transitions_starting_value, number_of_transitions_finishing_value, number_of_transitions_increment);
	write_meta_data_to_dat_file_number_of_transitions(stringstream_VIAE, S, A_num, epsilon, gamma, upper_reward, action_prob, number_of_transitions_starting_value, number_of_transitions_finishing_value, number_of_transitions_increment);
	write_meta_data_to_dat_file_number_of_transitions(stringstream_VIAEH, S, A_num, epsilon, gamma, upper_reward, action_prob, number_of_transitions_starting_value, number_of_transitions_finishing_value, number_of_transitions_increment);

	for (int number_of_transitions = number_of_transitions_starting_value; number_of_transitions <= number_of_transitions_finishing_value; number_of_transitions = number_of_transitions + number_of_transitions_increment)
	{

		// status message of experiment
		printf("Beginning iteration number_of_transitions= %d\n", number_of_transitions);

		// GENERATE THE MDP FROM CURRENT TIME SEED
		int seed = time(0);
		auto MDP = generate_random_MDP_with_variable_parameters_fixed_nonzero_trans_states(S, A_num, action_prob, number_of_transitions, upper_reward, seed);
		R_type R = get<0>(MDP);
		A_type A = get<1>(MDP);
		P_type P = get<2>(MDP);

		// VI testing
		// TODO: make method that takes arguments and performs the writing of data, takes filename, MDP etc. as argument
		A_type A1 = copy_A(A);
		auto start_VI = high_resolution_clock::now();

		V_type V_approx_solution_tuple = value_iteration(S, R, A1, P, gamma, epsilon);
		vector<double> V_approx_solution = get<0>(V_approx_solution_tuple);

		auto stop_VI = high_resolution_clock::now();
		auto duration_VI = duration_cast<microseconds>(stop_VI - start_VI);

		stringstream_VI << to_string(number_of_transitions) << " " << duration_VI.count() << endl;

		// VIU testing
		A_type A6 = copy_A(A);
		auto start_VIU = high_resolution_clock::now();

		V_type V_approx_solution_upper_tuple = value_iteration_upper(S, R, A6, P, gamma, epsilon);
		vector<double> V_approx_solution_upper = get<0>(V_approx_solution_upper_tuple);

		auto stop_VIU = high_resolution_clock::now();
		auto duration_VIU = duration_cast<microseconds>(stop_VIU - start_VIU);

		stringstream_VIU << to_string(number_of_transitions) << " " << duration_VIU.count() << endl;

		// VIH testing
		A_type A2 = copy_A(A);
		auto start_VIH = high_resolution_clock::now();

		V_type V_heap_approx_tuple = value_iteration_with_heap(S, R, A2, P, gamma, epsilon);
		vector<double> V_heap_approx = get<0>(V_heap_approx_tuple);

		auto stop_VIH = high_resolution_clock::now();
		auto duration_VIH = duration_cast<microseconds>(stop_VIH - start_VIH);

		stringstream_VIH << to_string(number_of_transitions) << " " << duration_VIH.count() << endl;

		// BVI
		A_type A3 = copy_A(A);
		auto start_BVI = high_resolution_clock::now();

		V_type V_bounded_approx_solution_tuple = bounded_value_iteration(S, R, A3, P, gamma, epsilon);
		vector<double> V_bounded_approx_solution = get<0>(V_bounded_approx_solution_tuple);

		auto stop_BVI = high_resolution_clock::now();
		auto duration_BVI = duration_cast<microseconds>(stop_BVI - start_BVI);

		stringstream_BVI << to_string(number_of_transitions) << " " << duration_BVI.count() << endl;

		// VIAE
		A_type A4 = copy_A(A);
		auto start_VIAE = high_resolution_clock::now();

		V_type V_AE_approx_solution_tuple = value_iteration_action_elimination(S, R, A4, P, gamma, epsilon);
		vector<double> V_AE_approx_solution = get<0>(V_AE_approx_solution_tuple);

		auto stop_VIAE = high_resolution_clock::now();
		auto duration_VIAE = duration_cast<microseconds>(stop_VIAE - start_VIAE);

		stringstream_VIAE << to_string(number_of_transitions) << " " << duration_VIAE.count() << endl;

		// VIAEH
		A_type A5 = copy_A(A);
		auto start_VIAEH = high_resolution_clock::now();

		V_type V_AE_H_approx_solution_tuple = value_iteration_action_elimination_heaps(S, R, A5, P, gamma, epsilon);
		vector<double> V_AE_H_approx_solution = get<0>(V_AE_H_approx_solution_tuple);

		auto stop_VIAEH = high_resolution_clock::now();
		auto duration_VIAEH = duration_cast<microseconds>(stop_VIAEH - start_VIAEH);

		stringstream_VIAEH << to_string(number_of_transitions) << " " << duration_VIAEH.count() << endl;

		// They should in theory all be epsilon from true value, and therefore, at most be 2 * epsilon from each other
		if (abs_max_diff_vectors(V_approx_solution, V_heap_approx) > (2 * epsilon))
		{
			printf("DIFFERENCE\n");
		}
		if (abs_max_diff_vectors(V_AE_approx_solution, V_AE_H_approx_solution) > (2 * epsilon))
		{
			printf("DIFFERENCE\n");
		}
		if (abs_max_diff_vectors(V_bounded_approx_solution, V_approx_solution_upper) > (2 * epsilon))
		{
			printf("DIFFERENCE\n");
		}
	}

	// WRITE ALL DATA TO THEIR RESPECTVIE FILES
	write_stringstream_to_file(stringstream_VI, output_stream_VI, file_name_VI);
	write_stringstream_to_file(stringstream_VIU, output_stream_VIU, file_name_VIU);
	write_stringstream_to_file(stringstream_BVI, output_stream_BVI, file_name_BVI);
	write_stringstream_to_file(stringstream_VIH, output_stream_VIH, file_name_VIH);
	write_stringstream_to_file(stringstream_VIAE, output_stream_VIAE, file_name_VIAE);
	write_stringstream_to_file(stringstream_VIAEH, output_stream_VIAEH, file_name_VIAEH);
}

// VARYING NUMBER OF ACTIONS
void write_meta_data_to_dat_file_number_of_actions_iterations(ostringstream &string_stream, int S, double epsilon, double gamma, double upper_reward, double non_zero_transition, double action_prob, int A_starting_value, int A_finishing_value, int A_increment)
{
	time_t time_now = time(0);
	string_stream << "# META DATA" << endl;
	string_stream << "# " << endl;
	string_stream << "# "
				  << "experiment run at: " << ctime(&time_now);
	string_stream << "# " << endl;
	string_stream << "# "
				  << "gamma = " << gamma << endl;
	string_stream << "# "
				  << "epsilon = " << epsilon << endl;
	string_stream << "# "
				  << "S = " << S << endl;
	string_stream << "# "
				  << "non_zero_transition = " << non_zero_transition << endl;
	string_stream << "# "
				  << "upper_reward = " << upper_reward << endl;
	string_stream << "# "
				  << "action_prob = " << action_prob << endl;
	string_stream << "# " << endl;
	string_stream << "# VARIABLE IN THIS EXPERIMENT" << endl;
	string_stream << "# "
				  << "Number of actions, A, is varied from " << A_starting_value << " to " << A_finishing_value << " with " << A_increment << " increment" << endl;
	string_stream << "# " << endl;
	string_stream << "# ACTUAL DATA" << endl;
	string_stream << "# number of actions | iterations" << endl;
}

void create_data_tables_number_of_actions_iterations(string filename, int S, int A_max, double epsilon, double gamma, double upper_reward, double non_zero_transition)
{

	// the stringstreams to create the test for the files
	ostringstream stringstream_BVI;
	ostringstream stringstream_VI;
	ostringstream stringstream_VIU;
	ostringstream stringstream_VIH;
	ostringstream stringstream_VIAE;
	ostringstream stringstream_VIAEH;

	// the file output objects
	ofstream output_stream_BVI;
	ofstream output_stream_VI;
	ofstream output_stream_VIU;
	ofstream output_stream_VIH;
	ofstream output_stream_VIAE;
	ofstream output_stream_VIAEH;

	// set the name of the file to write to
	string file_name_BVI = "data_tables/number_of_actions_iterations/" + filename + "_BVI.dat";
	string file_name_VI = "data_tables/number_of_actions_iterations/" + filename + "_VI.dat";
	string file_name_VIU = "data_tables/number_of_actions_iterations/" + filename + "_VIU.dat";
	string file_name_VIH = "data_tables/number_of_actions_iterations/" + filename + "_VIH.dat";
	string file_name_VIAE = "data_tables/number_of_actions_iterations/" + filename + "_VIAE.dat";
	string file_name_VIAEH = "data_tables/number_of_actions_iterations/" + filename + "_VIAEH.dat";

	// The varying parameters
	int A_starting_value = 50;
	int A_finishing_value = A_max;
	int A_increment = 50;

	// hardcoded parameter
	double action_prob = 1.0;

	// write meta data to all stringstreams as first in their respective files

	write_meta_data_to_dat_file_number_of_actions_iterations(stringstream_VI, S, epsilon, gamma, upper_reward, non_zero_transition, action_prob, A_starting_value, A_finishing_value, A_increment);
	write_meta_data_to_dat_file_number_of_actions_iterations(stringstream_VIU, S, epsilon, gamma, upper_reward, non_zero_transition, action_prob, A_starting_value, A_finishing_value, A_increment);
	write_meta_data_to_dat_file_number_of_actions_iterations(stringstream_BVI, S, epsilon, gamma, upper_reward, non_zero_transition, action_prob, A_starting_value, A_finishing_value, A_increment);
	write_meta_data_to_dat_file_number_of_actions_iterations(stringstream_VIH, S, epsilon, gamma, upper_reward, non_zero_transition, action_prob, A_starting_value, A_finishing_value, A_increment);
	write_meta_data_to_dat_file_number_of_actions_iterations(stringstream_VIAE, S, epsilon, gamma, upper_reward, non_zero_transition, action_prob, A_starting_value, A_finishing_value, A_increment);
	write_meta_data_to_dat_file_number_of_actions_iterations(stringstream_VIAEH, S, epsilon, gamma, upper_reward, non_zero_transition, action_prob, A_starting_value, A_finishing_value, A_increment);

	for (int A_num = A_starting_value; A_num <= A_finishing_value; A_num = A_num + A_increment)
	{

		printf("Beginning iteration A_num = %d\n", A_num);

		// GENERATE THE MDP
		int seed = time(0);
		printf("\nseed: %d\n", seed);
		auto MDP = generate_random_MDP_with_variable_parameters(S, A_num, action_prob, non_zero_transition, upper_reward, seed);
		R_type R = get<0>(MDP);
		A_type A = get<1>(MDP);
		P_type P = get<2>(MDP);

		// iterations test printing
		float R_max = find_max_R(R);
		float R_min = find_min_R(R);
		float iterations_bound = log(R_max / ((1.0 - gamma) * epsilon)) / log(1.0 / gamma);
		printf("R_max is %f\n", R_max);
		printf("R_min is %f\n", R_min);
		printf("upper bound is %f\n", R_max / (1.0 - gamma));
		printf("lower bound is %f\n", R_min / (1.0 - gamma));
		printf("iterations bound: %f\n", iterations_bound);

		// VI testing
		// TODO: make method that takes arguments and performs the writing of data, takes filename, MDP etc. as argument
		A_type A1 = copy_A(A);
		auto start_VI = high_resolution_clock::now();

		V_type V_approx_solution_tuple = value_iteration(S, R, A1, P, gamma, epsilon);
		vector<double> V_approx_solution = get<0>(V_approx_solution_tuple);
		int V_approx_solution_iterations = get<1>(V_approx_solution_tuple);

		printf("VI iterations: %d\n", V_approx_solution_iterations);

		auto stop_VI = high_resolution_clock::now();
		auto duration_VI = duration_cast<microseconds>(stop_VI - start_VI);

		stringstream_VI << to_string(A_num) << " " << V_approx_solution_iterations << endl;

		// VIU testing
		A_type A6 = copy_A(A);
		auto start_VIU = high_resolution_clock::now();

		V_type V_approx_solution_upper_tuple = value_iteration_upper(S, R, A6, P, gamma, epsilon);
		vector<double> V_approx_solution_upper = get<0>(V_approx_solution_upper_tuple);
		int V_approx_solution_upper_iterations = get<1>(V_approx_solution_upper_tuple);

		printf("VIU iterations: %d\n", V_approx_solution_upper_iterations);

		auto stop_VIU = high_resolution_clock::now();
		auto duration_VIU = duration_cast<microseconds>(stop_VIU - start_VIU);

		stringstream_VIU << to_string(A_num) << " " << V_approx_solution_upper_iterations << endl;

		// VIH testing
		A_type A2 = copy_A(A);
		auto start_VIH = high_resolution_clock::now();

		V_type V_heap_approx_tuple = value_iteration_with_heap(S, R, A2, P, gamma, epsilon);
		vector<double> V_heap_approx = get<0>(V_heap_approx_tuple);
		int V_heap_approx_iterations = get<1>(V_heap_approx_tuple);

		printf("VIH iterations: %d\n", V_heap_approx_iterations);

		auto stop_VIH = high_resolution_clock::now();
		auto duration_VIH = duration_cast<microseconds>(stop_VIH - start_VIH);

		stringstream_VIH << to_string(A_num) << " " << V_heap_approx_iterations << endl;

		// BVI
		A_type A3 = copy_A(A);
		auto start_BVI = high_resolution_clock::now();

		V_type V_bounded_approx_solution_tuple = bounded_value_iteration(S, R, A3, P, gamma, epsilon);
		vector<double> V_bounded_approx_solution = get<0>(V_bounded_approx_solution_tuple);
		int V_bounded_approx_solution_iterations = get<1>(V_bounded_approx_solution_tuple);

		printf("BVI iterations: %d\n", V_bounded_approx_solution_iterations);

		auto stop_BVI = high_resolution_clock::now();
		auto duration_BVI = duration_cast<microseconds>(stop_BVI - start_BVI);

		stringstream_BVI << to_string(A_num) << " " << V_bounded_approx_solution_iterations << endl;

		// VIAE
		A_type A4 = copy_A(A);
		auto start_VIAE = high_resolution_clock::now();

		V_type V_AE_approx_solution_tuple = value_iteration_action_elimination(S, R, A4, P, gamma, epsilon);
		vector<double> V_AE_approx_solution = get<0>(V_AE_approx_solution_tuple);
		int V_AE_approx_solution_iterations = get<1>(V_AE_approx_solution_tuple);

		printf("VIAE iterations: %d\n", V_AE_approx_solution_iterations);

		auto stop_VIAE = high_resolution_clock::now();
		auto duration_VIAE = duration_cast<microseconds>(stop_VIAE - start_VIAE);

		stringstream_VIAE << to_string(A_num) << " " << V_AE_approx_solution_iterations << endl;

		// VIAEH
		A_type A5 = copy_A(A);
		auto start_VIAEH = high_resolution_clock::now();

		V_type V_AE_H_approx_solution_tuple = value_iteration_action_elimination_heaps(S, R, A5, P, gamma, epsilon);
		vector<double> V_AE_H_approx_solution = get<0>(V_AE_H_approx_solution_tuple);
		int V_AE_H_approx_solution_iterations = get<1>(V_AE_H_approx_solution_tuple);

		printf("VIAEH iterations: %d\n", V_AE_H_approx_solution_iterations);

		auto stop_VIAEH = high_resolution_clock::now();
		auto duration_VIAEH = duration_cast<microseconds>(stop_VIAEH - start_VIAEH);

		stringstream_VIAEH << to_string(A_num) << " " << V_AE_H_approx_solution_iterations << endl;

		// They should in theory all be epsilon from true value, and therefore, at most be 2 * epsilon from each other
		if (abs_max_diff_vectors(V_approx_solution, V_heap_approx) > (2 * epsilon))
		{
			printf("DIFFERENCE\n");
		}
		if (abs_max_diff_vectors(V_AE_approx_solution, V_AE_H_approx_solution) > (2 * epsilon))
		{
			printf("DIFFERENCE\n");
		}
		if (abs_max_diff_vectors(V_bounded_approx_solution, V_approx_solution_upper) > (2 * epsilon))
		{
			printf("DIFFERENCE\n");
		}
	}

	// WRITE ALL DATA TO THEIR RESPECTVIE FILES
	write_stringstream_to_file(stringstream_VI, output_stream_VI, file_name_VI);
	write_stringstream_to_file(stringstream_VIU, output_stream_VIU, file_name_VIU);
	write_stringstream_to_file(stringstream_BVI, output_stream_BVI, file_name_BVI);
	write_stringstream_to_file(stringstream_VIH, output_stream_VIH, file_name_VIH);
	write_stringstream_to_file(stringstream_VIAE, output_stream_VIAE, file_name_VIAE);
	write_stringstream_to_file(stringstream_VIAEH, output_stream_VIAEH, file_name_VIAEH);
}

// VARYING THE REWARD SPACE
void write_meta_data_to_dat_file_max_reward(ostringstream &string_stream, int S, int A_num, double epsilon, double gamma, double non_zero_transition, double action_prob, double max_reward_starting_value, double max_reward_finishing_value, double max_reward_increment)
{
	time_t time_now = time(0);
	string_stream << "# META DATA" << endl;
	string_stream << "# " << endl;
	string_stream << "# "
				  << "experiment run at: " << ctime(&time_now);
	string_stream << "# " << endl;
	string_stream << "# "
				  << "gamma = " << gamma << endl;
	string_stream << "# "
				  << "epsilon = " << epsilon << endl;
	string_stream << "# "
				  << "S = " << S << endl;
	string_stream << "# "
				  << "A = " << A_num << endl;
	string_stream << "# "
				  << "non_zero_transition = " << non_zero_transition << endl;
	string_stream << "# "
				  << "action_prob = " << action_prob << endl;
	string_stream << "# " << endl;
	string_stream << "# VARIABLE IN THIS EXPERIMENT" << endl;
	string_stream << "# "
				  << "max_reward is varied from " << max_reward_starting_value << " to " << max_reward_finishing_value << " with " << max_reward_increment << " increment" << endl;
	string_stream << "# " << endl;
	string_stream << "# ACTUAL DATA" << endl;
	string_stream << "# action_prob | microseconds" << endl;
}

void create_data_tables_max_reward(string filename, int S, int A_num, double epsilon, double gamma, double action_prob, double max_reward_finishing_value, double non_zero_transition)
{

	// the stringstreams to create the test for the files
	ostringstream stringstream_BVI;
	ostringstream stringstream_VI;
	ostringstream stringstream_VIU;
	ostringstream stringstream_VIH;
	ostringstream stringstream_VIAE;
	ostringstream stringstream_VIAEH;

	// the file output objects
	ofstream output_stream_BVI;
	ofstream output_stream_VI;
	ofstream output_stream_VIU;
	ofstream output_stream_VIH;
	ofstream output_stream_VIAE;
	ofstream output_stream_VIAEH;

	// set the name of the file to write to
	string file_name_BVI = "data_tables/max_reward/" + filename + "_BVI.dat";
	string file_name_VI = "data_tables/max_reward/" + filename + "_VI.dat";
	string file_name_VIU = "data_tables/max_reward/" + filename + "_VIU.dat";
	string file_name_VIH = "data_tables/max_reward/" + filename + "_VIH.dat";
	string file_name_VIAE = "data_tables/max_reward/" + filename + "_VIAE.dat";
	string file_name_VIAEH = "data_tables/max_reward/" + filename + "_VIAEH.dat";

	double max_reward_starting_value = 100.0;
	// double max_reward_finishing_value = max_reward_finishing_value;
	double max_reward_increment = 100.0;

	// write meta data to all stringstreams as first in their respective files
	write_meta_data_to_dat_file_max_reward(stringstream_VI, S, A_num, epsilon, gamma, non_zero_transition, action_prob, max_reward_starting_value, max_reward_finishing_value, max_reward_increment);
	write_meta_data_to_dat_file_max_reward(stringstream_VIU, S, A_num, epsilon, gamma, non_zero_transition, action_prob, max_reward_starting_value, max_reward_finishing_value, max_reward_increment);
	write_meta_data_to_dat_file_max_reward(stringstream_VIH, S, A_num, epsilon, gamma, non_zero_transition, action_prob, max_reward_starting_value, max_reward_finishing_value, max_reward_increment);
	write_meta_data_to_dat_file_max_reward(stringstream_BVI, S, A_num, epsilon, gamma, non_zero_transition, action_prob, max_reward_starting_value, max_reward_finishing_value, max_reward_increment);
	write_meta_data_to_dat_file_max_reward(stringstream_VIAE, S, A_num, epsilon, gamma, non_zero_transition, action_prob, max_reward_starting_value, max_reward_finishing_value, max_reward_increment);
	write_meta_data_to_dat_file_max_reward(stringstream_VIAEH, S, A_num, epsilon, gamma, non_zero_transition, action_prob, max_reward_starting_value, max_reward_finishing_value, max_reward_increment);

	for (double max_reward = max_reward_starting_value; max_reward <= max_reward_finishing_value; max_reward = max_reward + max_reward_increment)
	{

		// status message of experiment
		printf("Beginning iteration max_reward = %f\n", max_reward);

		// GENERATE THE MDP FROM CURRENT TIME SEED
		int seed = time(0);
		auto MDP = generate_random_MDP_with_variable_parameters(S, A_num, action_prob, non_zero_transition, max_reward, seed);
		R_type R = get<0>(MDP);
		A_type A = get<1>(MDP);
		P_type P = get<2>(MDP);

		// iterations test printing
		float R_max = find_max_R(R);
		float R_min = find_min_R(R);
		float iterations_bound = log(R_max / ((1.0 - gamma) * epsilon)) / log(1.0 / gamma);
		printf("R_max is %f\n", R_max);
		printf("R_min is %f\n", R_min);
		printf("upper bound is %f\n", R_max / (1.0 - gamma));
		printf("lower bound is %f\n", R_min / (1.0 - gamma));
		printf("iterations bound: %f\n", iterations_bound);

		// VI testing
		A_type A1 = copy_A(A);
		auto start_VI = high_resolution_clock::now();

		V_type V_approx_solution_tuple = value_iteration(S, R, A1, P, gamma, epsilon);
		vector<double> V_approx_solution = get<0>(V_approx_solution_tuple);

		auto stop_VI = high_resolution_clock::now();
		auto duration_VI = duration_cast<microseconds>(stop_VI - start_VI);

		stringstream_VI << to_string(max_reward) << " " << duration_VI.count() << endl;

		// VIU testing
		A_type A6 = copy_A(A);
		auto start_VIU = high_resolution_clock::now();

		V_type V_approx_solution_upper_tuple = value_iteration_upper(S, R, A6, P, gamma, epsilon);
		vector<double> V_approx_solution_upper = get<0>(V_approx_solution_upper_tuple);

		auto stop_VIU = high_resolution_clock::now();
		auto duration_VIU = duration_cast<microseconds>(stop_VIU - start_VIU);

		stringstream_VIU << to_string(max_reward) << " " << duration_VIU.count() << endl;

		// VIH testing
		A_type A2 = copy_A(A);
		auto start_VIH = high_resolution_clock::now();

		V_type V_heap_approx_tuple = value_iteration_with_heap(S, R, A2, P, gamma, epsilon);
		vector<double> V_heap_approx = get<0>(V_heap_approx_tuple);

		auto stop_VIH = high_resolution_clock::now();
		auto duration_VIH = duration_cast<microseconds>(stop_VIH - start_VIH);

		stringstream_VIH << to_string(max_reward) << " " << duration_VIH.count() << endl;

		// BVI
		A_type A3 = copy_A(A);
		auto start_BVI = high_resolution_clock::now();

		V_type V_bounded_approx_solution_tuple = bounded_value_iteration(S, R, A3, P, gamma, epsilon);
		vector<double> V_bounded_approx_solution = get<0>(V_bounded_approx_solution_tuple);

		auto stop_BVI = high_resolution_clock::now();
		auto duration_BVI = duration_cast<microseconds>(stop_BVI - start_BVI);

		stringstream_BVI << to_string(max_reward) << " " << duration_BVI.count() << endl;

		// VIAE
		A_type A4 = copy_A(A);
		auto start_VIAE = high_resolution_clock::now();

		V_type V_AE_approx_solution_tuple = value_iteration_action_elimination(S, R, A4, P, gamma, epsilon);
		vector<double> V_AE_approx_solution = get<0>(V_AE_approx_solution_tuple);

		auto stop_VIAE = high_resolution_clock::now();
		auto duration_VIAE = duration_cast<microseconds>(stop_VIAE - start_VIAE);

		stringstream_VIAE << to_string(max_reward) << " " << duration_VIAE.count() << endl;

		// VIAEH
		A_type A5 = copy_A(A);
		auto start_VIAEH = high_resolution_clock::now();

		V_type V_AE_H_approx_solution_tuple = value_iteration_action_elimination_heaps(S, R, A5, P, gamma, epsilon);
		vector<double> V_AE_H_approx_solution = get<0>(V_AE_H_approx_solution_tuple);

		auto stop_VIAEH = high_resolution_clock::now();
		auto duration_VIAEH = duration_cast<microseconds>(stop_VIAEH - start_VIAEH);

		stringstream_VIAEH << to_string(max_reward) << " " << duration_VIAEH.count() << endl;

		// They should in theory all be epsilon from true value, and therefore, at most be 2 * epsilon from each other
		if (abs_max_diff_vectors(V_approx_solution, V_heap_approx) > (2 * epsilon))
		{
			printf("DIFFERENCE\n");
		}
		if (abs_max_diff_vectors(V_AE_approx_solution, V_AE_H_approx_solution) > (2 * epsilon))
		{
			printf("DIFFERENCE\n");
		}
		if (abs_max_diff_vectors(V_bounded_approx_solution, V_approx_solution_upper) > (2 * epsilon))
		{
			printf("DIFFERENCE\n");
		}
	}

	// WRITE ALL DATA TO THEIR RESPECTVIE FILES
	write_stringstream_to_file(stringstream_VI, output_stream_VI, file_name_VI);
	write_stringstream_to_file(stringstream_VIU, output_stream_VIU, file_name_VIU);
	write_stringstream_to_file(stringstream_BVI, output_stream_BVI, file_name_BVI);
	write_stringstream_to_file(stringstream_VIH, output_stream_VIH, file_name_VIH);
	write_stringstream_to_file(stringstream_VIAE, output_stream_VIAE, file_name_VIAE);
	write_stringstream_to_file(stringstream_VIAEH, output_stream_VIAEH, file_name_VIAEH);
}

// VARYING NUMBER OF ACTIONS - Iterations until convergence plot
void write_meta_data_to_dat_file_number_of_actions_first_convergence_iteration(ostringstream &string_stream, int S, double epsilon, double gamma, double upper_reward, double non_zero_transition, double action_prob, int A_starting_value, int A_finishing_value, int A_increment)
{
	time_t time_now = time(0);
	string_stream << "# META DATA" << endl;
	string_stream << "# " << endl;
	string_stream << "# "
				  << "experiment run at: " << ctime(&time_now);
	string_stream << "# " << endl;
	string_stream << "# "
				  << "gamma = " << gamma << endl;
	string_stream << "# "
				  << "epsilon = " << epsilon << endl;
	string_stream << "# "
				  << "S = " << S << endl;
	string_stream << "# "
				  << "non_zero_transition = " << non_zero_transition << endl;
	string_stream << "# "
				  << "upper_reward = " << upper_reward << endl;
	string_stream << "# "
				  << "action_prob = " << action_prob << endl;
	string_stream << "# " << endl;
	string_stream << "# VARIABLE IN THIS EXPERIMENT" << endl;
	string_stream << "# "
				  << "Number of actions, A, is varied from " << A_starting_value << " to " << A_finishing_value << " with " << A_increment << " increment" << endl;
	string_stream << "# " << endl;
	string_stream << "# ACTUAL DATA" << endl;
	string_stream << "# number of actions | iterations" << endl;
}

void create_data_tables_number_of_actions_first_convergence_iteration(string filename, int S, int A_max, double epsilon, double gamma, double upper_reward, double non_zero_transition)
{

	// the stringstreams to create the test for the files
	ostringstream stringstream_BVI;
	ostringstream stringstream_BVI_max;
	ostringstream stringstream_VI;
	ostringstream stringstream_VI_max;
	ostringstream stringstream_VIU;
	ostringstream stringstream_theoretical;

	// the file output objects
	ofstream output_stream_BVI;
	ofstream output_stream_BVI_max;
	ofstream output_stream_VI;
	ofstream output_stream_VI_max;
	ofstream output_stream_VIU;
	ofstream output_stream_theoretical;

	// set the name of the file to write to
	string file_name_BVI = "data_tables/number_of_actions_first_convergence_iteration/" + filename + "_BVI.dat";
	string file_name_BVI_max = "data_tables/number_of_actions_first_convergence_iteration/" + filename + "_BVI_max.dat";
	string file_name_VI = "data_tables/number_of_actions_first_convergence_iteration/" + filename + "_VI.dat";
	string file_name_VI_max = "data_tables/number_of_actions_first_convergence_iteration/" + filename + "_VI_max.dat";
	string file_name_VIU = "data_tables/number_of_actions_first_convergence_iteration/" + filename + "_VIU.dat";
	string file_name_theoretical = "data_tables/number_of_actions_first_convergence_iteration/" + filename + "_theoretical.dat";

	// The varying parameters
	int A_starting_value = 50;
	int A_finishing_value = A_max;
	int A_increment = 50;

	// hardcoded parameter
	double action_prob = 1.0;

	// write meta data to all stringstreams as first in their respective files
	write_meta_data_to_dat_file_number_of_actions_first_convergence_iteration(stringstream_VI, S, epsilon, gamma, upper_reward, non_zero_transition, action_prob, A_starting_value, A_finishing_value, A_increment);
	write_meta_data_to_dat_file_number_of_actions_first_convergence_iteration(stringstream_VI_max, S, epsilon, gamma, upper_reward, non_zero_transition, action_prob, A_starting_value, A_finishing_value, A_increment);
	write_meta_data_to_dat_file_number_of_actions_first_convergence_iteration(stringstream_VIU, S, epsilon, gamma, upper_reward, non_zero_transition, action_prob, A_starting_value, A_finishing_value, A_increment);
	write_meta_data_to_dat_file_number_of_actions_first_convergence_iteration(stringstream_BVI, S, epsilon, gamma, upper_reward, non_zero_transition, action_prob, A_starting_value, A_finishing_value, A_increment);
	write_meta_data_to_dat_file_number_of_actions_first_convergence_iteration(stringstream_BVI_max, S, epsilon, gamma, upper_reward, non_zero_transition, action_prob, A_starting_value, A_finishing_value, A_increment);
	write_meta_data_to_dat_file_number_of_actions_first_convergence_iteration(stringstream_theoretical, S, epsilon, gamma, upper_reward, non_zero_transition, action_prob, A_starting_value, A_finishing_value, A_increment);

	for (int A_num = A_starting_value; A_num <= A_finishing_value; A_num = A_num + A_increment)
	{

		printf("Beginning iteration A_num = %d\n", A_num);

		// GENERATE THE MDP
		int seed = time(0);
		auto MDP = generate_random_MDP_with_variable_parameters(S, A_num, 1.0, non_zero_transition, upper_reward, seed);
		R_type R = get<0>(MDP);
		A_type A = get<1>(MDP);
		P_type P = get<2>(MDP);

		float R_max = find_max_R(R);

		// theoretical bound from single instance
		float theoretical_single_instance_bound = log(R_max / ((1.0 - gamma) * epsilon)) / log(1.0 / gamma);

		// the convergence criterias
		vector<int> V_convergence_bounds = first_convergence_iteration(S, R, A, P, gamma, epsilon);

		stringstream_BVI << to_string(A_num) << " " << V_convergence_bounds[0] << endl;
		stringstream_VIU << to_string(A_num) << " " << V_convergence_bounds[1] << endl;
		stringstream_VI << to_string(A_num) << " " << V_convergence_bounds[2] << endl;
		stringstream_BVI_max << to_string(A_num) << " " << V_convergence_bounds[3] << endl;
		stringstream_theoretical << to_string(A_num) << " " << theoretical_single_instance_bound << endl;
	}

	// WRITE ALL DATA TO THEIR RESPECTVIE FILES
	write_stringstream_to_file(stringstream_VI, output_stream_VI, file_name_VI);
	write_stringstream_to_file(stringstream_VIU, output_stream_VIU, file_name_VIU);
	write_stringstream_to_file(stringstream_BVI, output_stream_BVI, file_name_BVI);
	write_stringstream_to_file(stringstream_BVI_max, output_stream_BVI_max, file_name_BVI_max);
	write_stringstream_to_file(stringstream_theoretical, output_stream_BVI, file_name_theoretical);
}

// WORK PER ITERATION
void write_meta_data_to_dat_file_work_per_iteration(ostringstream &string_stream, int S, int A_num, double epsilon, double gamma, int number_of_non_zero_transition, double action_prob)
{
	time_t time_now = time(0);
	string_stream << "# META DATA" << endl;
	string_stream << "# " << endl;
	string_stream << "# "
				  << "experiment run at: " << ctime(&time_now);
	string_stream << "# " << endl;
	string_stream << "# "
				  << "gamma = " << gamma << endl;
	string_stream << "# "
				  << "epsilon = " << epsilon << endl;
	string_stream << "# "
				  << "S = " << S << endl;
	string_stream << "# "
				  << "A = " << A_num << endl;
	string_stream << "# "
				  << "number_of_non_zero_transition = " << number_of_non_zero_transition << endl;
	string_stream << "# "
				  << "action_prob = " << action_prob << endl;
	string_stream << "# " << endl;
	string_stream << "# ACTUAL DATA" << endl;
	string_stream << "# iteration number | microseconds" << endl;
}

void create_data_tables_work_per_iteration(string filename, int S, int A_num, double epsilon, double gamma, double action_prob, int non_zero_transition, double mean, double variance)
{

	// FOR WORK PER ITERATION
	// the stringstreams to create the test for the files
	ostringstream stringstream_BVI;
	ostringstream stringstream_VI;
	ostringstream stringstream_VIU;
	ostringstream stringstream_VIH;
	ostringstream stringstream_VIAE;
	ostringstream stringstream_VIAEH;
	ostringstream stringstream_VIAEHL;
	ostringstream stringstream_BAO;

	// the file output objects
	ofstream output_stream_BVI;
	ofstream output_stream_VI;
	ofstream output_stream_VIU;
	ofstream output_stream_VIH;
	ofstream output_stream_VIAE;
	ofstream output_stream_VIAEH;
	ofstream output_stream_VIAEHL;
	ofstream output_stream_BAO;

	// set the name of the file to write to
	string file_name_BVI = "data_tables/work_per_iteration/" + filename + "_BVI.dat";
	string file_name_VI = "data_tables/work_per_iteration/" + filename + "_VI.dat";
	string file_name_VIU = "data_tables/work_per_iteration/" + filename + "_VIU.dat";
	string file_name_VIH = "data_tables/work_per_iteration/" + filename + "_VIH.dat";
	string file_name_VIAE = "data_tables/work_per_iteration/" + filename + "_VIAE.dat";
	string file_name_VIAEH = "data_tables/work_per_iteration/" + filename + "_VIAEH.dat";
	string file_name_VIAEHL = "data_tables/work_per_iteration/" + filename + "_VIAEHL.dat";
	string file_name_BAO = "data_tables/work_per_iteration/" + filename + "_BAO.dat";

	// write meta data to all stringstreams as first in their respective files
	write_meta_data_to_dat_file_work_per_iteration(stringstream_BVI, S, A_num, epsilon, gamma, non_zero_transition, action_prob);
	write_meta_data_to_dat_file_work_per_iteration(stringstream_VI, S, A_num, epsilon, gamma, non_zero_transition, action_prob);
	write_meta_data_to_dat_file_work_per_iteration(stringstream_VIU, S, A_num, epsilon, gamma, non_zero_transition, action_prob);
	write_meta_data_to_dat_file_work_per_iteration(stringstream_VIH, S, A_num, epsilon, gamma, non_zero_transition, action_prob);
	write_meta_data_to_dat_file_work_per_iteration(stringstream_VIAE, S, A_num, epsilon, gamma, non_zero_transition, action_prob);
	write_meta_data_to_dat_file_work_per_iteration(stringstream_VIAEH, S, A_num, epsilon, gamma, non_zero_transition, action_prob);
	write_meta_data_to_dat_file_work_per_iteration(stringstream_VIAEHL, S, A_num, epsilon, gamma, non_zero_transition, action_prob);
	write_meta_data_to_dat_file_work_per_iteration(stringstream_BAO, S, A_num, epsilon, gamma, non_zero_transition, action_prob);

	// FOR ACCUMULATED WORK
	// the stringstreams to create the test for the files
	ostringstream stringstream_accum_BVI;
	ostringstream stringstream_accum_VI;
	ostringstream stringstream_accum_VIU;
	ostringstream stringstream_accum_VIH;
	ostringstream stringstream_accum_VIAE;
	ostringstream stringstream_accum_VIAEH;
	ostringstream stringstream_accum_VIAEHL;
	ostringstream stringstream_accum_BAO;

	// the file output objects
	ofstream output_stream_accum_BVI;
	ofstream output_stream_accum_VI;
	ofstream output_stream_accum_VIU;
	ofstream output_stream_accum_VIH;
	ofstream output_stream_accum_VIAE;
	ofstream output_stream_accum_VIAEH;
	ofstream output_stream_accum_VIAEHL;
	ofstream output_stream_accum_BAO;

	// set the name of the file to write to
	string file_name_accum_BVI = "data_tables/work_per_iteration_accum/" + filename + "_accum_BVI.dat";
	string file_name_accum_VI = "data_tables/work_per_iteration_accum/" + filename + "_accum_VI.dat";
	string file_name_accum_VIU = "data_tables/work_per_iteration_accum/" + filename + "_accum_VIU.dat";
	string file_name_accum_VIH = "data_tables/work_per_iteration_accum/" + filename + "_accum_VIH.dat";
	string file_name_accum_VIAE = "data_tables/work_per_iteration_accum/" + filename + "_accum_VIAE.dat";
	string file_name_accum_VIAEH = "data_tables/work_per_iteration_accum/" + filename + "_accum_VIAEH.dat";
	string file_name_accum_VIAEHL = "data_tables/work_per_iteration_accum/" + filename + "_accum_VIAEHL.dat";
	string file_name_accum_BAO = "data_tables/work_per_iteration_accum/" + filename + "_accum_BAO.dat";

	// write meta data to all stringstreams as first in their respective files
	write_meta_data_to_dat_file_work_per_iteration(stringstream_accum_VI, S, A_num, epsilon, gamma, non_zero_transition, action_prob);
	write_meta_data_to_dat_file_work_per_iteration(stringstream_accum_VIU, S, A_num, epsilon, gamma, non_zero_transition, action_prob);
	write_meta_data_to_dat_file_work_per_iteration(stringstream_accum_VIH, S, A_num, epsilon, gamma, non_zero_transition, action_prob);
	write_meta_data_to_dat_file_work_per_iteration(stringstream_accum_VIAE, S, A_num, epsilon, gamma, non_zero_transition, action_prob);
	write_meta_data_to_dat_file_work_per_iteration(stringstream_accum_VIAEH, S, A_num, epsilon, gamma, non_zero_transition, action_prob);
	write_meta_data_to_dat_file_work_per_iteration(stringstream_accum_VIAEHL, S, A_num, epsilon, gamma, non_zero_transition, action_prob);
	write_meta_data_to_dat_file_work_per_iteration(stringstream_accum_BAO, S, A_num, epsilon, gamma, non_zero_transition, action_prob);
	write_meta_data_to_dat_file_work_per_iteration(stringstream_accum_BVI, S, A_num, epsilon, gamma, non_zero_transition, action_prob);

	// BEGIN EXPERIMENTATION
	// GENERATE THE MDP FROM CURRENT TIME SEED
	int seed = time(0);
	printf("seed: %d\n", seed);

	// TODO permament change to normal distribution here?
	// auto MDP = generate_random_MDP_normal_distributed_rewards(S, A_num, action_prob, non_zero_transition, seed, mean, variance);
	auto MDP = generate_random_MDP_exponential_distributed_rewards(S, A_num, action_prob, non_zero_transition, 0.02, seed);
	// auto MDP = generate_random_MDP_with_variable_parameters(S, A_num, action_prob, non_zero_transition, upper_reward, seed);
	R_type R = get<0>(MDP);
	A_type A = get<1>(MDP);
	P_type P = get<2>(MDP);

	// VIAEH
	printf("VIAEH\n");
	A_type A5 = copy_A(A);

	V_type V_AE_H_approx_solution_tuple = value_iteration_action_elimination_heaps(S, R, A5, P, gamma, epsilon);
	vector<double> V_AE_H_approx_solution = get<0>(V_AE_H_approx_solution_tuple);
	int V_AE_H_approx_solution_iterations = get<1>(V_AE_H_approx_solution_tuple);
	vector<microseconds> V_AE_H_approx_solution_work_per_iteration = get<2>(V_AE_H_approx_solution_tuple);

	auto tick_accumulator_VIAEH = V_AE_H_approx_solution_work_per_iteration[0].count();

	for (int iteration = 1; iteration <= V_AE_H_approx_solution_iterations; iteration++)
	{
		auto iteration_work = V_AE_H_approx_solution_work_per_iteration[iteration].count();
		tick_accumulator_VIAEH = tick_accumulator_VIAEH + iteration_work;

		stringstream_VIAEH << to_string(iteration) << " " << iteration_work << endl;
		stringstream_accum_VIAEH << to_string(iteration) << " " << tick_accumulator_VIAEH << endl;
	}

	// VI testing
	printf("VI\n");
	A_type A1 = copy_A(A);

	V_type V_approx_solution_tuple = value_iteration(S, R, A1, P, gamma, epsilon);
	vector<double> V_approx_solution = get<0>(V_approx_solution_tuple);
	int V_approx_solution_iterations = get<1>(V_approx_solution_tuple);
	vector<microseconds> V_approx_solution_work_per_iteration = get<2>(V_approx_solution_tuple);

	// first entry is zero of the tick type (check this)
	auto tick_accumulator_VI = V_approx_solution_work_per_iteration[0].count();

	for (int iteration = 1; iteration <= V_approx_solution_iterations; iteration++)
	{
		auto iteration_work = V_approx_solution_work_per_iteration[iteration].count();
		tick_accumulator_VI = tick_accumulator_VI + iteration_work;

		stringstream_VI << to_string(iteration) << " " << iteration_work << endl;
		stringstream_accum_VI << to_string(iteration) << " " << tick_accumulator_VI << endl;
	}

	// VIU testing
	printf("VIU\n");
	A_type A6 = copy_A(A);

	V_type V_approx_solution_upper_tuple = value_iteration_upper(S, R, A6, P, gamma, epsilon);
	vector<double> V_approx_solution_upper = get<0>(V_approx_solution_upper_tuple);
	int V_approx_solution_upper_iterations = get<1>(V_approx_solution_upper_tuple);
	vector<microseconds> V_approx_solution_upper_work_per_iteration = get<2>(V_approx_solution_upper_tuple);

	auto tick_accumulator_VIU = V_approx_solution_upper_work_per_iteration[0].count();

	for (int iteration = 1; iteration <= V_approx_solution_upper_iterations; iteration++)
	{
		auto iteration_work = V_approx_solution_upper_work_per_iteration[iteration].count();
		tick_accumulator_VIU = tick_accumulator_VIU + iteration_work;

		stringstream_VIU << to_string(iteration) << " " << iteration_work << endl;
		stringstream_accum_VIU << to_string(iteration) << " " << tick_accumulator_VIU << endl;
	}

	// VIH testing
	printf("VIH\n");
	A_type A2 = copy_A(A);

	V_type V_heap_approx_tuple = value_iteration_with_heap(S, R, A2, P, gamma, epsilon);
	vector<double> V_heap_approx = get<0>(V_heap_approx_tuple);
	int V_heap_approx_iterations = get<1>(V_heap_approx_tuple);
	vector<microseconds> V_heap_approx_work_per_iteration = get<2>(V_heap_approx_tuple);

	auto tick_accumulator_VIH = V_heap_approx_work_per_iteration[0].count();

	for (int iteration = 1; iteration <= V_heap_approx_iterations; iteration++)
	{
		auto iteration_work = V_heap_approx_work_per_iteration[iteration].count();
		tick_accumulator_VIH = tick_accumulator_VIH + iteration_work;

		stringstream_VIH << to_string(iteration) << " " << iteration_work << endl;
		stringstream_accum_VIH << to_string(iteration) << " " << tick_accumulator_VIH << endl;
	}

	// BVI
	printf("BVI\n");
	A_type A3 = copy_A(A);

	V_type V_bounded_approx_solution_tuple = bounded_value_iteration(S, R, A3, P, gamma, epsilon);
	vector<double> V_bounded_approx_solution = get<0>(V_bounded_approx_solution_tuple);
	int V_bounded_approx_solution_iterations = get<1>(V_bounded_approx_solution_tuple);
	vector<microseconds> V_bounded_approx_solution_work_per_iteration = get<2>(V_bounded_approx_solution_tuple);

	auto tick_accumulator_BVI = V_bounded_approx_solution_work_per_iteration[0].count();

	for (int iteration = 1; iteration <= V_bounded_approx_solution_iterations; iteration++)
	{
		auto iteration_work = V_bounded_approx_solution_work_per_iteration[iteration].count();
		tick_accumulator_BVI = tick_accumulator_BVI + iteration_work;

		stringstream_BVI << to_string(iteration) << " " << iteration_work << endl;
		stringstream_accum_BVI << to_string(iteration) << " " << tick_accumulator_BVI << endl;
	}

	// VIAE
	printf("VIAE\n");
	A_type A4 = copy_A(A);

	V_type V_AE_approx_solution_tuple = value_iteration_action_elimination(S, R, A4, P, gamma, epsilon);
	vector<double> V_AE_approx_solution = get<0>(V_AE_approx_solution_tuple);
	int V_AE_approx_solution_iterations = get<1>(V_AE_approx_solution_tuple);
	vector<microseconds> V_AE_approx_solution_work_per_iteration = get<2>(V_AE_approx_solution_tuple);

	auto tick_accumulator_VIAE = V_AE_approx_solution_work_per_iteration[0].count();

	for (int iteration = 1; iteration <= V_AE_approx_solution_iterations; iteration++)
	{
		auto iteration_work = V_AE_approx_solution_work_per_iteration[iteration].count();
		tick_accumulator_VIAE = tick_accumulator_VIAE + iteration_work;

		stringstream_VIAE << to_string(iteration) << " " << iteration_work << endl;
		stringstream_accum_VIAE << to_string(iteration) << " " << tick_accumulator_VIAE << endl;
	}

	// VIAEHL
	printf("VIAEHL\n");
	A_type A7 = copy_A(A);

	V_type VIAEHL_approx_solution_tuple = value_iteration_action_elimination_heaps_lower_bound_approx(S, R, A7, P, gamma, epsilon);
	vector<double> VIAEHL_approx_solution = get<0>(VIAEHL_approx_solution_tuple);
	int VIAEHL_approx_solution_iterations = get<1>(VIAEHL_approx_solution_tuple);
	vector<microseconds> VIAEHL_approx_solution_work_per_iteration = get<2>(VIAEHL_approx_solution_tuple);

	auto tick_accumulator_VIAEHL = VIAEHL_approx_solution_work_per_iteration[0].count();

	for (int iteration = 1; iteration <= VIAEHL_approx_solution_iterations; iteration++)
	{
		auto iteration_work = VIAEHL_approx_solution_work_per_iteration[iteration].count();
		tick_accumulator_VIAEHL = tick_accumulator_VIAEHL + iteration_work;

		stringstream_VIAEHL << to_string(iteration) << " " << iteration_work << endl;
		stringstream_accum_VIAEHL << to_string(iteration) << " " << tick_accumulator_VIAEHL << endl;
	}

	// BAO
	printf("BAO\n");
	A_type A8 = copy_A(A);

	V_type BAO_approx_solution_tuple = value_iteration_BAO(S, R, A8, P, gamma, epsilon);
	vector<double> BAO_approx_solution = get<0>(BAO_approx_solution_tuple);
	int BAO_approx_solution_iterations = get<1>(BAO_approx_solution_tuple);
	vector<microseconds> BAO_approx_solution_work_per_iteration = get<2>(BAO_approx_solution_tuple);

	auto tick_accumulator_BAO = BAO_approx_solution_work_per_iteration[0].count();

	for (int iteration = 1; iteration <= BAO_approx_solution_iterations; iteration++)
	{
		auto iteration_work = BAO_approx_solution_work_per_iteration[iteration].count();
		tick_accumulator_BAO = tick_accumulator_BAO + iteration_work;

		stringstream_BAO << to_string(iteration) << " " << iteration_work << endl;
		stringstream_accum_BAO << to_string(iteration) << " " << tick_accumulator_BAO << endl;
	}

	// They should in theory all be epsilon from true value, and therefore, at most be 2 * epsilon from each other
	if (abs_max_diff_vectors(V_approx_solution, V_heap_approx) > (2 * epsilon))
	{
		printf("DIFFERENCE\n");
	}
	if (abs_max_diff_vectors(V_AE_approx_solution, V_AE_H_approx_solution) > (2 * epsilon))
	{
		printf("DIFFERENCE\n");
	}
	if (abs_max_diff_vectors(V_bounded_approx_solution, V_approx_solution_upper) > (2 * epsilon))
	{
		printf("DIFFERENCE\n");
	}
	if (abs_max_diff_vectors(VIAEHL_approx_solution, V_AE_H_approx_solution) > (2 * epsilon))
	{
		printf("DIFFERENCE\n");
	}
	if (abs_max_diff_vectors(BAO_approx_solution, V_AE_H_approx_solution) > (2 * epsilon))
	{
		printf("DIFFERENCE\n");
	}

	// WRITE ALL DATA TO THEIR RESPECTVIE FILES
	write_stringstream_to_file(stringstream_VI, output_stream_VI, file_name_VI);
	write_stringstream_to_file(stringstream_VIU, output_stream_VIU, file_name_VIU);
	write_stringstream_to_file(stringstream_BVI, output_stream_BVI, file_name_BVI);
	write_stringstream_to_file(stringstream_VIH, output_stream_VIH, file_name_VIH);
	write_stringstream_to_file(stringstream_VIAE, output_stream_VIAE, file_name_VIAE);
	write_stringstream_to_file(stringstream_VIAEH, output_stream_VIAEH, file_name_VIAEH);
	write_stringstream_to_file(stringstream_VIAEHL, output_stream_VIAEHL, file_name_VIAEHL);
	write_stringstream_to_file(stringstream_BAO, output_stream_BAO, file_name_BAO);

	write_stringstream_to_file(stringstream_accum_VI, output_stream_accum_VI, file_name_accum_VI);
	write_stringstream_to_file(stringstream_accum_VIU, output_stream_accum_VIU, file_name_accum_VIU);
	write_stringstream_to_file(stringstream_accum_BVI, output_stream_accum_BVI, file_name_accum_BVI);
	write_stringstream_to_file(stringstream_accum_VIH, output_stream_accum_VIH, file_name_accum_VIH);
	write_stringstream_to_file(stringstream_accum_VIAE, output_stream_accum_VIAE, file_name_accum_VIAE);
	write_stringstream_to_file(stringstream_accum_VIAEH, output_stream_accum_VIAEH, file_name_accum_VIAEH);
	write_stringstream_to_file(stringstream_accum_VIAEHL, output_stream_accum_VIAEHL, file_name_accum_VIAEHL);
	write_stringstream_to_file(stringstream_accum_BAO, output_stream_accum_BAO, file_name_accum_BAO);
}

// TOP ACTION CHANGE EXPERIMENT
void write_meta_data_to_dat_file_top_action_change(ostringstream &string_stream, int S, int A_num, double epsilon, double gamma, int non_zero_transitions, double upper_reward, double action_prob)
{
	time_t time_now = time(0);
	string_stream << "# META DATA" << endl;
	string_stream << "# " << endl;
	string_stream << "# "
				  << "experiment run at: " << ctime(&time_now);
	string_stream << "# " << endl;
	string_stream << "# "
				  << "gamma = " << gamma << endl;
	string_stream << "# "
				  << "epsilon = " << epsilon << endl;
	string_stream << "# "
				  << "upper_reward = " << upper_reward << endl;
	string_stream << "# "
				  << "S = " << S << endl;
	string_stream << "# "
				  << "A = " << A_num << endl;
	string_stream << "# "
				  << "non_zero_transitions = " << non_zero_transitions << endl;
	string_stream << "# "
				  << "action_prob = " << action_prob << endl;
	string_stream << "# " << endl;
	string_stream << "# ACTUAL DATA" << endl;
	string_stream << "# iteration number | top action changes in current iteration" << endl;
}

void create_data_tables_top_action_change(string filename, int S, int A_num, double epsilon, double gamma, double upper_reward, double action_prob, int number_of_transitions, double mean, double variance, double lambda)
{

	// FOR WORK PER ITERATION
	// the stringstreams to create the test for the files
	ostringstream stringstream_uniform;
	ostringstream stringstream_normal;
	ostringstream stringstream_exponential;

	// the file output objects
	ofstream output_stream_uniform;
	ofstream output_stream_normal;
	ofstream output_stream_exponential;

	// set the name of the file to write to
	string file_name_uniform = "data_tables/top_action_change/" + filename + "_uniform.dat";
	string file_name_normal = "data_tables/top_action_change/" + filename + "_normal.dat";
	string file_name_exponential = "data_tables/top_action_change/" + filename + "_exponential.dat";

	// write meta data to all stringstreams as first in their respective files
	write_meta_data_to_dat_file_top_action_change(stringstream_uniform, S, A_num, epsilon, gamma, number_of_transitions, upper_reward, action_prob);
	write_meta_data_to_dat_file_top_action_change(stringstream_normal, S, A_num, epsilon, gamma, number_of_transitions, upper_reward, action_prob);
	write_meta_data_to_dat_file_top_action_change(stringstream_exponential, S, A_num, epsilon, gamma, number_of_transitions, upper_reward, action_prob);

	// UNIFORM REWARDS
	int seed = time(0);
	auto MDP_uniform_reward = generate_random_MDP_with_variable_parameters_fixed_nonzero_trans_states(S, A_num, action_prob, number_of_transitions, upper_reward, seed);
	R_type R_uniform_reward = get<0>(MDP_uniform_reward);
	A_type A_uniform_reward = get<1>(MDP_uniform_reward);
	P_type P_uniform_reward = get<2>(MDP_uniform_reward);

	printf("Beginning: uniform\n");
	A_type A1 = copy_A(A_uniform_reward);

	tuple<int, vector<int>> top_action_change_uniform_reward_tuple = top_action_change(S, R_uniform_reward, A1, P_uniform_reward, gamma, epsilon);
	int total_top_action_changes_uniform_reward = get<0>(top_action_change_uniform_reward_tuple);
	vector<int> top_action_change_per_iteration_uniform_reward = get<1>(top_action_change_uniform_reward_tuple);

	// -1 as there is the 0th element as dummy
	int number_of_iterations_uniform_reward = int(top_action_change_per_iteration_uniform_reward.size()) - 1;

	for (int iteration = 1; iteration <= number_of_iterations_uniform_reward; iteration++)
	{
		stringstream_uniform << to_string(iteration) << " " << (float(top_action_change_per_iteration_uniform_reward[iteration]) / float(S)) << endl;
	}

	// NORMAL DISTRIBUTED REWARDS
	seed = time(0);
	auto MDP_normal_reward = generate_random_MDP_normal_distributed_rewards(S, A_num, action_prob, number_of_transitions, seed, mean, variance);
	R_type R_normal_reward = get<0>(MDP_normal_reward);
	A_type A_normal_reward = get<1>(MDP_normal_reward);
	P_type P_normal_reward = get<2>(MDP_normal_reward);

	printf("Beginning: Normal\n");
	A_type A2 = copy_A(A_normal_reward);

	tuple<int, vector<int>> top_action_change_normal_reward_tuple = top_action_change(S, R_normal_reward, A2, P_normal_reward, gamma, epsilon);
	int total_top_action_changes_normal_reward = get<0>(top_action_change_normal_reward_tuple);
	vector<int> top_action_change_per_iteration_normal_reward = get<1>(top_action_change_normal_reward_tuple);

	// -1 as there is the 0th element as dummy
	int number_of_iterations_normal_reward = int(top_action_change_per_iteration_normal_reward.size()) - 1;

	for (int iteration = 1; iteration <= number_of_iterations_normal_reward; iteration++)
	{
		stringstream_normal << to_string(iteration) << " " << (float(top_action_change_per_iteration_normal_reward[iteration]) / float(S)) << endl;
	}

	// EXPONENTIAL DISTRIBUTED REWARDS
	seed = time(0);
	auto MDP_exponential_reward = generate_random_MDP_exponential_distributed_rewards(S, A_num, action_prob, number_of_transitions, lambda, seed);
	R_type R_exponential_reward = get<0>(MDP_exponential_reward);
	A_type A_exponential_reward = get<1>(MDP_exponential_reward);
	P_type P_exponential_reward = get<2>(MDP_exponential_reward);

	printf("Beginning: Exponential\n");
	A_type A3 = copy_A(A_exponential_reward);

	tuple<int, vector<int>> top_action_change_exponential_reward_tuple = top_action_change(S, R_exponential_reward, A3, P_exponential_reward, gamma, epsilon);
	int total_top_action_changes_exponential_reward = get<0>(top_action_change_exponential_reward_tuple);
	vector<int> top_action_change_per_iteration_exponential_reward = get<1>(top_action_change_exponential_reward_tuple);

	// -1 as there is the 0th element as dummy
	int number_of_iterations_exponential_reward = int(top_action_change_per_iteration_exponential_reward.size()) - 1;

	for (int iteration = 1; iteration <= number_of_iterations_exponential_reward; iteration++)
	{
		stringstream_exponential << to_string(iteration) << " " << (float(top_action_change_per_iteration_exponential_reward[iteration]) / float(S)) << endl;
	}

	// WRITE ALL DATA TO THEIR RESPECTVIE FILES
	write_stringstream_to_file(stringstream_normal, output_stream_normal, file_name_normal);
	write_stringstream_to_file(stringstream_uniform, output_stream_uniform, file_name_uniform);
	write_stringstream_to_file(stringstream_exponential, output_stream_exponential, file_name_exponential);
}

void write_meta_data_to_dat_file_normal_dist_varying_variance(ostringstream &string_stream, int S, int A_num, double epsilon, double gamma, int transition_list_size, double action_prob, double mean, double min_variance, double max_variance, double variance_increment)
{
	time_t time_now = time(0);
	string_stream << "# META DATA" << endl;
	string_stream << "# " << endl;
	string_stream << "# "
				  << "experiment run at: " << ctime(&time_now);
	string_stream << "# " << endl;
	string_stream << "# "
				  << "gamma = " << gamma << endl;
	string_stream << "# "
				  << "epsilon = " << epsilon << endl;
	string_stream << "# "
				  << "S = " << S << endl;
	string_stream << "# "
				  << "A = " << A_num << endl;
	string_stream << "# "
				  << "transition list size in each state = " << transition_list_size << endl;
	string_stream << "# "
				  << "action_prob = " << action_prob << endl;
	string_stream << "# "
				  << "mean = " << mean << endl;
	string_stream << "# " << endl;
	string_stream << "# VARIABLE IN THIS EXPERIMENT" << endl;
	string_stream << "# "
				  << "variance is varied from " << min_variance << " to " << max_variance << " with " << variance_increment << " increment" << endl;
	string_stream << "# " << endl;
	string_stream << "# ACTUAL DATA" << endl;
	string_stream << "# variance | microseconds" << endl;
}

void create_data_tables_normal_dist_varying_variance(string filename, int S, int A_num, double epsilon, double gamma, double action_prob, int transition_list_size, double mean, double min_variance, double max_variance, double variance_increment)
{

	// the stringstreams to create the test for the files
	ostringstream stringstream_BVI;
	ostringstream stringstream_VI;
	ostringstream stringstream_VIU;
	ostringstream stringstream_VIH;
	ostringstream stringstream_VIAE;
	ostringstream stringstream_VIAEH;

	// the file output objects
	ofstream output_stream_BVI;
	ofstream output_stream_VI;
	ofstream output_stream_VIU;
	ofstream output_stream_VIH;
	ofstream output_stream_VIAE;
	ofstream output_stream_VIAEH;

	// set the name of the file to write to
	string file_name_BVI = "data_tables/normal_dist_varying_variance/" + filename + "_BVI.dat";
	string file_name_VI = "data_tables/normal_dist_varying_variance/" + filename + "_VI.dat";
	string file_name_VIU = "data_tables/normal_dist_varying_variance/" + filename + "_VIU.dat";
	string file_name_VIH = "data_tables/normal_dist_varying_variance/" + filename + "_VIH.dat";
	string file_name_VIAE = "data_tables/normal_dist_varying_variance/" + filename + "_VIAE.dat";
	string file_name_VIAEH = "data_tables/normal_dist_varying_variance/" + filename + "_VIAEH.dat";

	// write meta data to all stringstreams as first in their respective files
	write_meta_data_to_dat_file_normal_dist_varying_variance(stringstream_VI, S, A_num, epsilon, gamma, transition_list_size, action_prob, mean, min_variance, max_variance, variance_increment);
	write_meta_data_to_dat_file_normal_dist_varying_variance(stringstream_VIU, S, A_num, epsilon, gamma, transition_list_size, action_prob, mean, min_variance, max_variance, variance_increment);
	write_meta_data_to_dat_file_normal_dist_varying_variance(stringstream_VIH, S, A_num, epsilon, gamma, transition_list_size, action_prob, mean, min_variance, max_variance, variance_increment);
	write_meta_data_to_dat_file_normal_dist_varying_variance(stringstream_BVI, S, A_num, epsilon, gamma, transition_list_size, action_prob, mean, min_variance, max_variance, variance_increment);
	write_meta_data_to_dat_file_normal_dist_varying_variance(stringstream_VIAE, S, A_num, epsilon, gamma, transition_list_size, action_prob, mean, min_variance, max_variance, variance_increment);
	write_meta_data_to_dat_file_normal_dist_varying_variance(stringstream_VIAEH, S, A_num, epsilon, gamma, transition_list_size, action_prob, mean, min_variance, max_variance, variance_increment);

	for (double variance = min_variance; variance <= max_variance; variance += variance_increment)
	{

		// status message of experiment
		printf("\nBeginning iteration variance = %f\n", variance);

		// GENERATE THE MDP FROM CURRENT TIME SEED
		int seed = time(0);
		auto MDP = generate_random_MDP_normal_distributed_rewards(S, A_num, action_prob, transition_list_size, seed, mean, variance);
		R_type R = get<0>(MDP);
		A_type A = get<1>(MDP);
		P_type P = get<2>(MDP);

		// iterations test printing
		float R_max = find_max_R(R);
		float R_min = find_min_R(R);
		float iterations_bound = log(R_max / ((1.0 - gamma) * epsilon)) / log(1.0 / gamma);
		printf("R_max is %f\n", R_max);
		printf("R_min is %f\n", R_min);
		printf("upper bound is %f\n", R_max / (1.0 - gamma));
		printf("lower bound is %f\n", R_min / (1.0 - gamma));
		printf("iterations bound: %f\n", iterations_bound);

		// VI testing
		A_type A1 = copy_A(A);
		auto start_VI = high_resolution_clock::now();

		V_type V_approx_solution_tuple = value_iteration(S, R, A1, P, gamma, epsilon);
		vector<double> V_approx_solution = get<0>(V_approx_solution_tuple);

		auto stop_VI = high_resolution_clock::now();
		auto duration_VI = duration_cast<microseconds>(stop_VI - start_VI);

		stringstream_VI << to_string(variance) << " " << duration_VI.count() << endl;

		// VIU testing
		A_type A6 = copy_A(A);
		auto start_VIU = high_resolution_clock::now();

		V_type V_approx_solution_upper_tuple = value_iteration_upper(S, R, A6, P, gamma, epsilon);
		vector<double> V_approx_solution_upper = get<0>(V_approx_solution_upper_tuple);

		auto stop_VIU = high_resolution_clock::now();
		auto duration_VIU = duration_cast<microseconds>(stop_VIU - start_VIU);

		stringstream_VIU << to_string(variance) << " " << duration_VIU.count() << endl;

		// VIH testing
		A_type A2 = copy_A(A);
		auto start_VIH = high_resolution_clock::now();

		V_type V_heap_approx_tuple = value_iteration_with_heap(S, R, A2, P, gamma, epsilon);
		vector<double> V_heap_approx = get<0>(V_heap_approx_tuple);

		auto stop_VIH = high_resolution_clock::now();
		auto duration_VIH = duration_cast<microseconds>(stop_VIH - start_VIH);

		stringstream_VIH << to_string(variance) << " " << duration_VIH.count() << endl;

		// BVI
		A_type A3 = copy_A(A);
		auto start_BVI = high_resolution_clock::now();

		V_type V_bounded_approx_solution_tuple = bounded_value_iteration(S, R, A3, P, gamma, epsilon);
		vector<double> V_bounded_approx_solution = get<0>(V_bounded_approx_solution_tuple);

		auto stop_BVI = high_resolution_clock::now();
		auto duration_BVI = duration_cast<microseconds>(stop_BVI - start_BVI);

		stringstream_BVI << to_string(variance) << " " << duration_BVI.count() << endl;

		// VIAE
		A_type A4 = copy_A(A);
		auto start_VIAE = high_resolution_clock::now();

		V_type V_AE_approx_solution_tuple = value_iteration_action_elimination(S, R, A4, P, gamma, epsilon);
		vector<double> V_AE_approx_solution = get<0>(V_AE_approx_solution_tuple);

		auto stop_VIAE = high_resolution_clock::now();
		auto duration_VIAE = duration_cast<microseconds>(stop_VIAE - start_VIAE);

		stringstream_VIAE << to_string(variance) << " " << duration_VIAE.count() << endl;

		// VIAEH
		A_type A5 = copy_A(A);
		auto start_VIAEH = high_resolution_clock::now();

		V_type V_AE_H_approx_solution_tuple = value_iteration_action_elimination_heaps(S, R, A5, P, gamma, epsilon);
		vector<double> V_AE_H_approx_solution = get<0>(V_AE_H_approx_solution_tuple);

		auto stop_VIAEH = high_resolution_clock::now();
		auto duration_VIAEH = duration_cast<microseconds>(stop_VIAEH - start_VIAEH);

		stringstream_VIAEH << to_string(variance) << " " << duration_VIAEH.count() << endl;

		// They should in theory all be epsilon from true value, and therefore, at most be 2 * epsilon from each other
		if (abs_max_diff_vectors(V_approx_solution, V_heap_approx) > (2 * epsilon))
		{
			printf("DIFFERENCE\n");
		}
		if (abs_max_diff_vectors(V_AE_approx_solution, V_AE_H_approx_solution) > (2 * epsilon))
		{
			printf("DIFFERENCE\n");
		}
		if (abs_max_diff_vectors(V_bounded_approx_solution, V_approx_solution_upper) > (2 * epsilon))
		{
			printf("DIFFERENCE\n");
		}
	}

	// WRITE ALL DATA TO THEIR RESPECTVIE FILES
	write_stringstream_to_file(stringstream_VI, output_stream_VI, file_name_VI);
	write_stringstream_to_file(stringstream_VIU, output_stream_VIU, file_name_VIU);
	write_stringstream_to_file(stringstream_BVI, output_stream_BVI, file_name_BVI);
	write_stringstream_to_file(stringstream_VIH, output_stream_VIH, file_name_VIH);
	write_stringstream_to_file(stringstream_VIAE, output_stream_VIAE, file_name_VIAE);
	write_stringstream_to_file(stringstream_VIAEH, output_stream_VIAEH, file_name_VIAEH);
}

// VIH DIFFERENT REWARD DISTRIBUTIONS - WORK PER ITERATION
void write_meta_data_to_dat_file_VIH_distributions_iterations_work(ostringstream &string_stream, int S, int A_num, double epsilon, double gamma, int non_zero_transitions, double upper_reward, double action_prob)
{
	time_t time_now = time(0);
	string_stream << "# META DATA" << endl;
	string_stream << "# " << endl;
	string_stream << "# "
				  << "experiment run at: " << ctime(&time_now);
	string_stream << "# " << endl;
	string_stream << "# "
				  << "gamma = " << gamma << endl;
	string_stream << "# "
				  << "epsilon = " << epsilon << endl;
	string_stream << "# "
				  << "upper_reward = " << upper_reward << endl;
	string_stream << "# "
				  << "S = " << S << endl;
	string_stream << "# "
				  << "A = " << A_num << endl;
	string_stream << "# "
				  << "non_zero_transitions = " << non_zero_transitions << endl;
	string_stream << "# "
				  << "action_prob = " << action_prob << endl;
	string_stream << "# " << endl;
	string_stream << "# ACTUAL DATA" << endl;
	string_stream << "# iteration number | top action changes in current iteration" << endl;
}

void create_data_tables_VIH_distributions_iterations_work(string filename, int S, int A_num, double epsilon, double gamma, double action_prob, int number_of_transitions, double mean_1, double mean_2, double mean_3, double variance_1, double variance_2, double variance_3, double lambda_1, double lambda_2, double lambda_3, double upper_reward_1, double upper_reward_2, double upper_reward_3)
{

	// FOR WORK PER ITERATION
	// the stringstreams to create the test for the files
	ostringstream stringstream_uniform_1;
	ostringstream stringstream_uniform_2;
	ostringstream stringstream_uniform_3;
	ostringstream stringstream_normal_1;
	ostringstream stringstream_normal_2;
	ostringstream stringstream_normal_3;
	ostringstream stringstream_exponential_1;
	ostringstream stringstream_exponential_2;
	ostringstream stringstream_exponential_3;

	// the file output objects
	ofstream output_stream_uniform_1;
	ofstream output_stream_uniform_2;
	ofstream output_stream_uniform_3;
	ofstream output_stream_normal_1;
	ofstream output_stream_normal_2;
	ofstream output_stream_normal_3;
	ofstream output_stream_exponential_1;
	ofstream output_stream_exponential_2;
	ofstream output_stream_exponential_3;

	// set the name of the file to write to
	string file_name_uniform_1 = "data_tables/VIH_distributions_iterations_work/" + filename + "_uniform_1.dat";
	string file_name_uniform_2 = "data_tables/VIH_distributions_iterations_work/" + filename + "_uniform_2.dat";
	string file_name_uniform_3 = "data_tables/VIH_distributions_iterations_work/" + filename + "_uniform_3.dat";
	string file_name_normal_1 = "data_tables/VIH_distributions_iterations_work/" + filename + "_normal_1.dat";
	string file_name_normal_2 = "data_tables/VIH_distributions_iterations_work/" + filename + "_normal_2.dat";
	string file_name_normal_3 = "data_tables/VIH_distributions_iterations_work/" + filename + "_normal_3.dat";
	string file_name_exponential_1 = "data_tables/VIH_distributions_iterations_work/" + filename + "_exponential_1.dat";
	string file_name_exponential_2 = "data_tables/VIH_distributions_iterations_work/" + filename + "_exponential_2.dat";
	string file_name_exponential_3 = "data_tables/VIH_distributions_iterations_work/" + filename + "_exponential_3.dat";

	// write meta data to all stringstreams as first in their respective files
	// write_meta_data_to_dat_file_work_per_iteration(stringstream_uniform_1, S, A_num, epsilon, gamma, number_of_transitions, upper_reward, action_prob);
	// write_meta_data_to_dat_file_work_per_iteration(stringstream_uniform_2, S, A_num, epsilon, gamma, number_of_transitions, upper_reward, action_prob);
	// write_meta_data_to_dat_file_work_per_iteration(stringstream_uniform_3, S, A_num, epsilon, gamma, number_of_transitions, upper_reward, action_prob);
	// write_meta_data_to_dat_file_work_per_iteration(stringstream_normal_1, S, A_num, epsilon, gamma, number_of_transitions, upper_reward, action_prob);
	// write_meta_data_to_dat_file_work_per_iteration(stringstream_normal_2, S, A_num, epsilon, gamma, number_of_transitions, upper_reward, action_prob);
	// write_meta_data_to_dat_file_work_per_iteration(stringstream_normal_3, S, A_num, epsilon, gamma, number_of_transitions, upper_reward, action_prob);
	// write_meta_data_to_dat_file_work_per_iteration(stringstream_exponential_1, S, A_num, epsilon, gamma, number_of_transitions, upper_reward, action_prob);
	// write_meta_data_to_dat_file_work_per_iteration(stringstream_exponential_2, S, A_num, epsilon, gamma, number_of_transitions, upper_reward, action_prob);
	// write_meta_data_to_dat_file_work_per_iteration(stringstream_exponential_3, S, A_num, epsilon, gamma, number_of_transitions, upper_reward, action_prob);

	// ACCUMULATED REWARDS
	// FOR WORK PER ITERATION
	// the stringstreams to create the test for the files
	ostringstream stringstream_uniform_acc_1;
	ostringstream stringstream_uniform_acc_2;
	ostringstream stringstream_uniform_acc_3;
	ostringstream stringstream_normal_acc_1;
	ostringstream stringstream_normal_acc_2;
	ostringstream stringstream_normal_acc_3;
	ostringstream stringstream_exponential_acc_1;
	ostringstream stringstream_exponential_acc_2;
	ostringstream stringstream_exponential_acc_3;

	// the file output objects
	ofstream output_stream_uniform_acc_1;
	ofstream output_stream_uniform_acc_2;
	ofstream output_stream_uniform_acc_3;
	ofstream output_stream_normal_acc_1;
	ofstream output_stream_normal_acc_2;
	ofstream output_stream_normal_acc_3;
	ofstream output_stream_exponential_acc_1;
	ofstream output_stream_exponential_acc_2;
	ofstream output_stream_exponential_acc_3;

	// set the name of the file to write to
	string file_name_uniform_acc_1 = "data_tables/VIH_distributions_iterations_work/" + filename + "_uniform_acc_1.dat";
	string file_name_uniform_acc_2 = "data_tables/VIH_distributions_iterations_work/" + filename + "_uniform_acc_2.dat";
	string file_name_uniform_acc_3 = "data_tables/VIH_distributions_iterations_work/" + filename + "_uniform_acc_3.dat";
	string file_name_normal_acc_1 = "data_tables/VIH_distributions_iterations_work/" + filename + "_normal_acc_1.dat";
	string file_name_normal_acc_2 = "data_tables/VIH_distributions_iterations_work/" + filename + "_normal_acc_2.dat";
	string file_name_normal_acc_3 = "data_tables/VIH_distributions_iterations_work/" + filename + "_normal_acc_3.dat";
	string file_name_exponential_acc_1 = "data_tables/VIH_distributions_iterations_work/" + filename + "_exponential_acc_1.dat";
	string file_name_exponential_acc_2 = "data_tables/VIH_distributions_iterations_work/" + filename + "_exponential_acc_2.dat";
	string file_name_exponential_acc_3 = "data_tables/VIH_distributions_iterations_work/" + filename + "_exponential_acc_3.dat";

	// write meta data to all stringstreams as first in their respective files
	// write_meta_data_to_dat_file_work_per_iteration(stringstream_uniform_acc_1, S, A_num, epsilon, gamma, number_of_transitions, upper_reward, action_prob);
	// write_meta_data_to_dat_file_work_per_iteration(stringstream_uniform_acc_2, S, A_num, epsilon, gamma, number_of_transitions, upper_reward, action_prob);
	// write_meta_data_to_dat_file_work_per_iteration(stringstream_uniform_acc_3, S, A_num, epsilon, gamma, number_of_transitions, upper_reward, action_prob);
	// write_meta_data_to_dat_file_work_per_iteration(stringstream_normal_acc_1, S, A_num, epsilon, gamma, number_of_transitions, upper_reward, action_prob);
	// write_meta_data_to_dat_file_work_per_iteration(stringstream_normal_acc_2, S, A_num, epsilon, gamma, number_of_transitions, upper_reward, action_prob);
	// write_meta_data_to_dat_file_work_per_iteration(stringstream_normal_acc_3, S, A_num, epsilon, gamma, number_of_transitions, upper_reward, action_prob);
	// write_meta_data_to_dat_file_work_per_iteration(stringstream_exponential_acc_1, S, A_num, epsilon, gamma, number_of_transitions, upper_reward, action_prob);
	// write_meta_data_to_dat_file_work_per_iteration(stringstream_exponential_acc_2, S, A_num, epsilon, gamma, number_of_transitions, upper_reward, action_prob);
	// write_meta_data_to_dat_file_work_per_iteration(stringstream_exponential_acc_3, S, A_num, epsilon, gamma, number_of_transitions, upper_reward, action_prob);

	// UNIFORM REWARDS 1
	int seed = time(0);
	auto MDP_uniform_1_reward = generate_random_MDP_with_variable_parameters_fixed_nonzero_trans_states(S, A_num, action_prob, number_of_transitions, upper_reward_1, seed);
	R_type R_uniform_1_reward = get<0>(MDP_uniform_1_reward);
	A_type A_uniform_1_reward = get<1>(MDP_uniform_1_reward);
	P_type P_uniform_1_reward = get<2>(MDP_uniform_1_reward);

	printf("Beginning: uniform_1\n");
	A_type A1 = copy_A(A_uniform_1_reward);

	V_type V_heap_approx_tuple_uniform_1 = value_iteration_with_heap(S, R_uniform_1_reward, A1, P_uniform_1_reward, gamma, epsilon);
	vector<double> V_heap_approx_uniform_1 = get<0>(V_heap_approx_tuple_uniform_1);
	int V_heap_approx_iterations_uniform_1 = get<1>(V_heap_approx_tuple_uniform_1);
	vector<microseconds> V_heap_approx_work_per_iteration_uniform_1 = get<2>(V_heap_approx_tuple_uniform_1);

	auto tick_accumulator_VIH_uniform_1 = V_heap_approx_work_per_iteration_uniform_1[0].count();

	for (int iteration = 1; iteration <= V_heap_approx_iterations_uniform_1; iteration++)
	{
		auto iteration_work_uniform_1 = V_heap_approx_work_per_iteration_uniform_1[iteration].count();
		tick_accumulator_VIH_uniform_1 = tick_accumulator_VIH_uniform_1 + iteration_work_uniform_1;

		stringstream_uniform_1 << to_string(iteration) << " " << iteration_work_uniform_1 << endl;
		stringstream_uniform_acc_1 << to_string(iteration) << " " << tick_accumulator_VIH_uniform_1 << endl;
	}

	// UNIFORM REWARDS 2
	seed = time(0);
	auto MDP_uniform_2_reward = generate_random_MDP_with_variable_parameters_fixed_nonzero_trans_states(S, A_num, action_prob, number_of_transitions, upper_reward_2, seed);
	R_type R_uniform_2_reward = get<0>(MDP_uniform_2_reward);
	A_type A_uniform_2_reward = get<1>(MDP_uniform_2_reward);
	P_type P_uniform_2_reward = get<2>(MDP_uniform_2_reward);

	printf("Beginning: uniform_2\n");
	A_type A2 = copy_A(A_uniform_2_reward);

	V_type V_heap_approx_tuple_uniform_2 = value_iteration_with_heap(S, R_uniform_2_reward, A2, P_uniform_2_reward, gamma, epsilon);
	vector<double> V_heap_approx_uniform_2 = get<0>(V_heap_approx_tuple_uniform_2);
	int V_heap_approx_iterations_uniform_2 = get<1>(V_heap_approx_tuple_uniform_2);
	vector<microseconds> V_heap_approx_work_per_iteration_uniform_2 = get<2>(V_heap_approx_tuple_uniform_2);

	auto tick_accumulator_VIH_uniform_2 = V_heap_approx_work_per_iteration_uniform_2[0].count();

	for (int iteration = 1; iteration <= V_heap_approx_iterations_uniform_2; iteration++)
	{
		auto iteration_work_uniform_2 = V_heap_approx_work_per_iteration_uniform_2[iteration].count();
		tick_accumulator_VIH_uniform_2 = tick_accumulator_VIH_uniform_2 + iteration_work_uniform_2;

		stringstream_uniform_2 << to_string(iteration) << " " << iteration_work_uniform_2 << endl;
		stringstream_uniform_acc_2 << to_string(iteration) << " " << tick_accumulator_VIH_uniform_2 << endl;
	}

	// UNIFORM REWARDS 3
	seed = time(0);
	auto MDP_uniform_3_reward = generate_random_MDP_with_variable_parameters_fixed_nonzero_trans_states(S, A_num, action_prob, number_of_transitions, upper_reward_3, seed);
	R_type R_uniform_3_reward = get<0>(MDP_uniform_3_reward);
	A_type A_uniform_3_reward = get<1>(MDP_uniform_3_reward);
	P_type P_uniform_3_reward = get<2>(MDP_uniform_3_reward);

	printf("Beginning: uniform_3\n");
	A_type A3 = copy_A(A_uniform_3_reward);

	V_type V_heap_approx_tuple_uniform_3 = value_iteration_with_heap(S, R_uniform_3_reward, A3, P_uniform_3_reward, gamma, epsilon);
	vector<double> V_heap_approx_uniform_3 = get<0>(V_heap_approx_tuple_uniform_3);
	int V_heap_approx_iterations_uniform_3 = get<1>(V_heap_approx_tuple_uniform_3);
	vector<microseconds> V_heap_approx_work_per_iteration_uniform_3 = get<2>(V_heap_approx_tuple_uniform_3);

	auto tick_accumulator_VIH_uniform_3 = V_heap_approx_work_per_iteration_uniform_3[0].count();

	for (int iteration = 1; iteration <= V_heap_approx_iterations_uniform_3; iteration++)
	{
		auto iteration_work_uniform_3 = V_heap_approx_work_per_iteration_uniform_3[iteration].count();
		tick_accumulator_VIH_uniform_3 = tick_accumulator_VIH_uniform_3 + iteration_work_uniform_3;

		stringstream_uniform_3 << to_string(iteration) << " " << iteration_work_uniform_3 << endl;
		stringstream_uniform_acc_3 << to_string(iteration) << " " << tick_accumulator_VIH_uniform_3 << endl;
	}

	// NORMAL REWARDS 1
	seed = time(0);
	auto MDP_normal_1_reward = generate_random_MDP_normal_distributed_rewards(S, A_num, action_prob, number_of_transitions, seed, mean_1, variance_1);
	R_type R_normal_1_reward = get<0>(MDP_normal_1_reward);
	A_type A_normal_1_reward = get<1>(MDP_normal_1_reward);
	P_type P_normal_1_reward = get<2>(MDP_normal_1_reward);

	printf("Beginning: normal_1\n");
	A_type A4 = copy_A(A_normal_1_reward);

	V_type V_heap_approx_tuple_normal_1 = value_iteration_with_heap(S, R_normal_1_reward, A4, P_normal_1_reward, gamma, epsilon);
	vector<double> V_heap_approx_normal_1 = get<0>(V_heap_approx_tuple_normal_1);
	int V_heap_approx_iterations_normal_1 = get<1>(V_heap_approx_tuple_normal_1);
	vector<microseconds> V_heap_approx_work_per_iteration_normal_1 = get<2>(V_heap_approx_tuple_normal_1);

	auto tick_accumulator_VIH_normal_1 = V_heap_approx_work_per_iteration_normal_1[0].count();

	for (int iteration = 1; iteration <= V_heap_approx_iterations_normal_1; iteration++)
	{
		auto iteration_work_normal_1 = V_heap_approx_work_per_iteration_normal_1[iteration].count();
		tick_accumulator_VIH_normal_1 = tick_accumulator_VIH_normal_1 + iteration_work_normal_1;

		stringstream_normal_1 << to_string(iteration) << " " << iteration_work_normal_1 << endl;
		stringstream_normal_acc_1 << to_string(iteration) << " " << tick_accumulator_VIH_normal_1 << endl;
	}

	// NORMAL REWARDS 2
	seed = time(0);
	auto MDP_normal_2_reward = generate_random_MDP_normal_distributed_rewards(S, A_num, action_prob, number_of_transitions, seed, mean_2, variance_2);
	R_type R_normal_2_reward = get<0>(MDP_normal_2_reward);
	A_type A_normal_2_reward = get<1>(MDP_normal_2_reward);
	P_type P_normal_2_reward = get<2>(MDP_normal_2_reward);

	printf("Beginning: normal_2\n");
	A_type A5 = copy_A(A_normal_2_reward);

	V_type V_heap_approx_tuple_normal_2 = value_iteration_with_heap(S, R_normal_2_reward, A5, P_normal_2_reward, gamma, epsilon);
	vector<double> V_heap_approx_normal_2 = get<0>(V_heap_approx_tuple_normal_2);
	int V_heap_approx_iterations_normal_2 = get<1>(V_heap_approx_tuple_normal_2);
	vector<microseconds> V_heap_approx_work_per_iteration_normal_2 = get<2>(V_heap_approx_tuple_normal_2);

	auto tick_accumulator_VIH_normal_2 = V_heap_approx_work_per_iteration_normal_2[0].count();

	for (int iteration = 1; iteration <= V_heap_approx_iterations_normal_2; iteration++)
	{
		auto iteration_work_normal_2 = V_heap_approx_work_per_iteration_normal_2[iteration].count();
		tick_accumulator_VIH_normal_2 = tick_accumulator_VIH_normal_2 + iteration_work_normal_2;

		stringstream_normal_2 << to_string(iteration) << " " << iteration_work_normal_2 << endl;
		stringstream_normal_acc_2 << to_string(iteration) << " " << tick_accumulator_VIH_normal_2 << endl;
	}

	// NORMAL REWARDS 3
	seed = time(0);
	auto MDP_normal_3_reward = generate_random_MDP_normal_distributed_rewards(S, A_num, action_prob, number_of_transitions, seed, mean_3, variance_3);
	R_type R_normal_3_reward = get<0>(MDP_normal_3_reward);
	A_type A_normal_3_reward = get<1>(MDP_normal_3_reward);
	P_type P_normal_3_reward = get<2>(MDP_normal_3_reward);

	printf("Beginning: normal_3\n");
	A_type A6 = copy_A(A_normal_3_reward);

	V_type V_heap_approx_tuple_normal_3 = value_iteration_with_heap(S, R_normal_3_reward, A6, P_normal_3_reward, gamma, epsilon);
	vector<double> V_heap_approx_normal_3 = get<0>(V_heap_approx_tuple_normal_3);
	int V_heap_approx_iterations_normal_3 = get<1>(V_heap_approx_tuple_normal_3);
	vector<microseconds> V_heap_approx_work_per_iteration_normal_3 = get<2>(V_heap_approx_tuple_normal_3);

	auto tick_accumulator_VIH_normal_3 = V_heap_approx_work_per_iteration_normal_3[0].count();

	for (int iteration = 1; iteration <= V_heap_approx_iterations_normal_3; iteration++)
	{
		auto iteration_work_normal_3 = V_heap_approx_work_per_iteration_normal_3[iteration].count();
		tick_accumulator_VIH_normal_3 = tick_accumulator_VIH_normal_3 + iteration_work_normal_3;

		stringstream_normal_3 << to_string(iteration) << " " << iteration_work_normal_3 << endl;
		stringstream_normal_acc_3 << to_string(iteration) << " " << tick_accumulator_VIH_normal_3 << endl;
	}

	// EXPONENTIAL REWARDS 1
	seed = time(0);
	auto MDP_exponential_1_reward = generate_random_MDP_exponential_distributed_rewards(S, A_num, action_prob, number_of_transitions, lambda_1, seed);
	R_type R_exponential_1_reward = get<0>(MDP_exponential_1_reward);
	A_type A_exponential_1_reward = get<1>(MDP_exponential_1_reward);
	P_type P_exponential_1_reward = get<2>(MDP_exponential_1_reward);

	printf("Beginning: exponential_1\n");
	A_type A7 = copy_A(A_exponential_1_reward);

	V_type V_heap_approx_tuple_exponential_1 = value_iteration_with_heap(S, R_exponential_1_reward, A7, P_exponential_1_reward, gamma, epsilon);
	vector<double> V_heap_approx_exponential_1 = get<0>(V_heap_approx_tuple_exponential_1);
	int V_heap_approx_iterations_exponential_1 = get<1>(V_heap_approx_tuple_exponential_1);
	vector<microseconds> V_heap_approx_work_per_iteration_exponential_1 = get<2>(V_heap_approx_tuple_exponential_1);

	auto tick_accumulator_VIH_exponential_1 = V_heap_approx_work_per_iteration_exponential_1[0].count();

	for (int iteration = 1; iteration <= V_heap_approx_iterations_exponential_1; iteration++)
	{
		auto iteration_work_exponential_1 = V_heap_approx_work_per_iteration_exponential_1[iteration].count();
		tick_accumulator_VIH_exponential_1 = tick_accumulator_VIH_exponential_1 + iteration_work_exponential_1;

		stringstream_exponential_1 << to_string(iteration) << " " << iteration_work_exponential_1 << endl;
		stringstream_exponential_acc_1 << to_string(iteration) << " " << tick_accumulator_VIH_exponential_1 << endl;
	}

	// EXPONENTIAL REWARDS 2
	seed = time(0);
	auto MDP_exponential_2_reward = generate_random_MDP_exponential_distributed_rewards(S, A_num, action_prob, number_of_transitions, lambda_2, seed);
	R_type R_exponential_2_reward = get<0>(MDP_exponential_2_reward);
	A_type A_exponential_2_reward = get<1>(MDP_exponential_2_reward);
	P_type P_exponential_2_reward = get<2>(MDP_exponential_2_reward);

	printf("Beginning: exponential_2\n");
	A_type A8 = copy_A(A_exponential_2_reward);

	V_type V_heap_approx_tuple_exponential_2 = value_iteration_with_heap(S, R_exponential_2_reward, A8, P_exponential_2_reward, gamma, epsilon);
	vector<double> V_heap_approx_exponential_2 = get<0>(V_heap_approx_tuple_exponential_2);
	int V_heap_approx_iterations_exponential_2 = get<1>(V_heap_approx_tuple_exponential_2);
	vector<microseconds> V_heap_approx_work_per_iteration_exponential_2 = get<2>(V_heap_approx_tuple_exponential_2);

	auto tick_accumulator_VIH_exponential_2 = V_heap_approx_work_per_iteration_exponential_2[0].count();

	for (int iteration = 1; iteration <= V_heap_approx_iterations_exponential_2; iteration++)
	{
		auto iteration_work_exponential_2 = V_heap_approx_work_per_iteration_exponential_2[iteration].count();
		tick_accumulator_VIH_exponential_2 = tick_accumulator_VIH_exponential_2 + iteration_work_exponential_2;

		stringstream_exponential_2 << to_string(iteration) << " " << iteration_work_exponential_2 << endl;
		stringstream_exponential_acc_2 << to_string(iteration) << " " << tick_accumulator_VIH_exponential_2 << endl;
	}

	// EXPONENTIAL REWARDS 3
	seed = time(0);
	auto MDP_exponential_3_reward = generate_random_MDP_exponential_distributed_rewards(S, A_num, action_prob, number_of_transitions, lambda_3, seed);
	R_type R_exponential_3_reward = get<0>(MDP_exponential_3_reward);
	A_type A_exponential_3_reward = get<1>(MDP_exponential_3_reward);
	P_type P_exponential_3_reward = get<2>(MDP_exponential_3_reward);

	printf("Beginning: exponential_3\n");
	A_type A9 = copy_A(A_exponential_3_reward);

	V_type V_heap_approx_tuple_exponential_3 = value_iteration_with_heap(S, R_exponential_3_reward, A9, P_exponential_3_reward, gamma, epsilon);
	vector<double> V_heap_approx_exponential_3 = get<0>(V_heap_approx_tuple_exponential_3);
	int V_heap_approx_iterations_exponential_3 = get<1>(V_heap_approx_tuple_exponential_3);
	vector<microseconds> V_heap_approx_work_per_iteration_exponential_3 = get<2>(V_heap_approx_tuple_exponential_3);

	auto tick_accumulator_VIH_exponential_3 = V_heap_approx_work_per_iteration_exponential_3[0].count();

	for (int iteration = 1; iteration <= V_heap_approx_iterations_exponential_3; iteration++)
	{
		auto iteration_work_exponential_3 = V_heap_approx_work_per_iteration_exponential_3[iteration].count();
		tick_accumulator_VIH_exponential_3 = tick_accumulator_VIH_exponential_3 + iteration_work_exponential_3;

		stringstream_exponential_3 << to_string(iteration) << " " << iteration_work_exponential_3 << endl;
		stringstream_exponential_acc_3 << to_string(iteration) << " " << tick_accumulator_VIH_exponential_3 << endl;
	}

	// WRITE ALL DATA TO THEIR RESPECTVIE FILES
	write_stringstream_to_file(stringstream_normal_1, output_stream_normal_1, file_name_normal_1);
	write_stringstream_to_file(stringstream_normal_2, output_stream_normal_2, file_name_normal_2);
	write_stringstream_to_file(stringstream_normal_3, output_stream_normal_3, file_name_normal_3);
	write_stringstream_to_file(stringstream_uniform_1, output_stream_uniform_1, file_name_uniform_1);
	write_stringstream_to_file(stringstream_uniform_2, output_stream_uniform_2, file_name_uniform_2);
	write_stringstream_to_file(stringstream_uniform_3, output_stream_uniform_3, file_name_uniform_3);
	write_stringstream_to_file(stringstream_exponential_1, output_stream_exponential_1, file_name_exponential_1);
	write_stringstream_to_file(stringstream_exponential_2, output_stream_exponential_2, file_name_exponential_2);
	write_stringstream_to_file(stringstream_exponential_3, output_stream_exponential_3, file_name_exponential_3);

	// WRITE ALL DATA TO THEIR RESPECTVIE FILES
	write_stringstream_to_file(stringstream_normal_acc_1, output_stream_normal_acc_1, file_name_normal_acc_1);
	write_stringstream_to_file(stringstream_normal_acc_2, output_stream_normal_acc_2, file_name_normal_acc_2);
	write_stringstream_to_file(stringstream_normal_acc_3, output_stream_normal_acc_3, file_name_normal_acc_3);
	write_stringstream_to_file(stringstream_uniform_acc_1, output_stream_uniform_acc_1, file_name_uniform_acc_1);
	write_stringstream_to_file(stringstream_uniform_acc_2, output_stream_uniform_acc_2, file_name_uniform_acc_2);
	write_stringstream_to_file(stringstream_uniform_acc_3, output_stream_uniform_acc_3, file_name_uniform_acc_3);
	write_stringstream_to_file(stringstream_exponential_acc_1, output_stream_exponential_acc_1, file_name_exponential_acc_1);
	write_stringstream_to_file(stringstream_exponential_acc_2, output_stream_exponential_acc_2, file_name_exponential_acc_2);
	write_stringstream_to_file(stringstream_exponential_acc_3, output_stream_exponential_acc_3, file_name_exponential_acc_3);
}

// EXPONENTIAL EXPERIMENT - VARYING LAMBDA
void write_meta_data_to_dat_file_exponential_dist_varying_lambda(ostringstream &string_stream, int S, int A_num, double epsilon, double gamma, int transition_list_size, double action_prob, double min_lambda, double max_lambda, double lambda_increment)
{
	time_t time_now = time(0);
	string_stream << "# META DATA" << endl;
	string_stream << "# " << endl;
	string_stream << "# "
				  << "experiment run at: " << ctime(&time_now);
	string_stream << "# " << endl;
	string_stream << "# "
				  << "gamma = " << gamma << endl;
	string_stream << "# "
				  << "epsilon = " << epsilon << endl;
	string_stream << "# "
				  << "S = " << S << endl;
	string_stream << "# "
				  << "A = " << A_num << endl;
	string_stream << "# "
				  << "transition list size in each state = " << transition_list_size << endl;
	string_stream << "# "
				  << "action_prob = " << action_prob << endl;
	string_stream << "# " << endl;
	string_stream << "# VARIABLE IN THIS EXPERIMENT" << endl;
	string_stream << "# "
				  << "lambda is varied from " << min_lambda << " to " << max_lambda << " with " << lambda_increment << " increment" << endl;
	string_stream << "# " << endl;
	string_stream << "# ACTUAL DATA" << endl;
	string_stream << "# lambda | microseconds" << endl;
}

void create_data_tables_exponential_dist_varying_lambda(string filename, int S, int A_num, double epsilon, double gamma, double action_prob, int transition_list_size, double min_lambda, double max_lambda, double lambda_increment)
{

	// the stringstreams to create the test for the files
	ostringstream stringstream_BVI;
	ostringstream stringstream_VI;
	ostringstream stringstream_VIU;
	ostringstream stringstream_VIH;
	ostringstream stringstream_VIAE;
	ostringstream stringstream_VIAEH;

	// the file output objects
	ofstream output_stream_BVI;
	ofstream output_stream_VI;
	ofstream output_stream_VIU;
	ofstream output_stream_VIH;
	ofstream output_stream_VIAE;
	ofstream output_stream_VIAEH;

	// set the name of the file to write to
	string file_name_BVI = "data_tables/exponential_dist_varying_lambda/" + filename + "_BVI.dat";
	string file_name_VI = "data_tables/exponential_dist_varying_lambda/" + filename + "_VI.dat";
	string file_name_VIU = "data_tables/exponential_dist_varying_lambda/" + filename + "_VIU.dat";
	string file_name_VIH = "data_tables/exponential_dist_varying_lambda/" + filename + "_VIH.dat";
	string file_name_VIAE = "data_tables/exponential_dist_varying_lambda/" + filename + "_VIAE.dat";
	string file_name_VIAEH = "data_tables/exponential_dist_varying_lambda/" + filename + "_VIAEH.dat";

	// write meta data to all stringstreams as first in their respective files
	write_meta_data_to_dat_file_exponential_dist_varying_lambda(stringstream_VI, S, A_num, epsilon, gamma, transition_list_size, action_prob, min_lambda, max_lambda, lambda_increment);
	write_meta_data_to_dat_file_exponential_dist_varying_lambda(stringstream_VIU, S, A_num, epsilon, gamma, transition_list_size, action_prob, min_lambda, max_lambda, lambda_increment);
	write_meta_data_to_dat_file_exponential_dist_varying_lambda(stringstream_VIH, S, A_num, epsilon, gamma, transition_list_size, action_prob, min_lambda, max_lambda, lambda_increment);
	write_meta_data_to_dat_file_exponential_dist_varying_lambda(stringstream_BVI, S, A_num, epsilon, gamma, transition_list_size, action_prob, min_lambda, max_lambda, lambda_increment);
	write_meta_data_to_dat_file_exponential_dist_varying_lambda(stringstream_VIAE, S, A_num, epsilon, gamma, transition_list_size, action_prob, min_lambda, max_lambda, lambda_increment);
	write_meta_data_to_dat_file_exponential_dist_varying_lambda(stringstream_VIAEH, S, A_num, epsilon, gamma, transition_list_size, action_prob, min_lambda, max_lambda, lambda_increment);

	for (double lambda = min_lambda; lambda <= max_lambda; lambda += lambda_increment)
	{

		// status message of experiment
		printf("\nBeginning iteration lambda = %f\n", lambda);

		// GENERATE THE MDP FROM CURRENT TIME SEED
		int seed = time(0);
		auto MDP = generate_random_MDP_exponential_distributed_rewards(S, A_num, action_prob, transition_list_size, lambda, seed);
		R_type R = get<0>(MDP);
		A_type A = get<1>(MDP);
		P_type P = get<2>(MDP);

		// iterations test printing
		float R_max = find_max_R(R);
		float R_min = find_min_R(R);
		float iterations_bound = log(R_max / ((1.0 - gamma) * epsilon)) / log(1.0 / gamma);
		printf("R_max is %f\n", R_max);
		printf("R_min is %f\n", R_min);
		printf("upper bound is %f\n", R_max / (1.0 - gamma));
		printf("lower bound is %f\n", R_min / (1.0 - gamma));
		printf("iterations bound: %f\n", iterations_bound);

		// VI testing
		A_type A1 = copy_A(A);
		auto start_VI = high_resolution_clock::now();

		V_type V_approx_solution_tuple = value_iteration(S, R, A1, P, gamma, epsilon);
		vector<double> V_approx_solution = get<0>(V_approx_solution_tuple);

		auto stop_VI = high_resolution_clock::now();
		auto duration_VI = duration_cast<microseconds>(stop_VI - start_VI);

		stringstream_VI << to_string(lambda) << " " << duration_VI.count() << endl;

		// VIU testing
		A_type A6 = copy_A(A);
		auto start_VIU = high_resolution_clock::now();

		V_type V_approx_solution_upper_tuple = value_iteration_upper(S, R, A6, P, gamma, epsilon);
		vector<double> V_approx_solution_upper = get<0>(V_approx_solution_upper_tuple);

		auto stop_VIU = high_resolution_clock::now();
		auto duration_VIU = duration_cast<microseconds>(stop_VIU - start_VIU);

		stringstream_VIU << to_string(lambda) << " " << duration_VIU.count() << endl;

		// VIH testing
		A_type A2 = copy_A(A);
		auto start_VIH = high_resolution_clock::now();

		V_type V_heap_approx_tuple = value_iteration_with_heap(S, R, A2, P, gamma, epsilon);
		vector<double> V_heap_approx = get<0>(V_heap_approx_tuple);

		auto stop_VIH = high_resolution_clock::now();
		auto duration_VIH = duration_cast<microseconds>(stop_VIH - start_VIH);

		stringstream_VIH << to_string(lambda) << " " << duration_VIH.count() << endl;

		// BVI
		A_type A3 = copy_A(A);
		auto start_BVI = high_resolution_clock::now();

		V_type V_bounded_approx_solution_tuple = bounded_value_iteration(S, R, A3, P, gamma, epsilon);
		vector<double> V_bounded_approx_solution = get<0>(V_bounded_approx_solution_tuple);

		auto stop_BVI = high_resolution_clock::now();
		auto duration_BVI = duration_cast<microseconds>(stop_BVI - start_BVI);

		stringstream_BVI << to_string(lambda) << " " << duration_BVI.count() << endl;

		// VIAE
		A_type A4 = copy_A(A);
		auto start_VIAE = high_resolution_clock::now();

		V_type V_AE_approx_solution_tuple = value_iteration_action_elimination(S, R, A4, P, gamma, epsilon);
		vector<double> V_AE_approx_solution = get<0>(V_AE_approx_solution_tuple);

		auto stop_VIAE = high_resolution_clock::now();
		auto duration_VIAE = duration_cast<microseconds>(stop_VIAE - start_VIAE);

		stringstream_VIAE << to_string(lambda) << " " << duration_VIAE.count() << endl;

		// VIAEH
		A_type A5 = copy_A(A);
		auto start_VIAEH = high_resolution_clock::now();

		V_type V_AE_H_approx_solution_tuple = value_iteration_action_elimination_heaps(S, R, A5, P, gamma, epsilon);
		vector<double> V_AE_H_approx_solution = get<0>(V_AE_H_approx_solution_tuple);

		auto stop_VIAEH = high_resolution_clock::now();
		auto duration_VIAEH = duration_cast<microseconds>(stop_VIAEH - start_VIAEH);

		stringstream_VIAEH << to_string(lambda) << " " << duration_VIAEH.count() << endl;

		// They should in theory all be epsilon from true value, and therefore, at most be 2 * epsilon from each other
		if (abs_max_diff_vectors(V_approx_solution, V_heap_approx) > (2 * epsilon))
		{
			printf("DIFFERENCE\n");
		}
		if (abs_max_diff_vectors(V_AE_approx_solution, V_AE_H_approx_solution) > (2 * epsilon))
		{
			printf("DIFFERENCE\n");
		}
		if (abs_max_diff_vectors(V_bounded_approx_solution, V_approx_solution_upper) > (2 * epsilon))
		{
			printf("DIFFERENCE\n");
		}
	}

	// WRITE ALL DATA TO THEIR RESPECTVIE FILES
	write_stringstream_to_file(stringstream_VI, output_stream_VI, file_name_VI);
	write_stringstream_to_file(stringstream_VIU, output_stream_VIU, file_name_VIU);
	write_stringstream_to_file(stringstream_BVI, output_stream_BVI, file_name_BVI);
	write_stringstream_to_file(stringstream_VIH, output_stream_VIH, file_name_VIH);
	write_stringstream_to_file(stringstream_VIAE, output_stream_VIAE, file_name_VIAE);
	write_stringstream_to_file(stringstream_VIAEH, output_stream_VIAEH, file_name_VIAEH);
}

// VIAEH IMPLEMENTATIONS WORK PER ITERATION TEST
void create_data_tables_work_per_iteration_VIAEH_implementations(string filename, int S, int A_num, double epsilon, double gamma, double action_prob, int number_of_transitions, double mean, double variance)
{

	// FOR WORK PER ITERATION
	// the stringstreams to create the test for the files
	ostringstream stringstream_VIAEH;
	ostringstream stringstream_VIAEH_no_pointers;
	ostringstream stringstream_VIAEH_lazy_update;
	ostringstream stringstream_VIAEH_set;
	ostringstream stringstream_VIAEH_maxmin_heap;
	ostringstream stringstream_VIAEH_approx_lower_bound;

	// the file output objects
	ofstream output_stream_VIAEH;
	ofstream output_stream_VIAEH_no_pointers;
	ofstream output_stream_VIAEH_lazy_update;
	ofstream output_stream_VIAEH_set;
	ofstream output_stream_VIAEH_maxmin_heap;
	ofstream output_stream_VIAEH_approx_lower_bound;

	// set the name of the file to write to
	string file_name_VIAEH = "data_tables/work_per_iteration_VIAEH_implementations/" + filename + "_VIAEH.dat";
	string file_name_VIAEH_no_pointers = "data_tables/work_per_iteration_VIAEH_implementations/" + filename + "_VIAEH_no_pointers.dat";
	string file_name_VIAEH_lazy_update = "data_tables/work_per_iteration_VIAEH_implementations/" + filename + "_VIAEH_lazy_update.dat";
	string file_name_VIAEH_set = "data_tables/work_per_iteration_VIAEH_implementations/" + filename + "_VIAEH_set.dat";
	string file_name_VIAEH_maxmin_heap = "data_tables/work_per_iteration_VIAEH_implementations/" + filename + "_VIAEH_maxmin_heap.dat";
	string file_name_VIAEH_approx_lower_bound = "data_tables/work_per_iteration_VIAEH_implementations/" + filename + "_VIAEH_approx_lower_bound.dat";

	// write meta data to all stringstreams as first in their respective files
	// write_meta_data_to_dat_file_work_per_iteration(stringstream_VIAEH, S, A_num, epsilon, gamma, non_zero_transition, upper_reward, action_prob);

	// FOR ACCUMULATED WORK
	// the stringstreams to create the test for the files
	ostringstream stringstream_accum_VIAEH;
	ostringstream stringstream_accum_VIAEH_no_pointers;
	ostringstream stringstream_accum_VIAEH_lazy_update;
	ostringstream stringstream_accum_VIAEH_set;
	ostringstream stringstream_accum_VIAEH_maxmin_heap;
	ostringstream stringstream_accum_VIAEH_approx_lower_bound;

	// the file output objects
	ofstream output_stream_accum_VIAEH;
	ofstream output_stream_accum_VIAEH_no_pointers;
	ofstream output_stream_accum_VIAEH_lazy_update;
	ofstream output_stream_accum_VIAEH_set;
	ofstream output_stream_accum_VIAEH_maxmin_heap;
	ofstream output_stream_accum_VIAEH_approx_lower_bound;

	// set the name of the file to write to
	string file_name_accum_VIAEH = "data_tables/work_per_iteration_VIAEH_implementations/" + filename + "_accum_VIAEH.dat";
	string file_name_accum_VIAEH_no_pointers = "data_tables/work_per_iteration_VIAEH_implementations/" + filename + "_accum_VIAEH_no_pointers.dat";
	string file_name_accum_VIAEH_lazy_update = "data_tables/work_per_iteration_VIAEH_implementations/" + filename + "_accum_VIAEH_lazy_update.dat";
	string file_name_accum_VIAEH_set = "data_tables/work_per_iteration_VIAEH_implementations/" + filename + "_accum_VIAEH_set.dat";
	string file_name_accum_VIAEH_maxmin_heap = "data_tables/work_per_iteration_VIAEH_implementations/" + filename + "_accum_VIAEH_maxmin_heap.dat";
	string file_name_accum_VIAEH_approx_lower_bound = "data_tables/work_per_iteration_VIAEH_implementations/" + filename + "_accum_VIAEH_approx_lower_bound.dat";

	// FOR ACTION ELIMINATION
	// the stringstreams to create the test for the files
	ostringstream stringstream_action_elimination_VIAEH;
	ostringstream stringstream_action_elimination_VIAE;
	ostringstream stringstream_action_elimination_VIAEH_no_pointers;
	ostringstream stringstream_action_elimination_VIAEH_lazy_update;
	ostringstream stringstream_action_elimination_VIAEH_set;
	ostringstream stringstream_action_elimination_VIAEH_maxmin_heap;
	ostringstream stringstream_action_elimination_VIAEH_approx_lower_bound;

	// the file output objects
	ofstream output_stream_action_elimination_VIAEH;
	ofstream output_stream_action_elimination_VIAE;
	ofstream output_stream_action_elimination_VIAEH_no_pointers;
	ofstream output_stream_action_elimination_VIAEH_lazy_update;
	ofstream output_stream_action_elimination_VIAEH_set;
	ofstream output_stream_action_elimination_VIAEH_maxmin_heap;
	ofstream output_stream_action_elimination_VIAEH_approx_lower_bound;

	// set the name of the file to write to
	string file_name_action_elimination_VIAEH = "data_tables/work_per_iteration_VIAEH_implementations/" + filename + "_action_elimination_VIAEH.dat";
	string file_name_action_elimination_VIAE = "data_tables/work_per_iteration_VIAEH_implementations/" + filename + "_action_elimination_VIAE.dat";
	string file_name_action_elimination_VIAEH_no_pointers = "data_tables/work_per_iteration_VIAEH_implementations/" + filename + "_action_elimination_VIAEH_no_pointers.dat";
	string file_name_action_elimination_VIAEH_lazy_update = "data_tables/work_per_iteration_VIAEH_implementations/" + filename + "_action_elimination_VIAEH_lazy_update.dat";
	string file_name_action_elimination_VIAEH_set = "data_tables/work_per_iteration_VIAEH_implementations/" + filename + "_action_elimination_VIAEH_set.dat";
	string file_name_action_elimination_VIAEH_maxmin_heap = "data_tables/work_per_iteration_VIAEH_implementations/" + filename + "_action_elimination_VIAEH_maxmin_heap.dat";
	string file_name_action_elimination_VIAEH_approx_lower_bound = "data_tables/work_per_iteration_VIAEH_implementations/" + filename + "_action_elimination_VIAEH_approx_lower_bound.dat";

	// write meta data to all stringstreams as first in their respective files
	// write_meta_data_to_dat_file_work_per_iteration(stringstream_accum_VIAEH, S, A_num, epsilon, gamma, non_zero_transition, upper_reward, action_prob);

	// BEGIN EXPERIMENTATION
	// GENERATE THE MDP FROM CURRENT TIME SEED
	int seed = time(0);
	printf("seed: %d\n", seed);

	// TODO permament change to normal distribution here?
	auto MDP = generate_random_MDP_normal_distributed_rewards(S, A_num, action_prob, number_of_transitions, seed, mean, variance);
	R_type R = get<0>(MDP);
	A_type A = get<1>(MDP);
	P_type P = get<2>(MDP);

	// VIAEH
	printf("VIAEH\n");
	A_type A1 = copy_A(A);

	V_type V_AE_H_approx_solution_tuple = value_iteration_action_elimination_heapsGS(S, R, A1, P, gamma, epsilon);
	vector<double> V_AE_H_approx_solution = get<0>(V_AE_H_approx_solution_tuple);
	int V_AE_H_approx_solution_iterations = get<1>(V_AE_H_approx_solution_tuple);
	vector<microseconds> V_AE_H_approx_solution_work_per_iteration = get<2>(V_AE_H_approx_solution_tuple);
	vector<vector<pair<int, int>>> V_AE_H_approx_solution_actions_eliminated_per_iteration = get<3>(V_AE_H_approx_solution_tuple);

	auto tick_accumulator_VIAEH = V_AE_H_approx_solution_work_per_iteration[0].count();

	for (int iteration = 1; iteration <= V_AE_H_approx_solution_iterations; iteration++)
	{
		auto iteration_work = V_AE_H_approx_solution_work_per_iteration[iteration].count();
		auto actions_eliminated_iteration = V_AE_H_approx_solution_actions_eliminated_per_iteration[iteration].size();
		tick_accumulator_VIAEH = tick_accumulator_VIAEH + iteration_work;

		stringstream_VIAEH << to_string(iteration) << " " << iteration_work << endl;
		stringstream_action_elimination_VIAEH << to_string(iteration) << " " << actions_eliminated_iteration << endl;
		stringstream_accum_VIAEH << to_string(iteration) << " " << tick_accumulator_VIAEH << endl;
	}

	// VIAE
	printf("VIAE\n");
	A_type A5 = copy_A(A);

	V_type V_AE_approx_solution_tuple = value_iteration_action_eliminationGS(S, R, A5, P, gamma, epsilon);
	vector<double> V_AE_approx_solution = get<0>(V_AE_approx_solution_tuple);
	int V_AE_approx_solution_iterations = get<1>(V_AE_approx_solution_tuple);
	vector<microseconds> V_AE_approx_solution_work_per_iteration = get<2>(V_AE_approx_solution_tuple);
	vector<vector<pair<int, int>>> V_AE_approx_solution_actions_eliminated_per_iteration = get<3>(V_AE_approx_solution_tuple);

	auto tick_accumulator_VIAE = V_AE_approx_solution_work_per_iteration[0].count();

	for (int iteration = 1; iteration <= V_AE_approx_solution_iterations; iteration++)
	{
		auto iteration_work = V_AE_approx_solution_work_per_iteration[iteration].count();
		auto actions_eliminated_iteration = V_AE_approx_solution_actions_eliminated_per_iteration[iteration].size();
		tick_accumulator_VIAE = tick_accumulator_VIAE + iteration_work;

		// stringstream_VIAE << to_string(iteration) << " " << iteration_work << endl;
		stringstream_action_elimination_VIAE << to_string(iteration) << " " << actions_eliminated_iteration << endl;
		// stringstream_accum_VIAE << to_string(iteration) << " " << tick_accumulator_VIAE << endl;
	}

	// VIAEH_no_pointers
	printf("VIAEH_no_pointers\n");
	A_type A2 = copy_A(A);

	V_type V_AE_H_no_pointers_approx_solution_tuple = value_iteration_action_elimination_heaps_no_pointers(S, R, A2, P, gamma, epsilon);
	vector<double> V_AE_H_no_pointers_approx_solution = get<0>(V_AE_H_no_pointers_approx_solution_tuple);
	int V_AE_H_no_pointers_approx_solution_iterations = get<1>(V_AE_H_no_pointers_approx_solution_tuple);
	vector<microseconds> V_AE_H_no_pointers_approx_solution_work_per_iteration = get<2>(V_AE_H_no_pointers_approx_solution_tuple);
	vector<vector<pair<int, int>>> V_AE_H_no_pointers_approx_solution_actions_eliminated_per_iteration = get<3>(V_AE_H_no_pointers_approx_solution_tuple);

	auto tick_accumulator_VIAEH_no_pointers = V_AE_H_no_pointers_approx_solution_work_per_iteration[0].count();

	for (int iteration = 1; iteration <= V_AE_H_no_pointers_approx_solution_iterations; iteration++)
	{
		auto iteration_work = V_AE_H_no_pointers_approx_solution_work_per_iteration[iteration].count();
		auto actions_eliminated_iteration = V_AE_H_no_pointers_approx_solution_actions_eliminated_per_iteration[iteration].size();
		tick_accumulator_VIAEH_no_pointers = tick_accumulator_VIAEH_no_pointers + iteration_work;

		stringstream_VIAEH_no_pointers << to_string(iteration) << " " << iteration_work << endl;
		stringstream_action_elimination_VIAEH_no_pointers << to_string(iteration) << " " << actions_eliminated_iteration << endl;
		stringstream_accum_VIAEH_no_pointers << to_string(iteration) << " " << tick_accumulator_VIAEH_no_pointers << endl;
	}

	// VIAEH_lazy_update
	printf("VIAEH_lazy_update\n");
	A_type A3 = copy_A(A);

	V_type V_AE_H_lazy_update_approx_solution_tuple = value_iteration_action_elimination_heaps_lazy_update(S, R, A3, P, gamma, epsilon);
	vector<double> V_AE_H_lazy_update_approx_solution = get<0>(V_AE_H_lazy_update_approx_solution_tuple);
	int V_AE_H_lazy_update_approx_solution_iterations = get<1>(V_AE_H_lazy_update_approx_solution_tuple);
	vector<microseconds> V_AE_H_lazy_update_approx_solution_work_per_iteration = get<2>(V_AE_H_lazy_update_approx_solution_tuple);
	vector<vector<pair<int, int>>> V_AE_H_lazy_update_approx_solution_actions_eliminated_per_iteration = get<3>(V_AE_H_lazy_update_approx_solution_tuple);

	auto tick_accumulator_VIAEH_lazy_update = V_AE_H_lazy_update_approx_solution_work_per_iteration[0].count();

	for (int iteration = 1; iteration <= V_AE_H_lazy_update_approx_solution_iterations; iteration++)
	{
		auto iteration_work = V_AE_H_lazy_update_approx_solution_work_per_iteration[iteration].count();
		auto actions_eliminated_iteration = V_AE_H_lazy_update_approx_solution_actions_eliminated_per_iteration[iteration].size();
		tick_accumulator_VIAEH_lazy_update = tick_accumulator_VIAEH_lazy_update + iteration_work;

		stringstream_VIAEH_lazy_update << to_string(iteration) << " " << iteration_work << endl;
		stringstream_action_elimination_VIAEH_lazy_update << to_string(iteration) << " " << actions_eliminated_iteration << endl;
		stringstream_accum_VIAEH_lazy_update << to_string(iteration) << " " << tick_accumulator_VIAEH_lazy_update << endl;
	}

	// VIAEH_set
	printf("VIAEH_set\n");
	A_type A4 = copy_A(A);

	V_type V_AE_H_set_approx_solution_tuple = value_iteration_action_elimination_heaps_set(S, R, A4, P, gamma, epsilon);
	vector<double> V_AE_H_set_approx_solution = get<0>(V_AE_H_set_approx_solution_tuple);
	int V_AE_H_set_approx_solution_iterations = get<1>(V_AE_H_set_approx_solution_tuple);
	vector<microseconds> V_AE_H_set_approx_solution_work_per_iteration = get<2>(V_AE_H_set_approx_solution_tuple);
	vector<vector<pair<int, int>>> V_AE_H_set_approx_solution_actions_eliminated_per_iteration = get<3>(V_AE_H_set_approx_solution_tuple);

	auto tick_accumulator_VIAEH_set = V_AE_H_set_approx_solution_work_per_iteration[0].count();

	for (int iteration = 1; iteration <= V_AE_H_set_approx_solution_iterations; iteration++)
	{
		auto iteration_work = V_AE_H_set_approx_solution_work_per_iteration[iteration].count();
		auto actions_eliminated_iteration = V_AE_H_set_approx_solution_actions_eliminated_per_iteration[iteration].size();
		tick_accumulator_VIAEH_set = tick_accumulator_VIAEH_set + iteration_work;

		stringstream_VIAEH_set << to_string(iteration) << " " << iteration_work << endl;
		stringstream_action_elimination_VIAEH_set << to_string(iteration) << " " << actions_eliminated_iteration << endl;
		stringstream_accum_VIAEH_set << to_string(iteration) << " " << tick_accumulator_VIAEH_set << endl;
	}

	// VIAEH_maxmin_heap
	printf("VIAEH_maxmin_heap\n");
	A_type A6 = copy_A(A);

	V_type V_AE_H_maxmin_heap_approx_solution_tuple = value_iteration_action_elimination_heaps_max_min_heap(S, R, A6, P, gamma, epsilon);
	vector<double> V_AE_H_maxmin_heap_approx_solution = get<0>(V_AE_H_maxmin_heap_approx_solution_tuple);
	int V_AE_H_maxmin_heap_approx_solution_iterations = get<1>(V_AE_H_maxmin_heap_approx_solution_tuple);
	vector<microseconds> V_AE_H_maxmin_heap_approx_solution_work_per_iteration = get<2>(V_AE_H_maxmin_heap_approx_solution_tuple);
	vector<vector<pair<int, int>>> V_AE_H_maxmin_heap_approx_solution_actions_eliminated_per_iteration = get<3>(V_AE_H_maxmin_heap_approx_solution_tuple);

	auto tick_accumulator_VIAEH_maxmin_heap = V_AE_H_maxmin_heap_approx_solution_work_per_iteration[0].count();

	for (int iteration = 1; iteration <= V_AE_H_maxmin_heap_approx_solution_iterations; iteration++)
	{
		auto iteration_work = V_AE_H_maxmin_heap_approx_solution_work_per_iteration[iteration].count();
		auto actions_eliminated_iteration = V_AE_H_maxmin_heap_approx_solution_actions_eliminated_per_iteration[iteration].size();
		tick_accumulator_VIAEH_maxmin_heap = tick_accumulator_VIAEH_maxmin_heap + iteration_work;

		stringstream_VIAEH_maxmin_heap << to_string(iteration) << " " << iteration_work << endl;
		stringstream_action_elimination_VIAEH_maxmin_heap << to_string(iteration) << " " << actions_eliminated_iteration << endl;
		stringstream_accum_VIAEH_maxmin_heap << to_string(iteration) << " " << tick_accumulator_VIAEH_maxmin_heap << endl;
	}

	// VIAEH_approx_lower_bound
	printf("VIAEH_approx_lower_bound\n");
	A_type A7 = copy_A(A);

	V_type V_AE_H_approx_lower_bound_approx_solution_tuple = value_iteration_action_elimination_heaps_lower_bound_approxGS(S, R, A7, P, gamma, epsilon);
	vector<double> V_AE_H_approx_lower_bound_approx_solution = get<0>(V_AE_H_approx_lower_bound_approx_solution_tuple);
	int V_AE_H_approx_lower_bound_approx_solution_iterations = get<1>(V_AE_H_approx_lower_bound_approx_solution_tuple);
	vector<microseconds> V_AE_H_approx_lower_bound_approx_solution_work_per_iteration = get<2>(V_AE_H_approx_lower_bound_approx_solution_tuple);
	vector<vector<pair<int, int>>> V_AE_H_approx_lower_bound_approx_solution_actions_eliminated_per_iteration = get<3>(V_AE_H_approx_lower_bound_approx_solution_tuple);

	auto tick_accumulator_VIAEH_approx_lower_bound = V_AE_H_approx_lower_bound_approx_solution_work_per_iteration[0].count();

	for (int iteration = 1; iteration <= V_AE_H_approx_lower_bound_approx_solution_iterations; iteration++)
	{
		auto iteration_work = V_AE_H_approx_lower_bound_approx_solution_work_per_iteration[iteration].count();
		auto actions_eliminated_iteration = V_AE_H_approx_lower_bound_approx_solution_actions_eliminated_per_iteration[iteration].size();
		tick_accumulator_VIAEH_approx_lower_bound = tick_accumulator_VIAEH_approx_lower_bound + iteration_work;

		stringstream_VIAEH_approx_lower_bound << to_string(iteration) << " " << iteration_work << endl;
		stringstream_action_elimination_VIAEH_approx_lower_bound << to_string(iteration) << " " << actions_eliminated_iteration << endl;
		stringstream_accum_VIAEH_approx_lower_bound << to_string(iteration) << " " << tick_accumulator_VIAEH_approx_lower_bound << endl;
	}

	printf("Difference: %f\n", abs_max_diff_vectors(V_AE_H_lazy_update_approx_solution, V_AE_H_no_pointers_approx_solution));
	printf("Difference: %f\n", abs_max_diff_vectors(V_AE_H_lazy_update_approx_solution, V_AE_H_approx_solution));
	printf("Difference: %f\n", abs_max_diff_vectors(V_AE_H_set_approx_solution, V_AE_H_approx_solution));
	printf("Difference: %f\n", abs_max_diff_vectors(V_AE_approx_solution, V_AE_H_no_pointers_approx_solution));
	printf("Difference: %f\n", abs_max_diff_vectors(V_AE_H_maxmin_heap_approx_solution, V_AE_H_no_pointers_approx_solution));
	printf("Difference: %f\n", abs_max_diff_vectors(V_AE_H_approx_lower_bound_approx_solution, V_AE_H_no_pointers_approx_solution));

	// WRITE ALL DATA TO THEIR RESPECTVIE FILES
	write_stringstream_to_file(stringstream_VIAEH, output_stream_VIAEH, file_name_VIAEH);
	write_stringstream_to_file(stringstream_VIAEH_no_pointers, output_stream_VIAEH_no_pointers, file_name_VIAEH_no_pointers);
	write_stringstream_to_file(stringstream_VIAEH_lazy_update, output_stream_VIAEH_lazy_update, file_name_VIAEH_lazy_update);
	write_stringstream_to_file(stringstream_VIAEH_set, output_stream_VIAEH_set, file_name_VIAEH_set);
	write_stringstream_to_file(stringstream_VIAEH_maxmin_heap, output_stream_VIAEH_maxmin_heap, file_name_VIAEH_maxmin_heap);
	write_stringstream_to_file(stringstream_VIAEH_approx_lower_bound, output_stream_VIAEH_approx_lower_bound, file_name_VIAEH_approx_lower_bound);

	write_stringstream_to_file(stringstream_accum_VIAEH, output_stream_accum_VIAEH, file_name_accum_VIAEH);
	write_stringstream_to_file(stringstream_accum_VIAEH_no_pointers, output_stream_accum_VIAEH_no_pointers, file_name_accum_VIAEH_no_pointers);
	write_stringstream_to_file(stringstream_accum_VIAEH_lazy_update, output_stream_accum_VIAEH_lazy_update, file_name_accum_VIAEH_lazy_update);
	write_stringstream_to_file(stringstream_accum_VIAEH_set, output_stream_accum_VIAEH_set, file_name_accum_VIAEH_set);
	write_stringstream_to_file(stringstream_accum_VIAEH_maxmin_heap, output_stream_accum_VIAEH_maxmin_heap, file_name_accum_VIAEH_maxmin_heap);
	write_stringstream_to_file(stringstream_accum_VIAEH_approx_lower_bound, output_stream_accum_VIAEH_approx_lower_bound, file_name_accum_VIAEH_approx_lower_bound);

	write_stringstream_to_file(stringstream_action_elimination_VIAE, output_stream_action_elimination_VIAE, file_name_action_elimination_VIAE);
	write_stringstream_to_file(stringstream_action_elimination_VIAEH, output_stream_action_elimination_VIAEH, file_name_action_elimination_VIAEH);
	write_stringstream_to_file(stringstream_action_elimination_VIAEH_no_pointers, output_stream_action_elimination_VIAEH_no_pointers, file_name_action_elimination_VIAEH_no_pointers);
	write_stringstream_to_file(stringstream_action_elimination_VIAEH_lazy_update, output_stream_action_elimination_VIAEH_lazy_update, file_name_action_elimination_VIAEH_lazy_update);
	write_stringstream_to_file(stringstream_action_elimination_VIAEH_set, output_stream_action_elimination_VIAEH_set, file_name_action_elimination_VIAEH_set);
	write_stringstream_to_file(stringstream_action_elimination_VIAEH_maxmin_heap, output_stream_action_elimination_VIAEH_maxmin_heap, file_name_action_elimination_VIAEH_maxmin_heap);
	write_stringstream_to_file(stringstream_action_elimination_VIAEH_approx_lower_bound, output_stream_action_elimination_VIAEH_approx_lower_bound, file_name_action_elimination_VIAEH_approx_lower_bound);
}

void create_data_tables_bounds_comparisons(string filename, int S, int A_num, double epsilon, double gamma, double action_prob, int number_of_transitions, double mean, double variance)
{

	// FOR WORK PER ITERATION
	// the stringstreams to create the test for the files
	ostringstream stringstream_VIAE_improved_bounds;
	ostringstream stringstream_VIAE_old_bounds;

	// the file output objects
	ofstream output_stream_VIAE_improved_bounds;
	ofstream output_stream_VIAE_old_bounds;

	// set the name of the file to write to
	string file_name_VIAE_improved_bounds = "data_tables/bounds_comparisons/" + filename + "_VIAE_improved_bounds.dat";
	string file_name_VIAE_old_bounds = "data_tables/bounds_comparisons/" + filename + "_VIAE_old_bounds.dat";

	// write meta data to all stringstreams as first in their respective files
	// write_meta_data_to_dat_file_work_per_iteration(stringstream_VIAE_improved_bounds, S, A_num, epsilon, gamma, non_zero_transition, upper_reward, action_prob);

	// FOR ACCUMULATED WORK
	// the stringstreams to create the test for the files
	ostringstream stringstream_accum_VIAE_improved_bounds;
	ostringstream stringstream_accum_VIAE_old_bounds;

	// the file output objects
	ofstream output_stream_accum_VIAE_improved_bounds;
	ofstream output_stream_accum_VIAE_old_bounds;

	// set the name of the file to write to
	string file_name_accum_VIAE_improved_bounds = "data_tables/bounds_comparisons/" + filename + "_accum_VIAE_improved_bounds.dat";
	string file_name_accum_VIAE_old_bounds = "data_tables/bounds_comparisons/" + filename + "_accum_VIAE_old_bounds.dat";

	// FOR ACTION ELIMINATION
	// the stringstreams to create the test for the files
	ostringstream stringstream_action_elimination_VIAE_improved_bounds;
	ostringstream stringstream_action_elimination_VIAE_old_bounds;

	// the file output objects
	ofstream output_stream_action_elimination_VIAE_improved_bounds;
	ofstream output_stream_action_elimination_VIAE_old_bounds;

	// set the name of the file to write to
	string file_name_action_elimination_VIAE_improved_bounds = "data_tables/bounds_comparisons/" + filename + "_action_elimination_VIAE_improved_bounds.dat";
	string file_name_action_elimination_VIAE_old_bounds = "data_tables/bounds_comparisons/" + filename + "_action_elimination_VIAE_old_bounds.dat";

	// write meta data to all stringstreams as first in their respective files
	// write_meta_data_to_dat_file_work_per_iteration(stringstream_accum_VIAEH, S, A_num, epsilon, gamma, non_zero_transition, upper_reward, action_prob);

	// BEGIN EXPERIMENTATION
	// GENERATE THE MDP FROM CURRENT TIME SEED
	int seed = time(0);
	printf("seed: %d\n", seed);

	// auto MDP = generate_random_MDP_with_variable_parameters_fixed_nonzero_trans_states(S, A_num, action_prob, number_of_transitions, 100.0, seed);
	// auto MDP = generate_random_MDP_normal_distributed_rewards(S, A_num, action_prob, number_of_transitions, seed, mean, variance);
	auto MDP = generate_random_MDP_exponential_distributed_rewards(S, A_num, action_prob, number_of_transitions, 0.02, seed);
	R_type R = get<0>(MDP);
	A_type A = get<1>(MDP);
	P_type P = get<2>(MDP);

	// VIAE_improved_bounds
	printf("VIAE_improved_bounds\n");
	A_type A1 = copy_A(A);

	V_type VIAE_improved_bounds_solution_tuple = value_iteration_action_elimination_improved_bounds(S, R, A1, P, gamma, epsilon);
	vector<double> VIAE_improved_bounds_solution = get<0>(VIAE_improved_bounds_solution_tuple);
	int VIAE_improved_bounds_solution_iterations = get<1>(VIAE_improved_bounds_solution_tuple);
	vector<microseconds> VIAE_improved_bounds_solution_work_per_iteration = get<2>(VIAE_improved_bounds_solution_tuple);
	vector<vector<pair<int, int>>> VIAE_improved_bounds_solution_actions_eliminated_per_iteration = get<3>(VIAE_improved_bounds_solution_tuple);

	auto tick_accumulator_VIAE_improved_bounds = VIAE_improved_bounds_solution_work_per_iteration[0].count();

	for (int iteration = 1; iteration <= VIAE_improved_bounds_solution_iterations; iteration++)
	{
		auto iteration_work = VIAE_improved_bounds_solution_work_per_iteration[iteration].count();
		auto actions_eliminated_iteration = VIAE_improved_bounds_solution_actions_eliminated_per_iteration[iteration].size();
		tick_accumulator_VIAE_improved_bounds = tick_accumulator_VIAE_improved_bounds + iteration_work;

		stringstream_VIAE_improved_bounds << to_string(iteration) << " " << iteration_work << endl;
		stringstream_action_elimination_VIAE_improved_bounds << to_string(iteration) << " " << actions_eliminated_iteration << endl;
		stringstream_accum_VIAE_improved_bounds << to_string(iteration) << " " << tick_accumulator_VIAE_improved_bounds << endl;
	}

	// VIAE_old_bounds
	printf("VIAE_old_bounds\n");
	A_type A2 = copy_A(A);

	V_type VIAE_old_bounds_solution_tuple = value_iteration_action_elimination_old_bounds(S, R, A2, P, gamma, epsilon);
	vector<double> VIAE_old_bounds_solution = get<0>(VIAE_old_bounds_solution_tuple);
	int VIAE_old_bounds_solution_iterations = get<1>(VIAE_old_bounds_solution_tuple);
	vector<microseconds> VIAE_old_bounds_solution_work_per_iteration = get<2>(VIAE_old_bounds_solution_tuple);
	vector<vector<pair<int, int>>> VIAE_old_bounds_solution_actions_eliminated_per_iteration = get<3>(VIAE_old_bounds_solution_tuple);

	auto tick_accumulator_VIAE_old_bounds = VIAE_old_bounds_solution_work_per_iteration[0].count();

	for (int iteration = 1; iteration <= VIAE_old_bounds_solution_iterations; iteration++)
	{
		auto iteration_work = VIAE_old_bounds_solution_work_per_iteration[iteration].count();
		auto actions_eliminated_iteration = VIAE_old_bounds_solution_actions_eliminated_per_iteration[iteration].size();
		tick_accumulator_VIAE_old_bounds = tick_accumulator_VIAE_old_bounds + iteration_work;

		stringstream_VIAE_old_bounds << to_string(iteration) << " " << iteration_work << endl;
		stringstream_action_elimination_VIAE_old_bounds << to_string(iteration) << " " << actions_eliminated_iteration << endl;
		stringstream_accum_VIAE_old_bounds << to_string(iteration) << " " << tick_accumulator_VIAE_old_bounds << endl;
	}
	printf("Solutions:\n");
	for (int s = 0; s < 5; s++)
	{
		printf("state %d: %f\n", s, VIAE_improved_bounds_solution[s]);
		printf("state %d: %f\n", s, VIAE_old_bounds_solution[s]);
	}

	printf("Difference: %f\n", abs_max_diff_vectors(VIAE_improved_bounds_solution, VIAE_old_bounds_solution));

	// WRITE ALL DATA TO THEIR RESPECTVIE FILES
	write_stringstream_to_file(stringstream_VIAE_improved_bounds, output_stream_VIAE_improved_bounds, file_name_VIAE_improved_bounds);
	write_stringstream_to_file(stringstream_VIAE_old_bounds, output_stream_VIAE_old_bounds, file_name_VIAE_old_bounds);

	write_stringstream_to_file(stringstream_accum_VIAE_improved_bounds, output_stream_accum_VIAE_improved_bounds, file_name_accum_VIAE_improved_bounds);
	write_stringstream_to_file(stringstream_accum_VIAE_old_bounds, output_stream_accum_VIAE_old_bounds, file_name_accum_VIAE_old_bounds);

	write_stringstream_to_file(stringstream_action_elimination_VIAE_improved_bounds, output_stream_action_elimination_VIAE_improved_bounds, file_name_action_elimination_VIAE_improved_bounds);
	write_stringstream_to_file(stringstream_action_elimination_VIAE_old_bounds, output_stream_action_elimination_VIAE_old_bounds, file_name_action_elimination_VIAE_old_bounds);
}

// ACTIONS TOUCHED VS ELIMINATED
void create_data_tables_actions_touched(string filename, int S, int A_num, double epsilon, double gamma, double action_prob, int number_of_transitions, double mean, double variance)
{

	// FOR ACTION ELIMINATION
	// the stringstreams to create the test for the files
	ostringstream stringstream_action_elimination_VIAEH;
	ostringstream stringstream_action_elimination_accum_VIAEH;
	ostringstream stringstream_implicit_action_elimination_VIH_actions_touched;
	ostringstream stringstream_implicit_action_elimination_accum_VIH_actions_touched;

	ostringstream stringstream_action_elimination_VIH_actions_touched;
	ostringstream stringstream_actions_touched_after_elimination;

	// the file output objects
	ofstream output_stream_action_elimination_VIAEH;
	ofstream output_stream_action_elimination_accum_VIAEH;
	ofstream output_stream_implicit_action_elimination_VIH_actions_touched;
	ofstream output_stream_implicit_action_elimination_accum_VIH_actions_touched;

	ofstream output_stream_action_elimination_VIH_actions_touched;
	ofstream output_stream_actions_touched_after_elimination;

	// set the name of the file to write to
	string file_name_action_elimination_VIAEH = "data_tables/actions_touched/" + filename + "_action_elimination_VIAEH.dat";
	string file_name_action_elimination_accum_VIAEH = "data_tables/actions_touched/" + filename + "_action_elimination_accum_VIAEH.dat";
	string file_name_implicit_action_elimination_VIH_actions_touched = "data_tables/actions_touched/" + filename + "_implicit_action_elimination_VIH.dat";
	string file_name_implicit_action_elimination_accum_VIH_actions_touched = "data_tables/actions_touched/" + filename + "_implicit_action_elimination_accum_VIH.dat";

	string file_name_action_elimination_VIH_actions_touched = "data_tables/actions_touched/" + filename + "_action_elimination_VIH_actions_touched.dat";
	string file_name_actions_touched_after_elimination = "data_tables/actions_touched/" + filename + "_actions_touched_after_elimination.dat";

	// BEGIN EXPERIMENTATION
	// GENERATE THE MDP FROM CURRENT TIME SEED
	int seed = time(0);
	printf("seed: %d\n", seed);

	// TODO permament change to normal distribution here?
	auto MDP = generate_random_MDP_normal_distributed_rewards(S, A_num, action_prob, number_of_transitions, seed, mean, variance);
	// auto MDP = generate_random_MDP_exponential_distributed_rewards(S, A_num, action_prob, number_of_transitions, 0.02, seed);
	R_type R = get<0>(MDP);
	A_type A = get<1>(MDP);
	P_type P = get<2>(MDP);

	// VIAEH
	printf("VIAEH\n");
	A_type A1 = copy_A(A);

	// V_type V_AE_H_approx_solution_tuple = value_iteration_action_elimination_heapsGS(S, R, A1, P, gamma, epsilon);
	V_type V_AE_H_approx_solution_tuple = value_iteration_action_elimination_heaps_no_pointersGS(S, R, A1, P, gamma, epsilon);
	printf("VIAEH\n");
	vector<double> V_AE_H_approx_solution = get<0>(V_AE_H_approx_solution_tuple);
	printf("VIAEH\n");
	int V_AE_H_approx_solution_iterations = get<1>(V_AE_H_approx_solution_tuple);
	printf("VIAEH\n");
	// THIS IS per iteration, we have (state, action) pairs
	vector<vector<pair<int, int>>> V_AE_H_approx_solution_actions_eliminated_per_iteration = get<3>(V_AE_H_approx_solution_tuple);
	printf("VIAEH\n");
	// for(int iteration = 1; iteration <= V_AE_H_approx_solution_iterations; iteration++) {
	//		auto iteration_work = V_AE_H_approx_solution_work_per_iteration[iteration].count();
	//		auto actions_eliminated_iteration = V_AE_H_approx_solution_actions_eliminated_per_iteration[iteration].size();
	//		tick_accumulator_VIAEH = tick_accumulator_VIAEH + iteration_work;

	//		stringstream_action_elimination_VIAEH << to_string(iteration) << " " << actions_eliminated_iteration << endl;
	//}

	// VIAE
	printf("VIH actions touched\n");
	A_type A5 = copy_A(A);

	V_type VIH_actions_touched_approx_solution_tuple = value_iteration_actions_touchedGS(S, R, A5, P, gamma, epsilon);
	vector<double> VIH_actions_touched_approx_solution = get<0>(VIH_actions_touched_approx_solution_tuple);
	int VIH_actions_touched_approx_solution_iterations = get<1>(VIH_actions_touched_approx_solution_tuple);

	// THIS IS per iteration, we have (state, action) pairs
	vector<vector<pair<int, int>>> VIH_actions_touched_approx_solution_actions_eliminated_per_iteration = get<3>(VIH_actions_touched_approx_solution_tuple);

	// RECORD ACTIONS TOUCHED IN EACH ITERATION
	for (int iteration = 1; iteration <= VIH_actions_touched_approx_solution_iterations; iteration++)
	{

		// this is number of actions touched in this iteration - to compare with number of saved per iteration per state, just to set it into perspective
		auto actions_touched_iteration = VIH_actions_touched_approx_solution_actions_eliminated_per_iteration[iteration].size();

		stringstream_action_elimination_VIH_actions_touched << to_string(iteration) << " " << (float(actions_touched_iteration) / float(S)) << endl;
	}

	int max_iteration = max(VIH_actions_touched_approx_solution_iterations, V_AE_H_approx_solution_iterations);
	int min_iteration = min(VIH_actions_touched_approx_solution_iterations, V_AE_H_approx_solution_iterations);

	// WE NOW WANT A PER STATE VIEW OF EACH
	vector<vector<pair<int, int>>> V_AE_H_approx_solution_actions_eliminated_per_state(S);
	for (int iteration = 1; iteration <= V_AE_H_approx_solution_iterations; iteration++)
	{

		// We iterate through the (s,a) list in each iteration
		for (auto pair : V_AE_H_approx_solution_actions_eliminated_per_iteration[iteration])
		{

			// We add (iterations, action)
			V_AE_H_approx_solution_actions_eliminated_per_state[pair.first].push_back(make_pair(iteration, pair.second));
		}
	}

	// WE NOW WANT A PER STATE VIEW OF EACH ACTION TOUCED
	vector<vector<pair<int, int>>> VIH_actions_touched_approx_solution_actions_eliminated_per_state(S);
	for (int iteration = 1; iteration <= VIH_actions_touched_approx_solution_iterations; iteration++)
	{

		// We iterate through the (s,a) list in each iteration
		for (auto pair : VIH_actions_touched_approx_solution_actions_eliminated_per_iteration[iteration])
		{

			// We add (iterations, action)
			VIH_actions_touched_approx_solution_actions_eliminated_per_state[pair.first].push_back(make_pair(iteration, pair.second));
		}
	}

	// WE NOW LOOK AT EACH STATE AND MATCH ELIMINATION ITERATION WITH ITERATIONS TOUCHED AFTER THAT
	// consists of (state, action) pairs
	vector<vector<pair<int, int>>> action_touched_after_elimination_per_iteration(max_iteration + 1);
	for (int s = 0; s < S; s++)
	{
		vector<pair<int, int>> VIAEH_per_state = V_AE_H_approx_solution_actions_eliminated_per_state[s];
		vector<pair<int, int>> VIH_per_state = VIH_actions_touched_approx_solution_actions_eliminated_per_state[s];

		for (auto pair_VIAEH : VIAEH_per_state)
		{
			for (auto pair_VIH : VIH_per_state)
			{

				// check if action is the same AND elimination iteration is before touch iteration
				if ((pair_VIAEH.second == pair_VIH.second) && (pair_VIAEH.first < pair_VIH.first))
				{
					action_touched_after_elimination_per_iteration[pair_VIH.first].push_back(pair_VIH);
				}
			}
		}
	}

	for (int iteration = 1; iteration <= max_iteration; iteration++)
	{
		auto number_to_write_to_file = action_touched_after_elimination_per_iteration[iteration].size();
		stringstream_actions_touched_after_elimination << to_string(iteration) << " " << (float(number_to_write_to_file) / float(S)) << endl;
	}

	// PREPARE FOR IMPLICIT- VS EXPLICIT ACTION ELIMINATION NUMBER OF ACTION "ELIMINTED" (last iteration where it is never again considered) PER ITERATION
	// AND ACCUMULATION PLOT

	// recording actions eliminated through VIAEH
	int actions_eliminated_accumulator_VIAEH = 0;
	for (int iteration = 1; iteration <= V_AE_H_approx_solution_iterations; iteration++)
	{
		auto actions_eliminated_iteration = int(V_AE_H_approx_solution_actions_eliminated_per_iteration[iteration].size());
		actions_eliminated_accumulator_VIAEH = actions_eliminated_accumulator_VIAEH + actions_eliminated_iteration;

		stringstream_action_elimination_VIAEH << to_string(iteration) << " " << actions_eliminated_iteration << endl;
		stringstream_action_elimination_accum_VIAEH << to_string(iteration) << " " << actions_eliminated_accumulator_VIAEH << endl;
	}

	// recording actions eliminated "implicitly" through VIH

	vector<vector<pair<int, int>>> VIH_actions_touched_actions_implicitly_eliminated_per_iteration(VIH_actions_touched_approx_solution_iterations + 1);

	// record if action a in state s is touched in last iteration! (S, A(s)) matrix
	//-1 = not seen yet, so record iteartion number when seen
	//-2 = in last iteration, dont consider when going backwards in iterations
	vector<vector<int>> touched_in_last_iteration;
	int A_max = find_max_A(A) + 1;
	for (int s = 0; s < S; s++)
	{
		// record not yet seen
		vector<int> for_As_iteration(A_max, -1);
		touched_in_last_iteration.push_back(for_As_iteration);
	}

	for (auto pair : action_touched_after_elimination_per_iteration[VIH_actions_touched_approx_solution_iterations])
	{
		// record that this (s,a) is in the last iteration, such that it is NEVER eliminated
		touched_in_last_iteration[pair.first][pair.second] = -2;
	}

	for (int iteration = VIH_actions_touched_approx_solution_iterations - 1; iteration >= 1; iteration--)
	{
		for (auto pair : VIH_actions_touched_approx_solution_actions_eliminated_per_iteration[iteration])
		{
			if (touched_in_last_iteration[pair.first][pair.second] == -1)
			{
				touched_in_last_iteration[pair.first][pair.second] = iteration;
				VIH_actions_touched_actions_implicitly_eliminated_per_iteration[iteration].push_back(pair);
			}
		}
	}

	int actions_eliminated_accumulator_VIH_actions_touched = 0;
	for (int iteration = 1; iteration <= VIH_actions_touched_approx_solution_iterations; iteration++)
	{
		auto actions_eliminated_iteration = int(VIH_actions_touched_actions_implicitly_eliminated_per_iteration[iteration].size());
		actions_eliminated_accumulator_VIH_actions_touched = actions_eliminated_accumulator_VIH_actions_touched + actions_eliminated_iteration;

		stringstream_implicit_action_elimination_VIH_actions_touched << to_string(iteration) << " " << actions_eliminated_iteration << endl;
		stringstream_implicit_action_elimination_accum_VIH_actions_touched << to_string(iteration) << " " << actions_eliminated_accumulator_VIH_actions_touched << endl;
	}

	printf("Difference: %f\n", abs_max_diff_vectors(V_AE_H_approx_solution, VIH_actions_touched_approx_solution));

	// WRITE ALL DATA TO THEIR RESPECTVIE FILES
	write_stringstream_to_file(stringstream_action_elimination_VIAEH, output_stream_action_elimination_VIAEH, file_name_action_elimination_VIAEH);
	write_stringstream_to_file(stringstream_action_elimination_accum_VIAEH, output_stream_action_elimination_accum_VIAEH, file_name_action_elimination_accum_VIAEH);
	write_stringstream_to_file(stringstream_implicit_action_elimination_VIH_actions_touched, output_stream_implicit_action_elimination_VIH_actions_touched, file_name_implicit_action_elimination_VIH_actions_touched);
	write_stringstream_to_file(stringstream_implicit_action_elimination_accum_VIH_actions_touched, output_stream_implicit_action_elimination_accum_VIH_actions_touched, file_name_implicit_action_elimination_accum_VIH_actions_touched);

	write_stringstream_to_file(stringstream_action_elimination_VIH_actions_touched, output_stream_action_elimination_VIH_actions_touched, file_name_action_elimination_VIH_actions_touched);
	write_stringstream_to_file(stringstream_actions_touched_after_elimination, output_stream_actions_touched_after_elimination, file_name_actions_touched_after_elimination);
}

// WORK PER ITERATION BEST ALGORITHMS ()
void create_data_tables_work_per_iteration_BEST_implementations(string filename, int S, int A_num, double epsilon, double gamma, double action_prob, int non_zero_transition, double mean, double variance)
{

	// FOR WORK PER ITERATION
	// the stringstreams to create the test for the files
	ostringstream stringstream_BAO;
	ostringstream stringstream_VIH;
	ostringstream stringstream_VIAEHL;

	// the file output objects
	ofstream output_stream_BAO;
	ofstream output_stream_VIH;
	ofstream output_stream_VIAEHL;

	// set the name of the file to write to
	string file_name_BAO = "data_tables/work_per_iteration_BEST_implementations/" + filename + "_BAO.dat";
	string file_name_VIH = "data_tables/work_per_iteration_BEST_implementations/" + filename + "_VIH.dat";
	string file_name_VIAEHL = "data_tables/work_per_iteration_BEST_implementations/" + filename + "_VIAEHL.dat";

	// write meta data to all stringstreams as first in their respective files
	// write_meta_data_to_dat_file_work_per_iteration_BEST_implementations(stringstream_BAO, S, A_num, epsilon, gamma, non_zero_transition, action_prob);
	// write_meta_data_to_dat_file_work_per_iteration_BEST_implementations(stringstream_VIH, S, A_num, epsilon, gamma, non_zero_transition, action_prob);
	// write_meta_data_to_dat_file_work_per_iteration_BEST_implementations(stringstream_VIAEHL, S, A_num, epsilon, gamma, non_zero_transition, action_prob);

	// FOR ACCUMULATED WORK
	// the stringstreams to create the test for the files
	ostringstream stringstream_accum_BAO;
	ostringstream stringstream_accum_VIH;
	ostringstream stringstream_accum_VIAEHL;

	// the file output objects
	ofstream output_stream_accum_BAO;
	ofstream output_stream_accum_VIH;
	ofstream output_stream_accum_VIAEHL;

	// set the name of the file to write to
	string file_name_accum_BAO = "data_tables/work_per_iteration_BEST_implementations/" + filename + "_accum_BAO.dat";
	string file_name_accum_VIH = "data_tables/work_per_iteration_BEST_implementations/" + filename + "_accum_VIH.dat";
	string file_name_accum_VIAEHL = "data_tables/work_per_iteration_BEST_implementations/" + filename + "_accum_VIAEHL.dat";

	// write meta data to all stringstreams as first in their respective files
	// write_meta_data_to_dat_file_work_per_iteration(stringstream_accum_BAO, S, A_num, epsilon, gamma, non_zero_transition, action_prob);
	// write_meta_data_to_dat_file_work_per_iteration(stringstream_accum_VIH, S, A_num, epsilon, gamma, non_zero_transition, action_prob);
	// write_meta_data_to_dat_file_work_per_iteration(stringstream_accum_VIAEHL, S, A_num, epsilon, gamma, non_zero_transition, action_prob);

	// BEGIN EXPERIMENTATION
	// GENERATE THE MDP FROM CURRENT TIME SEED
	int seed = time(0);
	printf("seed: %d\n", seed);

	// TODO permament change to normal distribution here?
	// auto MDP = generate_random_MDP_normal_distributed_rewards(S, A_num, action_prob, non_zero_transition, seed, mean, variance);
	auto MDP = generate_random_MDP_exponential_distributed_rewards(S, A_num, action_prob, non_zero_transition, 0.02, seed);
	// auto MDP = generate_random_MDP_with_variable_parameters(S, A_num, action_prob, non_zero_transition, upper_reward, seed);
	R_type R = get<0>(MDP);
	A_type A = get<1>(MDP);
	P_type P = get<2>(MDP);

	// VIH
	printf("VIH\n");
	A_type A2 = copy_A(A);

	V_type V_heap_approx_tuple = value_iteration_with_heap(S, R, A2, P, gamma, epsilon);
	vector<double> V_heap_approx = get<0>(V_heap_approx_tuple);
	int V_heap_approx_iterations = get<1>(V_heap_approx_tuple);
	vector<microseconds> V_heap_approx_work_per_iteration = get<2>(V_heap_approx_tuple);

	auto tick_accumulator_VIH = V_heap_approx_work_per_iteration[0].count();

	for (int iteration = 1; iteration <= V_heap_approx_iterations; iteration++)
	{
		auto iteration_work = V_heap_approx_work_per_iteration[iteration].count();
		tick_accumulator_VIH = tick_accumulator_VIH + iteration_work;

		stringstream_VIH << to_string(iteration) << " " << iteration_work << endl;
		stringstream_accum_VIH << to_string(iteration) << " " << tick_accumulator_VIH << endl;
	}

	// BAO
	printf("BAO\n");
	A_type A3 = copy_A(A);

	V_type BAO_approx_solution_tuple = value_iteration_BAO(S, R, A3, P, gamma, epsilon);
	vector<double> BAO_approx_solution = get<0>(BAO_approx_solution_tuple);
	int BAO_approx_solution_iterations = get<1>(BAO_approx_solution_tuple);
	vector<microseconds> BAO_approx_solution_work_per_iteration = get<2>(BAO_approx_solution_tuple);

	auto tick_accumulator_BAO = BAO_approx_solution_work_per_iteration[0].count();

	for (int iteration = 1; iteration <= BAO_approx_solution_iterations; iteration++)
	{
		auto iteration_work = BAO_approx_solution_work_per_iteration[iteration].count();
		tick_accumulator_BAO = tick_accumulator_BAO + iteration_work;

		stringstream_BAO << to_string(iteration) << " " << iteration_work << endl;
		stringstream_accum_BAO << to_string(iteration) << " " << tick_accumulator_BAO << endl;
	}

	// VIAEHL
	printf("VIAEHL\n");
	A_type A7 = copy_A(A);

	V_type VIAEHL_approx_solution_tuple = value_iteration_action_elimination_heaps_lower_bound_approx(S, R, A7, P, gamma, epsilon);
	vector<double> VIAEHL_approx_solution = get<0>(VIAEHL_approx_solution_tuple);
	int VIAEHL_approx_solution_iterations = get<1>(VIAEHL_approx_solution_tuple);
	vector<microseconds> VIAEHL_approx_solution_work_per_iteration = get<2>(VIAEHL_approx_solution_tuple);

	auto tick_accumulator_VIAEHL = VIAEHL_approx_solution_work_per_iteration[0].count();

	for (int iteration = 1; iteration <= VIAEHL_approx_solution_iterations; iteration++)
	{
		auto iteration_work = VIAEHL_approx_solution_work_per_iteration[iteration].count();
		tick_accumulator_VIAEHL = tick_accumulator_VIAEHL + iteration_work;

		stringstream_VIAEHL << to_string(iteration) << " " << iteration_work << endl;
		stringstream_accum_VIAEHL << to_string(iteration) << " " << tick_accumulator_VIAEHL << endl;
	}

	// They should in theory all be epsilon from true value, and therefore, at most be 2 * epsilon from each other
	printf("Difference: %f\n", abs_max_diff_vectors(V_heap_approx, BAO_approx_solution));
	printf("Difference: %f\n", abs_max_diff_vectors(V_heap_approx, VIAEHL_approx_solution));
	printf("Difference: %f\n", abs_max_diff_vectors(BAO_approx_solution, VIAEHL_approx_solution));

	// WRITE ALL DATA TO THEIR RESPECTVIE FILES
	write_stringstream_to_file(stringstream_BAO, output_stream_BAO, file_name_BAO);
	write_stringstream_to_file(stringstream_VIH, output_stream_VIH, file_name_VIH);
	write_stringstream_to_file(stringstream_VIAEHL, output_stream_VIAEHL, file_name_VIAEHL);

	write_stringstream_to_file(stringstream_accum_BAO, output_stream_accum_BAO, file_name_accum_BAO);
	write_stringstream_to_file(stringstream_accum_VIH, output_stream_accum_VIH, file_name_accum_VIH);
	write_stringstream_to_file(stringstream_accum_VIAEHL, output_stream_accum_VIAEHL, file_name_accum_VIAEHL);
}

void create_data_tables_number_of_actions_best_implementations(string filename, int S, int A_max, double epsilon, double gamma, double upper_reward, double non_zero_transition)
{

	// the stringstreams to create the test for the files
	ostringstream stringstream_VIH;
	ostringstream stringstream_VIAEHL;
	ostringstream stringstream_BAO;

	// the file output objects
	ofstream output_stream_VIH;
	ofstream output_stream_VIAEHL;
	ofstream output_stream_BAO;

	// set the name of the file to write to
	string file_name_VIH = "data_tables/number_of_actions_best/" + filename + "_VIH.dat";
	string file_name_VIAEHL = "data_tables/number_of_actions_best/" + filename + "_VIAEHL.dat";
	string file_name_BAO = "data_tables/number_of_actions_best/" + filename + "_BAO.dat";

	// The varying parameters
	int A_starting_value = 50;
	int A_finishing_value = A_max;
	int A_increment = 50;

	// hardcoded parameter
	double action_prob = 1.0;

	for (int A_num = A_starting_value; A_num <= A_finishing_value; A_num = A_num + A_increment)
	{

		printf("Beginning iteration A_num = %d\n", A_num);

		// auto MDP = generate_random_MDP_normal_distributed_rewards(S, A_num, action_prob, non_zero_transition, seed, 1000, 10);
		// auto MDP = generate_random_MDP_exponential_distributed_rewards(S, A_num, action_prob, non_zero_transition, 0.02, seed);
		// auto MDP = generate_random_MDP_with_variable_parameters(S, A_num, action_prob, non_zero_transition, 1000, seed);
		// GENERATE THE MDP
		int seed = time(0);
		// auto MDP = generate_random_MDP_exponential_distributed_rewards(S, A_num, 1.0, non_zero_transition, 0.02, seed);
		auto MDP = generate_random_MDP_normal_distributed_rewards(S, A_num, 1.0, S, seed, 1000, 10);
		// auto MDP = generate_random_MDP_with_variable_parameters(S, A_num, 1.0, non_zero_transition, upper_reward, seed);
		R_type R = get<0>(MDP);
		A_type A = get<1>(MDP);
		P_type P = get<2>(MDP);

		// VIH testing
		A_type A2 = copy_A(A);
		auto start_VIH = high_resolution_clock::now();

		V_type V_heap_approx_tuple = value_iteration_with_heap(S, R, A2, P, gamma, epsilon);
		vector<double> V_heap_approx = get<0>(V_heap_approx_tuple);

		auto stop_VIH = high_resolution_clock::now();
		auto duration_VIH = duration_cast<microseconds>(stop_VIH - start_VIH);

		stringstream_VIH << to_string(A_num) << " " << duration_VIH.count() << endl;

		// VIAEHL
		A_type A9 = copy_A(A);
		auto start_VIAEHL = high_resolution_clock::now();

		V_type VIAEHL_approx_solution_tuple = value_iteration_action_elimination_heaps_lower_bound_approx(S, R, A9, P, gamma, epsilon);
		vector<double> VIAEHL_approx_solution = get<0>(VIAEHL_approx_solution_tuple);

		auto stop_VIAEHL = high_resolution_clock::now();
		auto duration_VIAEHL = duration_cast<microseconds>(stop_VIAEHL - start_VIAEHL);

		stringstream_VIAEHL << to_string(A_num) << " " << duration_VIAEHL.count() << endl;

		// BAO
		A_type A8 = copy_A(A);
		auto start_BAO = high_resolution_clock::now();

		V_type BAO_approx_solution_tuple = value_iteration_BAO(S, R, A8, P, gamma, epsilon);
		vector<double> BAO_approx_solution = get<0>(BAO_approx_solution_tuple);

		auto stop_BAO = high_resolution_clock::now();
		auto duration_BAO = duration_cast<microseconds>(stop_BAO - start_BAO);

		stringstream_BAO << to_string(A_num) << " " << duration_BAO.count() << endl;

		A_type A3 = copy_A(A);
		start_BAO = high_resolution_clock::now();

		BAO_approx_solution_tuple = value_iteration_BAOSK(S, R, A3, P, gamma, epsilon);

		stop_BAO = high_resolution_clock::now();
		duration_BAO = duration_cast<microseconds>(stop_BAO - start_BAO);

		stringstream_BAO << to_string(A_num) << " " << duration_BAO.count() << endl;

		// They should in theory all be epsilon from true value, and therefore, at most be 2 * epsilon from each other
		if (abs_max_diff_vectors(VIAEHL_approx_solution, V_heap_approx) > (2 * epsilon))
		{
			printf("DIFFERENCE\n");
		}
		if (abs_max_diff_vectors(VIAEHL_approx_solution, BAO_approx_solution) > (2 * epsilon))
		{
			printf("DIFFERENCE\n");
		}
	}

	// WRITE ALL DATA TO THEIR RESPECTVIE FILES
	write_stringstream_to_file(stringstream_VIH, output_stream_VIH, file_name_VIH);
	write_stringstream_to_file(stringstream_VIAEHL, output_stream_VIAEHL, file_name_VIAEHL);
	write_stringstream_to_file(stringstream_BAO, output_stream_BAO, file_name_BAO);
}

void create_data_tables_number_of_states_best_implementations(string filename, int S_max, int A_num, double epsilon, double gamma, double upper_reward, double non_zero_transition)
{

	// the stringstreams to create the test for the files
	ostringstream stringstream_VIH;
	ostringstream stringstream_VIAEHL;
	ostringstream stringstream_BAO;

	// the file output objects
	ofstream output_stream_VIH;
	ofstream output_stream_VIAEHL;
	ofstream output_stream_BAO;

	// set the name of the file to write to
	string file_name_VIH = "data_tables/number_of_states_best/" + filename + "_VIH.dat";
	string file_name_VIAEHL = "data_tables/number_of_states_best/" + filename + "_VIAEHL.dat";
	string file_name_BAO = "data_tables/number_of_states_best/" + filename + "_BAO.dat";

	// The varying parameters
	int S_starting_value = 100;
	int S_finishing_value = S_max;
	int S_increment = 100;

	// hardcoded parameter
	double action_prob = 1.0;

	// write meta data to all stringstreams as first in their respective files
	for (int S = S_starting_value; S <= S_finishing_value; S = S + S_increment)
	{

		printf("Beginning iteration S = %d\n", S);

		// GENERATE THE MDP
		int seed = time(0);
		auto MDP = generate_random_MDP_normal_distributed_rewards(S, A_num, action_prob, S, seed, 1000, 10);
		// auto MDP = generate_random_MDP_with_variable_parameters(S, A_num, action_prob, non_zero_transition, upper_reward, seed);
		R_type R = get<0>(MDP);
		A_type A = get<1>(MDP);
		P_type P = get<2>(MDP);
		// VIH testing
		A_type A2 = copy_A(A);
		auto start_VIH = high_resolution_clock::now();

		V_type V_heap_approx_tuple = value_iteration_with_heap(S, R, A2, P, gamma, epsilon);
		vector<double> V_heap_approx = get<0>(V_heap_approx_tuple);

		auto stop_VIH = high_resolution_clock::now();
		auto duration_VIH = duration_cast<microseconds>(stop_VIH - start_VIH);

		stringstream_VIH << to_string(S) << " " << duration_VIH.count() << endl;

		// VIAEHL
		A_type A8 = copy_A(A);
		auto start_VIAEHL = high_resolution_clock::now();

		// V_type VIAEHL_approx_solution_tuple = value_iteration_action_elimination_heaps_lower_bound_approx(S, R, A8, P, gamma, epsilon);
		V_type VIAEHL_approx_solution_tuple = value_iteration_with_heapGS(S, R, A8, P, gamma, epsilon);
		vector<double> VIAEHL_approx_solution = get<0>(VIAEHL_approx_solution_tuple);

		auto stop_VIAEHL = high_resolution_clock::now();
		auto duration_VIAEHL = duration_cast<microseconds>(stop_VIAEHL - start_VIAEHL);

		stringstream_VIAEHL << to_string(S) << " " << duration_VIAEHL.count() << endl;

		// BAO
		A_type A9 = copy_A(A);
		auto start_BAO = high_resolution_clock::now();

		V_type BAO_approx_solution_tuple = value_iteration_BAOSKGS(S, R, A9, P, gamma, epsilon);
		vector<double> BAO_approx_solution = get<0>(BAO_approx_solution_tuple);

		auto stop_BAO = high_resolution_clock::now();
		auto duration_BAO = duration_cast<microseconds>(stop_BAO - start_BAO);

		stringstream_BAO << to_string(S) << " " << duration_BAO.count() << endl;

		start_BAO = high_resolution_clock::now();

		V_type BAO_approx_solution_tuple1 = value_iteration_BAOSK(S, R, A9, P, gamma, epsilon);
		vector<double> BAO_approx_solution1 = get<0>(BAO_approx_solution_tuple1);

		stop_BAO = high_resolution_clock::now();
		duration_BAO = duration_cast<microseconds>(stop_BAO - start_BAO);

		stringstream_BAO << to_string(S) << " " << duration_BAO.count() << endl;
		if (abs_max_diff_vectors(BAO_approx_solution1, BAO_approx_solution) > (2 * epsilon))
		{
			printf("DIFFERENCE\n");
		}
		// They should in theory all be epsilon from true value, and therefore, at most be 2 * epsilon from each other
		if (abs_max_diff_vectors(VIAEHL_approx_solution, BAO_approx_solution) > (2 * epsilon))
		{
			printf("DIFFERENCE1\n");
		}
		if (abs_max_diff_vectors(VIAEHL_approx_solution, V_heap_approx) > (2 * epsilon))
		{
			printf("DIFFERENCE2\n");
		}
	}

	// WRITE ALL DATA TO THEIR RESPECTVIE FILES
	write_stringstream_to_file(stringstream_VIH, output_stream_VIH, file_name_VIH);
	write_stringstream_to_file(stringstream_VIAEHL, output_stream_VIAEHL, file_name_VIAEHL);
	write_stringstream_to_file(stringstream_BAO, output_stream_BAO, file_name_BAO);
}

void create_data_tables_number_of_actions_VIH_implementations(string filename, int S, int A_max, double epsilon, double gamma, double upper_reward, double non_zero_transition)
{

	// the stringstreams to create the test for the files
	ostringstream stringstream_VIH;
	ostringstream stringstream_VIH_custom;

	// the file output objects
	ofstream output_stream_VIH;
	ofstream output_stream_VIH_custom;

	// set the name of the file to write to
	string file_name_VIH = "data_tables/number_of_actions_VIH_impl/" + filename + "_VIH.dat";
	string file_name_VIH_custom = "data_tables/number_of_actions_VIH_impl/" + filename + "_VIH_custom.dat";

	// The varying parameters
	int A_starting_value = 50;
	int A_finishing_value = A_max;
	int A_increment = 50;

	// hardcoded parameter
	double action_prob = 1.0;

	for (int A_num = A_starting_value; A_num <= A_finishing_value; A_num = A_num + A_increment)
	{

		printf("Beginning iteration A_num = %d\n", A_num);

		// auto MDP = generate_random_MDP_normal_distributed_rewards(S, A_num, action_prob, non_zero_transition, seed, 1000, 10);
		// auto MDP = generate_random_MDP_exponential_distributed_rewards(S, A_num, action_prob, non_zero_transition, 0.02, seed);
		// auto MDP = generate_random_MDP_with_variable_parameters(S, A_num, action_prob, non_zero_transition, 1000, seed);
		// GENERATE THE MDP
		int seed = time(0);
		auto MDP = generate_random_MDP_exponential_distributed_rewards(S, A_num, 1.0, S, 0.02, seed);
		// auto MDP = generate_random_MDP_normal_distributed_rewards(S, A_num, 1.0, S, seed, 1000, 10);
		// auto MDP = generate_random_MDP_with_variable_parameters(S, A_num, 1.0, non_zero_transition, upper_reward, seed);
		R_type R = get<0>(MDP);
		A_type A = get<1>(MDP);
		P_type P = get<2>(MDP);

		// VIH testing
		A_type A2 = copy_A(A);
		auto start_VIH = high_resolution_clock::now();

		V_type V_heap_approx_tuple = value_iteration_with_heap(S, R, A2, P, gamma, epsilon);
		vector<double> V_heap_approx = get<0>(V_heap_approx_tuple);

		auto stop_VIH = high_resolution_clock::now();
		auto duration_VIH = duration_cast<microseconds>(stop_VIH - start_VIH);

		stringstream_VIH << to_string(A_num) << " " << duration_VIH.count() << endl;

		// VIH_custom
		A_type A8 = copy_A(A);
		auto start_VIH_custom = high_resolution_clock::now();

		V_type VIH_custom_approx_solution_tuple = value_iteration_VIH_custom(S, R, A8, P, gamma, epsilon);
		vector<double> VIH_custom_approx_solution = get<0>(VIH_custom_approx_solution_tuple);

		auto stop_VIH_custom = high_resolution_clock::now();
		auto duration_VIH_custom = duration_cast<microseconds>(stop_VIH_custom - start_VIH_custom);

		stringstream_VIH_custom << to_string(A_num) << " " << duration_VIH_custom.count() << endl;

		// They should in theory all be epsilon from true value, and therefore, at most be 2 * epsilon from each other
		if (abs_max_diff_vectors(V_heap_approx, VIH_custom_approx_solution) > (2 * epsilon))
		{
			printf("DIFFERENCE\n");
		}
	}

	// WRITE ALL DATA TO THEIR RESPECTVIE FILES
	write_stringstream_to_file(stringstream_VIH, output_stream_VIH, file_name_VIH);
	write_stringstream_to_file(stringstream_VIH_custom, output_stream_VIH_custom, file_name_VIH_custom);
}

void create_data_tables_VMS(string filename, double epsilon, double gamma)
{
	ostringstream stringstream_BVI;
	ostringstream stringstream_VI;
	ostringstream stringstream_VIU;
	ostringstream stringstream_VIH;
	ostringstream stringstream_VIAE;
	ostringstream stringstream_VIAEH;
	ostringstream stringstream_VIAEHL;
	ostringstream stringstream_BAO;
	ostringstream stringstream_BAON;
	// the file output objects
	ofstream output_stream_BVI;
	ofstream output_stream_VI;
	ofstream output_stream_VIU;
	ofstream output_stream_VIH;
	ofstream output_stream_VIAE;
	ofstream output_stream_VIAEH;
	ofstream output_stream_VIAEHL;
	ofstream output_stream_BAO;
	ofstream output_stream_BAON;
	string seed1 = "X";
	string seed11[5] = {"1", "2", "3", "4", "5"};
	int Rseed = 5;
	// set the name of the file to write to
	string file_name_BVI = "data_tables/number_of_statesVM/" + filename + seed1 + "_BVI.dat";
	string file_name_VI = "data_tables/number_of_statesVM/" + filename + seed1 + "_VI.dat";
	string file_name_VIU = "data_tables/number_of_statesVM/" + filename + seed1 + "_VIU.dat";
	string file_name_VIH = "data_tables/number_of_statesVM/" + filename + seed1 + "_VIH.dat";
	string file_name_VIAE = "data_tables/number_of_statesVM/" + filename + seed1 + "_VIAE.dat";
	string file_name_VIAEH = "data_tables/number_of_statesVM/" + filename + seed1 + "_VIAEH.dat";
	string file_name_VIAEHL = "data_tables/number_of_statesVM/" + filename + seed1 + "_VIAEHL.dat";
	string file_name_BAO = "data_tables/number_of_statesVM/" + filename + seed1 + "_BAO.dat";
	string file_name_BAON = "data_tables/number_of_statesVM/" + filename + seed1 + "_BAON.dat";
	// The varying parameters
	int S_starting_value = 50;
	// int S_finishing_value = S_max;
	// int S_increment = 50;

	// hardcoded parameter
	double action_prob = 1.0;

	// write meta data to all stringstreams as first in their respective files
	/*
	write_meta_data_to_dat_file_number_of_states(stringstream_VI, A_num, epsilon, gamma, upper_reward, non_zero_transition, action_prob, S_starting_value, S_finishing_value, S_increment);
	write_meta_data_to_dat_file_number_of_states(stringstream_VIU, A_num, epsilon, gamma, upper_reward, non_zero_transition, action_prob, S_starting_value, S_finishing_value, S_increment);
	write_meta_data_to_dat_file_number_of_states(stringstream_BVI, A_num, epsilon, gamma, upper_reward, non_zero_transition, action_prob, S_starting_value, S_finishing_value, S_increment);
	write_meta_data_to_dat_file_number_of_states(stringstream_VIH, A_num, epsilon, gamma, upper_reward, non_zero_transition, action_prob, S_starting_value, S_finishing_value, S_increment);
	write_meta_data_to_dat_file_number_of_states(stringstream_VIAE, A_num, epsilon, gamma, upper_reward, non_zero_transition, action_prob, S_starting_value, S_finishing_value, S_increment);
	write_meta_data_to_dat_file_number_of_states(stringstream_VIAEH, A_num, epsilon, gamma, upper_reward, non_zero_transition, action_prob, S_starting_value, S_finishing_value, S_increment);
*/
	S_type S = 0;
	bool StaA = false;
	if (filename == "numberVMS")
		StaA = true;
	cout << "i am good" << StaA << endl;
	// string SE,AE;
	// string SE[5] = new string[5]{"100","200","300","400","500"};
	// string AE[5]=new string[5]{"2","10","20","30","40"};
	string SE[5] = {"10S", "20S", "30S", "40S", "50S"};
	// string AE[6]={"2A","10A","20A","30A","40A","50A"};
	string AE[5] = {"10A", "20A", "30A", "40A", "50A"};
	MDP_type MDP;
	float VI[10][5];
	int k;
	for (int iter = 0; iter < 5; iter++)
	{
		seed1 = seed11[iter];
		k = 0;
		for (int temp = 0; temp < 5; temp = temp + 1)
		{
			// GENERATE THE MDP
			int seed = time(0);

			// auto MDP= readMDPS(seed1,to_string(temp));
			if (StaA)
			{
				MDP = readMDPS(seed1, SE[temp]);
				S = (temp + 1) * 100;
			}
			else
			{
				MDP = readMDPS(seed1, AE[temp]);
				S = 500;
			}
			printf("Beginning iteration iter,seed = %d,%d \n", temp, iter);

			// auto MDP = generate_random_MDP_with_variable_parameters(S, A_num, action_prob, non_zero_transition, upper_reward, seed);
			R_type R = get<0>(MDP);
			A_type A = get<1>(MDP);
			P_type P = get<2>(MDP);
			// VI testing
			// TODO: make method that takes arguments and performs the writing of data, takes filename, MDP etc. as argument
			A_type A1 = copy_A(A);
			auto start_VI = high_resolution_clock::now();
			V_type V_approx_solution_tuple = value_iterationGS(S, R, A1, P, gamma, epsilon);
			auto stop_VI = high_resolution_clock::now();
			vector<double> V_approx_solution = get<0>(V_approx_solution_tuple);
			auto duration_VI = duration_cast<milliseconds>(stop_VI - start_VI);
			cout << "CheckpointVI" << endl;
			stringstream_VI << to_string(S) << " " << duration_VI.count() << endl;

			// VIU testing
			A_type A6 = copy_A(A);
			auto start_VIU = high_resolution_clock::now();
			V_type V_approx_solution_upper_tuple = value_iteration_upperGS(S, R, A6, P, gamma, epsilon);
			auto stop_VIU = high_resolution_clock::now();
			vector<double> V_approx_solution_upper = get<0>(V_approx_solution_upper_tuple);
			auto duration_VIU = duration_cast<milliseconds>(stop_VIU - start_VIU);
			cout << "CheckpointVIU" << endl;
			stringstream_VIU << to_string(S) << " " << duration_VIU.count() << endl;

			// VIH testing
			A_type A2 = copy_A(A);
			auto start_VIH = high_resolution_clock::now();
			V_type V_heap_approx_tuple = value_iteration_with_heapGS(S, R, A2, P, gamma, epsilon);
			auto stop_VIH = high_resolution_clock::now();
			auto duration_VIH = duration_cast<milliseconds>(stop_VIH - start_VIH);
			stringstream_VIH << to_string(S) << " " << duration_VIH.count() << endl;
			vector<double> V_heap_approx = get<0>(V_heap_approx_tuple);
			// BVI
			A_type A3 = copy_A(A);
			auto start_BVI = high_resolution_clock::now();

			V_type V_bounded_approx_solution_tuple = bounded_value_iterationGS(S, R, A3, P, gamma, epsilon);
			cout << "CheckpointBVI" << endl;
			auto stop_BVI = high_resolution_clock::now();
			vector<double> V_bounded_approx_solution = get<0>(V_bounded_approx_solution_tuple);
			auto duration_BVI = duration_cast<milliseconds>(stop_BVI - start_BVI);
			stringstream_BVI << to_string(S) << " " << duration_BVI.count() << endl;

			// VIAE
			A_type A4 = copy_A(A);
			auto start_VIAE = high_resolution_clock::now();

			V_type V_AE_approx_solution_tuple = value_iteration_action_eliminationGS(S, R, A4, P, gamma, epsilon);
			auto stop_VIAE = high_resolution_clock::now();
			auto duration_VIAE = duration_cast<milliseconds>(stop_VIAE - start_VIAE);
			vector<double> V_AE_approx_solution = get<0>(V_AE_approx_solution_tuple);
			stringstream_VIAE << to_string(S) << " " << duration_VIAE.count() << endl;
			// VIAEH
			// VIAEHPROB
			A_type A5 = copy_A(A);
			auto start_VIAEH = high_resolution_clock::now();
			V_type V_AE_H_approx_solution_tuple = value_iteration_action_elimination_heapsGS(S, R, A5, P, gamma, epsilon);
			auto stop_VIAEH = high_resolution_clock::now();
			vector<double> V_AE_H_approx_solution = get<0>(V_AE_H_approx_solution_tuple);
			auto duration_VIAEH = duration_cast<milliseconds>(stop_VIAEH - start_VIAEH);

			stringstream_VIAEH << to_string(S) << " " << duration_VIAEH.count() << endl;
			// VIAEHL
			A_type A8 = copy_A(A);
			auto start_VIAEHL = high_resolution_clock::now();
			V_type VIAEHL_approx_solution_tuple = value_iteration_action_elimination_heaps_lower_bound_approxGS(S, R, A8, P, gamma, epsilon);
			auto stop_VIAEHL = high_resolution_clock::now();
			vector<double> VIAEHL_approx_solution = get<0>(VIAEHL_approx_solution_tuple);

			// auto duration_VIAEHL = duration_cast<microseconds>(stop_VIAEHL - start_VIAEHL);
			auto duration_VIAEHL = duration_cast<milliseconds>(stop_VIAEHL - start_VIAEHL);
			stringstream_VIAEHL << to_string(S) << " " << duration_VIAEHL.count() << endl;
			cout << "CheckpointVIAEH" << endl;
			// BAO
			A_type A9 = copy_A(A);
			auto start_BAO = high_resolution_clock::now();
			V_type BAO_approx_solution_tuple = value_iteration_BAOGS(S, R, A9, P, gamma, epsilon);
			auto stop_BAO = high_resolution_clock::now();
			vector<double> BAO_approx_solution = get<0>(BAO_approx_solution_tuple);
			// cout<<"CheckpointBAO"<<endl;
			auto duration_BAO = duration_cast<milliseconds>(stop_BAO - start_BAO);
			stringstream_BAO << to_string(S) << " " << duration_BAO.count() << endl;
			// auto duration_BAOSK = duration_cast<milliseconds>(stop_BAOSK - start_BAOSK);
			cout << "value_iteration_BAOGS," << duration_BAO.count() << endl;
			A_type A10 = copy_A(A);

			/*
				auto start_BAOSK = high_resolution_clock::now();
				V_type BAO_approx_solution_tuple1 = value_iteration_BAOSKGS(S, R, A10, P, gamma, epsilon);
				auto stop_BAOSK = high_resolution_clock::now();
				vector<double> BAO_approx_solution1 = get<0>(BAO_approx_solution_tuple1);
				//auto duration_BAO = duration_cast<microseconds>(stop_BAO - start_BAO);
				auto duration_BAOSK = duration_cast<milliseconds>(stop_BAOSK - start_BAOSK);
				cout<<"value_iteration_BAOSKGS,"<<duration_BAOSK.count()<<endl;
				stringstream_BAON << to_string(S) << " " << duration_BAOSK.count() << endl;
*/
			// They should in theory all be epsilon from true value, and therefore, at most be 2 * epsilon from each other

			if (abs_max_diff_vectors(V_approx_solution, V_heap_approx) > (2 * epsilon))
			{
				printf("DIFFERENCE1a\n");
			}
			if (abs_max_diff_vectors(V_approx_solution, V_AE_approx_solution) > (2 * epsilon))
			{
				printf("DIFFERENCE1b\n");
			}
			if (abs_max_diff_vectors(V_approx_solution, V_AE_H_approx_solution) > (2 * epsilon))
			{
				printf("DIFFERENCE1c\n");
			}
			if (abs_max_diff_vectors(V_approx_solution, V_bounded_approx_solution) > (2 * epsilon))
			{
				printf("DIFFERENCE1d\n");
			}
			if (abs_max_diff_vectors(V_approx_solution, VIAEHL_approx_solution) > (2 * epsilon))
			{
				printf("DIFFERENCE1e\n");
			}
			if (abs_max_diff_vectors(V_approx_solution, V_approx_solution_upper) > (2 * epsilon))
			{
				printf("DIFFERENCE1f\n");
			}
			if (abs_max_diff_vectors(V_approx_solution, BAO_approx_solution) > (2 * epsilon))
			{
				printf("DIFFERENCE1g\n");
			}
			if (abs_max_diff_vectors(V_approx_solution, BAO_approx_solution) > (2 * epsilon))
			{
				printf("DIFFERENCE1h\n");
			}
			if (abs_max_diff_vectors(V_AE_approx_solution, V_AE_H_approx_solution) > (2 * epsilon))
			{
				printf("DIFFERENCE2j\n");
			}
			if (abs_max_diff_vectors(V_bounded_approx_solution, V_approx_solution_upper) > (2 * epsilon))
			{
				printf("DIFFERENCE3\n");
			}
			if (abs_max_diff_vectors(VIAEHL_approx_solution, BAO_approx_solution) > (2 * epsilon))
			{
				printf("DIFFERENC4E\n");
			}
			if (abs_max_diff_vectors(BAO_approx_solution, BAO_approx_solution) > (2 * epsilon))
			{
				printf("DIFFERENCE5\n");
			}
			VI[0][temp] += duration_VI.count();
			VI[1][temp] += duration_VIU.count();
			VI[2][temp] += duration_VIH.count();
			VI[3][temp] += duration_BVI.count();
			VI[4][temp] += duration_VIAE.count();
			VI[5][temp] += duration_VIAEH.count();
			VI[6][temp] += duration_VIAEHL.count();
			VI[7][temp] += duration_BAO.count();
			// VI[8][temp]+=duration_BAOSK.count();
			// k++;
		}
	}
	for (int k = 0; k < 5; k++)
	{
		stringstream_VI << VI[0][k] / 5 << endl;
		stringstream_VIU << VI[1][k] / 5 << endl;
		stringstream_VIH << VI[2][k] / 5 << endl;
		stringstream_BVI << VI[3][k] / 5 << endl;
		stringstream_VIAE << VI[4][k] / 5 << endl;
		stringstream_VIAEH << VI[5][k] / 5 << endl;
		stringstream_VIAEHL << VI[6][k] / 5 << endl;
		stringstream_BAO << VI[7][k] / 5 << endl;
		stringstream_BAON << VI[8][k] / 5 << endl;
		// cout<<"writeA"<<endl;
	}

	// WRITE ALL DATA TO THEIR RESPECTVIE FILES
	write_stringstream_to_file(stringstream_VI, output_stream_VI, file_name_VI);
	write_stringstream_to_file(stringstream_VIU, output_stream_VIU, file_name_VIU);
	write_stringstream_to_file(stringstream_BVI, output_stream_BVI, file_name_BVI);
	write_stringstream_to_file(stringstream_VIH, output_stream_VIH, file_name_VIH);
	write_stringstream_to_file(stringstream_VIAE, output_stream_VIAE, file_name_VIAE);
	write_stringstream_to_file(stringstream_VIAEH, output_stream_VIAEH, file_name_VIAEH);
	write_stringstream_to_file(stringstream_VIAEHL, output_stream_VIAEHL, file_name_VIAEHL);
	write_stringstream_to_file(stringstream_BAO, output_stream_BAO, file_name_BAO);
	write_stringstream_to_file(stringstream_BAON, output_stream_BAON, file_name_BAON);
}

void create_data_tables_number_of_statesGS(string filename, int S_max, int A_num, double epsilon, double gamma, double upper_reward, double non_zero_transition)
{

	// the stringstreams to create the test for the files
	ostringstream stringstream_BVI;
	ostringstream stringstream_VI;
	ostringstream stringstream_VIU;
	ostringstream stringstream_VIH;
	ostringstream stringstream_VIHN;
	ostringstream stringstream_VIAE;
	ostringstream stringstream_VIAEH;
	ostringstream stringstream_VIAEHL;
	ostringstream stringstream_BAO;
	ostringstream stringstream_BAON;

	// the file output objects
	ofstream output_stream_BVI;
	ofstream output_stream_VI;
	ofstream output_stream_VIU;
	ofstream output_stream_VIH;
	ofstream output_stream_VIHN;
	ofstream output_stream_VIAE;
	ofstream output_stream_VIAEH;
	ofstream output_stream_VIAEHL;
	ofstream output_stream_BAO;
	ofstream output_stream_BAON;

	// set the name of the file to write to
	string file_name_BVI = "data_tables/number_of_states/" + filename + "ScActionsGS_BVI.dat";
	string file_name_VI = "data_tables/number_of_states/" + filename + "ScActionsGS_VI.dat";
	string file_name_VIU = "data_tables/number_of_states/" + filename + "ScActionsGS_VIU.dat";
	string file_name_VIH = "data_tables/number_of_states/" + filename + "ScActionsGS_VIH.dat";
	string file_name_VIHN = "data_tables/number_of_states/" + filename + "ScActionsGS_VIHN.dat";
	string file_name_VIAE = "data_tables/number_of_states/" + filename + "ScActionsGS_VIAE.dat";
	string file_name_VIAEH = "data_tables/number_of_states/" + filename + "ScActionsGS_VIAEH.dat";
	string file_name_VIAEHL = "data_tables/number_of_states/" + filename + "ScActionsGS_VIAEHL.dat";
	string file_name_BAO = "data_tables/number_of_states/" + filename + "ScActionsGS_BAO.dat";
	string file_name_BAON = "data_tables/number_of_states/" + filename + "ScActionsGS_BAON.dat";

	// The varying parameters
	int S_starting_value = 100;
	int S_finishing_value = S_max;
	int S_increment = 100;
	S_finishing_value = 2000;
	// hardcoded parameter
	double action_prob = 1.0;
	// A_num=100;
	// write meta data to all stringstreams as first in their respective files

	write_meta_data_to_dat_file_number_of_states(stringstream_VI, A_num, epsilon, gamma, upper_reward, non_zero_transition, action_prob, S_starting_value, S_finishing_value, S_increment);
	write_meta_data_to_dat_file_number_of_states(stringstream_VIU, A_num, epsilon, gamma, upper_reward, non_zero_transition, action_prob, S_starting_value, S_finishing_value, S_increment);
	write_meta_data_to_dat_file_number_of_states(stringstream_BVI, A_num, epsilon, gamma, upper_reward, non_zero_transition, action_prob, S_starting_value, S_finishing_value, S_increment);
	write_meta_data_to_dat_file_number_of_states(stringstream_VIH, A_num, epsilon, gamma, upper_reward, non_zero_transition, action_prob, S_starting_value, S_finishing_value, S_increment);
	write_meta_data_to_dat_file_number_of_states(stringstream_VIAE, A_num, epsilon, gamma, upper_reward, non_zero_transition, action_prob, S_starting_value, S_finishing_value, S_increment);
	write_meta_data_to_dat_file_number_of_states(stringstream_VIAEH, A_num, epsilon, gamma, upper_reward, non_zero_transition, action_prob, S_starting_value, S_finishing_value, S_increment);
	float VI[10][20];
	int S2 = 0;
	int k = 0;
	// int S=500;
	// int S=1000;
	int S = 500;
	// A_num=100;
	S = 500;
	S2 = S / 10;
	S2 = 100;
	A_num = 100;
	S = 100;
	S2 = 10;
	for (int iters = 0; iters < 5; iters++)
	{
		k = 0;
		for (A_num = S_starting_value; A_num <= S_finishing_value; A_num = A_num + S_increment)
		{
			// S2=S/10;
			printf("Beginning iteration %d  S2,  %d, A, S %d = %d\n", iters, S2, A_num, S);

			// GENERATE THE MDP
			int seed = time(0);
			// auto MDP = generate_random_MDP_exponential_distributed_rewards(S, A_num, 1.0, S, 0.02, seed);
			auto MDP = generate_random_MDP_normal_distributed_rewards(S, A_num, action_prob, S2, seed, 1000, 10);
			// auto MDP = RiverSwim(S);
			// int xs=90;
			// auto MDP=Maze(xs,xs,seed);
			// S=xs*xs+1;
			// auto MDP = generate_random_MDP_with_variable_parameters(S, A_num, action_prob, non_zero_transition, upper_reward, seed);
			cout << "HI" << endl;

			R_type R = get<0>(MDP);
			A_type A = get<1>(MDP);
			P_type P = get<2>(MDP);
			int counter = 0;
			/*
			for (int t = 0; t < S; t++) {
					for (auto a : A[t]) {
							auto& [P_s_a, P_s_a_nonzero] = P[t][a];
								for (int k : P_s_a_nonzero) {
									//cout<<P_s_a[counter]<<"PSA"<<P_s_a_nonzero[counter]<<"P_s_a_nonzero"<< R[t][a]  <<"Rew"<<a<<"A"<<endl;
									counter++;
									}
									counter=0;

							}
					}*/
			// cout<<"MDP"<<endl;
			// VI testing
			// TODO: make method that takes arguments and performs the writing of data, takes filename, MDP etc. as argument
			A_type A1 = copy_A(A);
			auto start_VI = high_resolution_clock::now();
			// cout<<getValue()<<endl;
			// V_type V_approx_solution_tuple = value_iterationGS(S, R, A1, P, gamma, epsilon);
			// vector<double> V_approx_solution = get<0>(V_approx_solution_tuple);
			auto stop_VI = high_resolution_clock::now();
			auto duration_VI = duration_cast<milliseconds>(stop_VI - start_VI);

			stringstream_VI << to_string(S) << " " << duration_VI.count() << endl;
			VI[0][k] += duration_VI.count();
			cout << "VI," << duration_VI.count() << endl;
			// VIU testing
			A_type A6 = copy_A(A);
			auto start_VIU = high_resolution_clock::now();
			// V_type V_approx_solution_upper_tuple = value_iteration_upperGS(S, R, A6, P, gamma, epsilon);
			// vector<double> V_approx_solution_upper = get<0>(V_approx_solution_upper_tuple);
			auto stop_VIU = high_resolution_clock::now();
			auto duration_VIU = duration_cast<milliseconds>(stop_VIU - start_VIU);

			VI[1][k] += duration_VIU.count();
			cout << "VIU," << duration_VIU.count() << endl;
			stringstream_VIU << to_string(S) << " " << duration_VIU.count() << endl;

			// VIH testing
			A_type A2 = copy_A(A);
			auto start_VIH = high_resolution_clock::now();

			V_type V_heap_approx_tuple = value_iteration_with_heapGS(S, R, A2, P, gamma, epsilon);
			vector<double> V_heap_approx = get<0>(V_heap_approx_tuple);
			auto stop_VIH = high_resolution_clock::now();
			auto duration_VIH = duration_cast<milliseconds>(stop_VIH - start_VIH);
			VI[2][k] += duration_VIH.count();
			cout << "VIHGS," << duration_VIH.count() << endl;
			stringstream_VIH << to_string(S) << " " << duration_VIH.count() << endl;
			// A_type A12 = copy_A(A);
			auto start_VIHN = high_resolution_clock::now();

			// V_type V_heap_approx_tupleN = value_iteration_VIH_custom(S, R, A12, P, gamma, epsilon);
			// vector<double> V_heap_approxN = get<0>(V_heap_approx_tuple);

			auto stop_VIHN = high_resolution_clock::now();
			auto duration_VIHN = duration_cast<milliseconds>(stop_VIHN - start_VIHN);

			// stringstream_VIHN << to_string(A_num) << " " << duration_VIHN.count() << endl;
			VI[9][k] += duration_VIHN.count();

			// BVI
			A_type A3 = copy_A(A);
			auto start_BVI = high_resolution_clock::now();

			// V_type V_bounded_approx_solution_tuple = bounded_value_iterationGS(S, R, A3, P, gamma, epsilon);
			// vector<double> V_bounded_approx_solution = get<0>(V_bounded_approx_solution_tuple);
			auto stop_BVI = high_resolution_clock::now();
			auto duration_BVI = duration_cast<milliseconds>(stop_BVI - start_BVI);

			stringstream_BVI << to_string(S) << " " << duration_BVI.count() << endl;
			VI[3][k] += duration_BVI.count();
			cout << "bvigs," << duration_BVI.count() << endl;
			A_type A4 = copy_A(A);
			auto start_VIAE = high_resolution_clock::now();
			// V_type V_AE_approx_solution_tuple = value_iteration_action_eliminationGS(S, R, A4, P, gamma, epsilon);
			// vector<double> V_AE_approx_solution = get<0>(V_AE_approx_solution_tuple);

			auto stop_VIAE = high_resolution_clock::now();
			auto duration_VIAE = duration_cast<milliseconds>(stop_VIAE - start_VIAE);

			stringstream_VIAE << to_string(S) << " " << duration_VIAE.count() << endl;
			VI[4][k] += duration_VIAE.count();
			// VIAEH
			A_type A5 = copy_A(A);
			auto start_VIAEH = high_resolution_clock::now();

			// V_type V_AE_H_approx_solution_tuple = value_iteration_action_elimination_heapsGS(S, R, A5, P, gamma, epsilon);
			// vector<double> V_AE_H_approx_solution = get<0>(V_AE_H_approx_solution_tuple);

			auto stop_VIAEH = high_resolution_clock::now();
			auto duration_VIAEH = duration_cast<milliseconds>(stop_VIAEH - start_VIAEH);

			stringstream_VIAEH << to_string(S) << " " << duration_VIAEH.count() << endl;
			VI[5][k] += duration_VIAEH.count();
			cout << "viae," << duration_VIAE.count() << endl;
			// VIAEHL
			cout << "viaeh" << duration_VIAEH.count() << endl;
			A_type A8 = copy_A(A);
			auto start_VIAEHL = high_resolution_clock::now();

			V_type VIAEHL_approx_solution_tuple = value_iteration_action_elimination_heaps_lower_bound_approxGS(S, R, A8, P, gamma, epsilon);
			vector<double> VIAEHL_approx_solution = get<0>(VIAEHL_approx_solution_tuple);

			auto stop_VIAEHL = high_resolution_clock::now();
			auto duration_VIAEHL = duration_cast<milliseconds>(stop_VIAEHL - start_VIAEHL);
			stringstream_VIAEHL << to_string(S) << " " << duration_VIAEHL.count() << endl;
			VI[6][k] += duration_VIAEHL.count();
			cout << "VIAEHL," << duration_VIAEHL.count() << endl;
			// BAO
			A_type A9 = copy_A(A);
			auto start_BAO = high_resolution_clock::now();

			V_type BAO_approx_solution_tuple = value_iteration_BAOGS(S, R, A9, P, gamma, epsilon);
			vector<double> BAO_approx_solution = get<0>(BAO_approx_solution_tuple);

			auto stop_BAO = high_resolution_clock::now();
			auto duration_BAO = duration_cast<milliseconds>(stop_BAO - start_BAO);

			stringstream_BAO << to_string(S) << " " << duration_BAO.count() << endl;
			VI[7][k] += duration_BAO.count();

			A_type A10 = copy_A(A);
			auto start_BAOSK = high_resolution_clock::now();

			V_type BAO_approx_solution_tuple1 = value_iteration_BAOSKGS(S, R, A10, P, gamma, epsilon);
			vector<double> BAO_approx_solution1 = get<0>(BAO_approx_solution_tuple1);
			auto stop_BAOSK = high_resolution_clock::now();
			cout << "BAOGS," << duration_BAO.count() << endl;

			auto duration_BAOSK = duration_cast<milliseconds>(stop_BAOSK - start_BAOSK);
			stringstream_BAON << to_string(S) << " " << duration_BAOSK.count() << endl;
			VI[8][k] += duration_BAOSK.count();
			cout << "BAOOLDGS," << duration_BAOSK.count() << endl;
			// They should in theory all be epsilon from true value, and therefore, at most be 2 * epsilon from each other
			/*
			if (abs_max_diff_vectors(V_approx_solution, BAO_approx_solution) > (2 * epsilon)){
					printf("DIFFERENCE1\n");
			}
			if (abs_max_diff_vectors(V_approx_solution, V_heap_approx) > (2 * epsilon)){
					printf("DIFFERENCE2\n");
			}

			if (abs_max_diff_vectors(V_approx_solution, V_AE_H_approx_solution) > (2 * epsilon)){
					printf("DIFFERENCE3\n");
			}
			if (abs_max_diff_vectors(V_approx_solution, V_approx_solution_upper) > (2 * epsilon)){
					printf("DIFFERENCE4\n");
			}
			if (abs_max_diff_vectors(V_approx_solution, V_bounded_approx_solution) > (2 * epsilon)){
					printf("DIFFERENCE5\n");
			}
			if (abs_max_diff_vectors(V_approx_solution, VIAEHL_approx_solution) > (2 * epsilon)){
					printf("DIFFERENCE6\n");
			}
			if (abs_max_diff_vectors(V_approx_solution, V_AE_approx_solution) > (2 * epsilon)){
					printf("DIFFERENCE7\n");
			}
				if (abs_max_diff_vectors(V_approx_solution, BAO_approx_solution) > (2 * epsilon)){
					printf("DIFFERENCE8\n");
			}
*/
			k++;
		}
	}
	for (int k = 0; k < 20; k++)
	{
		stringstream_VI << VI[0][k] / 5 << endl;
		stringstream_VIU << VI[1][k] / 5 << endl;
		stringstream_VIH << VI[2][k] / 5 << endl;
		stringstream_VIHN << VI[9][k] / 5 << endl;
		stringstream_BVI << VI[3][k] / 5 << endl;
		stringstream_VIAE << VI[4][k] / 5 << endl;
		stringstream_VIAEH << VI[5][k] / 5 << endl;
		stringstream_VIAEHL << VI[6][k] / 5 << endl;
		stringstream_BAO << VI[7][k] / 5 << endl;
		stringstream_BAON << VI[8][k] / 5 << endl;
		// cout<<"writeA"<<endl;
	}
	// WRITE ALL DATA TO THEIR RESPECTVIE FILES
	write_stringstream_to_file(stringstream_VI, output_stream_VI, file_name_VI);
	write_stringstream_to_file(stringstream_VIU, output_stream_VIU, file_name_VIU);
	write_stringstream_to_file(stringstream_BVI, output_stream_BVI, file_name_BVI);
	write_stringstream_to_file(stringstream_VIH, output_stream_VIH, file_name_VIH);
	write_stringstream_to_file(stringstream_VIHN, output_stream_VIHN, file_name_VIHN);
	write_stringstream_to_file(stringstream_VIAE, output_stream_VIAE, file_name_VIAE);
	write_stringstream_to_file(stringstream_VIAEH, output_stream_VIAEH, file_name_VIAEH);
	write_stringstream_to_file(stringstream_VIAEHL, output_stream_VIAEHL, file_name_VIAEHL);
	write_stringstream_to_file(stringstream_BAO, output_stream_BAO, file_name_BAO);
	write_stringstream_to_file(stringstream_BAON, output_stream_BAON, file_name_BAON);
	// cout<<"writeF"<<endl;
}

void create_data_tables_number_of_statesGSAll(string filename, int S_max, int A_num, double epsilon, double gamma, double upper_reward, double non_zero_transition)
{

	// the stringstreams to create the test for the files
	ostringstream stringstream_BVI;
	ostringstream stringstream_VI;
	ostringstream stringstream_VIU;
	ostringstream stringstream_VIH;
	ostringstream stringstream_VIHN;
	ostringstream stringstream_VIAE;
	ostringstream stringstream_VIAEH;
	ostringstream stringstream_VIAEHL;
	ostringstream stringstream_BAO;
	ostringstream stringstream_BAON;

	// the file output objects
	ofstream output_stream_BVI;
	ofstream output_stream_VI;
	ofstream output_stream_VIU;
	ofstream output_stream_VIH;
	ofstream output_stream_VIHN;
	ofstream output_stream_VIAE;
	ofstream output_stream_VIAEH;
	ofstream output_stream_VIAEHL;
	ofstream output_stream_BAO;
	ofstream output_stream_BAON;

	// set the name of the file to write to
	string file_name_BVI = "data_tables/number_of_states/" + filename + "SSaa_BVI.dat";
	string file_name_VI = "data_tables/number_of_states/" + filename + "SSaa_VI.dat";
	string file_name_VIU = "data_tables/number_of_states/" + filename + "SaaS_VIU.dat";
	string file_name_VIH = "data_tables/number_of_states/" + filename + "S500_SSA1000_VIH.dat";
	string file_name_VIHN = "data_tables/number_of_states/" + filename + "SaaS_VIHN.dat";
	string file_name_VIAE = "data_tables/number_of_states/" + filename + "SSaa_VIAE.dat";
	string file_name_VIAEH = "data_tables/number_of_states/" + filename + "SSaa_VIAEH.dat";
	string file_name_VIAEHL = "data_tables/number_of_states/" + filename + "S500_SSA1000_VIAEHL.dat";
	string file_name_BAO = "data_tables/number_of_states/" + filename + "SSaa_BAO.dat";
	string file_name_BAON = "data_tables/number_of_states/" + filename + "S500_SSA1000_BAON.dat";

	// The varying parameters
	int S_starting_value = 50;
	int S_finishing_value = S_max;
	int S_increment = 50;

	// hardcoded parameter
	double action_prob = 1.0;

	// write meta data to all stringstreams as first in their respective files

	write_meta_data_to_dat_file_number_of_states(stringstream_VI, A_num, epsilon, gamma, upper_reward, non_zero_transition, action_prob, S_starting_value, S_finishing_value, S_increment);
	write_meta_data_to_dat_file_number_of_states(stringstream_VIU, A_num, epsilon, gamma, upper_reward, non_zero_transition, action_prob, S_starting_value, S_finishing_value, S_increment);
	write_meta_data_to_dat_file_number_of_states(stringstream_BVI, A_num, epsilon, gamma, upper_reward, non_zero_transition, action_prob, S_starting_value, S_finishing_value, S_increment);
	write_meta_data_to_dat_file_number_of_states(stringstream_VIH, A_num, epsilon, gamma, upper_reward, non_zero_transition, action_prob, S_starting_value, S_finishing_value, S_increment);
	write_meta_data_to_dat_file_number_of_states(stringstream_VIAE, A_num, epsilon, gamma, upper_reward, non_zero_transition, action_prob, S_starting_value, S_finishing_value, S_increment);
	write_meta_data_to_dat_file_number_of_states(stringstream_VIAEH, A_num, epsilon, gamma, upper_reward, non_zero_transition, action_prob, S_starting_value, S_finishing_value, S_increment);
	float VI[10][10];
	int S2 = 0;
	int k = 0;
	// int S=500;
	int S = 100;
	for (int iters = 0; iters < 5; iters++)
	{
		k = 0;
		for (int An = S_starting_value; An <= S_finishing_value; An = An + S_increment)
		{
			int S2 = S / 10;
			printf("Beginning iteration %d A = %d\n", iters, An);

			// GENERATE THE MDP
			int seed = time(0);
			// auto MDP = generate_random_MDP_exponential_distributed_rewards(S, A_num, 1.0, S/2, 0.02, seed);
			auto MDP = generate_random_MDP_normal_distributed_rewards(S, An, action_prob, S2, seed, 1000, 10);
			// auto MDP = generate_random_MDP_with_variable_parameters(S, A_num, action_prob, non_zero_transition, upper_reward, seed);
			cout << "HI" << endl;

			R_type R = get<0>(MDP);
			A_type A = get<1>(MDP);
			P_type P = get<2>(MDP);
			cout << "MDP" << endl;
			// VI testing
			// TODO: make method that takes arguments and performs the writing of data, takes filename, MDP etc. as argument
			A_type A1 = copy_A(A);
			auto start_VI = high_resolution_clock::now();
			// cout<<getValue()<<endl;
			V_type V_approx_solution_tuple = value_iterationGS(S, R, A1, P, gamma, epsilon);
			vector<double> V_approx_solution = get<0>(V_approx_solution_tuple);
			auto stop_VI = high_resolution_clock::now();
			auto duration_VI = duration_cast<milliseconds>(stop_VI - start_VI);
			stringstream_VI << to_string(S) << " " << duration_VI.count() << endl;
			VI[0][k] += duration_VI.count();

			// VIold
			auto start_VIo = high_resolution_clock::now();
			// cout<<getValue()<<endl;
			V_type V_approx_solution_tupleO = value_iteration(S, R, A1, P, gamma, epsilon);
			vector<double> V_approx_solutionO = get<0>(V_approx_solution_tupleO);
			auto stop_VIo = high_resolution_clock::now();
			auto duration_VIo = duration_cast<milliseconds>(stop_VIo - start_VIo);
			stringstream_VI << to_string(S) << " " << duration_VIo.count() << endl;

			// VIU testing
			A_type A6 = copy_A(A);
			auto start_VIU = high_resolution_clock::now();
			V_type V_approx_solution_upper_tuple = value_iteration_upperGS(S, R, A6, P, gamma, epsilon);
			vector<double> V_approx_solution_upper = get<0>(V_approx_solution_upper_tuple);
			auto stop_VIU = high_resolution_clock::now();
			auto duration_VIU = duration_cast<milliseconds>(stop_VIU - start_VIU);
			VI[1][k] += duration_VIU.count();
			stringstream_VIU << to_string(S) << " " << duration_VIU.count() << endl;

			auto start_VIUo = high_resolution_clock::now();
			V_type V_approx_solution_upper_tupleO = value_iteration_upper(S, R, A6, P, gamma, epsilon);
			vector<double> V_approx_solution_upperO = get<0>(V_approx_solution_upper_tupleO);
			auto stop_VIUo = high_resolution_clock::now();
			auto duration_VIUo = duration_cast<milliseconds>(stop_VIUo - start_VIUo);
			stringstream_VIU << to_string(S) << " " << duration_VIUo.count() << endl;

			// VIH testing
			A_type A2 = copy_A(A);
			auto start_VIH = high_resolution_clock::now();

			V_type V_heap_approx_tuple = value_iteration_with_heapGS(S, R, A2, P, gamma, epsilon);
			vector<double> V_heap_approx = get<0>(V_heap_approx_tuple);
			auto stop_VIH = high_resolution_clock::now();
			auto duration_VIH = duration_cast<milliseconds>(stop_VIH - start_VIH);
			VI[2][k] += duration_VIH.count();

			stringstream_VIH << to_string(S) << " " << duration_VIH.count() << endl;
			// A_type A12 = copy_A(A);

			auto start_VIHo = high_resolution_clock::now();

			V_type V_heap_approx_tupleO = value_iteration_with_heap(S, R, A2, P, gamma, epsilon);
			vector<double> V_heap_approxO = get<0>(V_heap_approx_tupleO);
			auto stop_VIHo = high_resolution_clock::now();
			auto duration_VIHo = duration_cast<milliseconds>(stop_VIHo - start_VIHo);

			stringstream_VIH << to_string(S) << " " << duration_VIHo.count() << endl;

			auto start_VIHN = high_resolution_clock::now();

			// V_type V_heap_approx_tupleN = value_iteration_VIH_custom(S, R, A12, P, gamma, epsilon);
			// vector<double> V_heap_approxN = get<0>(V_heap_approx_tuple);

			auto stop_VIHN = high_resolution_clock::now();
			auto duration_VIHN = duration_cast<milliseconds>(stop_VIHN - start_VIHN);

			// stringstream_VIHN << to_string(A_num) << " " << duration_VIHN.count() << endl;
			VI[9][k] += duration_VIHo.count();

			// BVI
			A_type A3 = copy_A(A);
			auto start_BVI = high_resolution_clock::now();

			V_type V_bounded_approx_solution_tuple = bounded_value_iterationGS(S, R, A3, P, gamma, epsilon);
			vector<double> V_bounded_approx_solution = get<0>(V_bounded_approx_solution_tuple);
			auto stop_BVI = high_resolution_clock::now();
			auto duration_BVI = duration_cast<milliseconds>(stop_BVI - start_BVI);

			stringstream_BVI << to_string(S) << " " << duration_BVI.count() << endl;
			VI[3][k] += duration_BVI.count();

			auto start_BVIo = high_resolution_clock::now();

			V_type V_bounded_approx_solution_tupleO = bounded_value_iteration(S, R, A3, P, gamma, epsilon);
			vector<double> V_bounded_approx_solutionO = get<0>(V_bounded_approx_solution_tupleO);
			auto stop_BVIo = high_resolution_clock::now();
			auto duration_BVIo = duration_cast<milliseconds>(stop_BVIo - start_BVIo);

			stringstream_BVI << to_string(S) << " " << duration_BVIo.count() << endl;

			A_type A4 = copy_A(A);
			auto start_VIAE = high_resolution_clock::now();

			V_type V_AE_approx_solution_tuple = value_iteration_action_eliminationGS(S, R, A4, P, gamma, epsilon);
			vector<double> V_AE_approx_solution = get<0>(V_AE_approx_solution_tuple);

			auto stop_VIAE = high_resolution_clock::now();
			auto duration_VIAE = duration_cast<milliseconds>(stop_VIAE - start_VIAE);

			stringstream_VIAE << to_string(S) << " " << duration_VIAE.count() << endl;

			auto start_VIAEo = high_resolution_clock::now();

			V_type V_AE_approx_solution_tupleO = value_iteration_action_elimination(S, R, A1, P, gamma, epsilon);
			vector<double> V_AE_approx_solutionO = get<0>(V_AE_approx_solution_tupleO);

			auto stop_VIAEo = high_resolution_clock::now();
			auto duration_VIAEo = duration_cast<milliseconds>(stop_VIAEo - start_VIAEo);

			stringstream_VIAE << to_string(S) << " " << duration_VIAEo.count() << endl;
			VI[4][k] += duration_VIAE.count();
			// VIAEH
			A_type A5 = copy_A(A);
			auto start_VIAEH = high_resolution_clock::now();

			V_type V_AE_H_approx_solution_tuple = value_iteration_action_elimination_heapsGS(S, R, A5, P, gamma, epsilon);
			vector<double> V_AE_H_approx_solution = get<0>(V_AE_H_approx_solution_tuple);

			auto stop_VIAEH = high_resolution_clock::now();
			auto duration_VIAEH = duration_cast<milliseconds>(stop_VIAEH - start_VIAEH);

			stringstream_VIAEH << to_string(S) << " " << duration_VIAEH.count() << endl;
			VI[5][k] += duration_VIAEH.count();

			auto start_VIAEHo = high_resolution_clock::now();

			V_type V_AE_H_approx_solution_tupleO = value_iteration_action_elimination_heaps(S, R, A2, P, gamma, epsilon);
			vector<double> V_AE_H_approx_solutionO = get<0>(V_AE_H_approx_solution_tupleO);

			auto stop_VIAEHo = high_resolution_clock::now();
			auto duration_VIAEHo = duration_cast<milliseconds>(stop_VIAEHo - start_VIAEHo);

			stringstream_VIAEH << to_string(S) << " " << duration_VIAEHo.count() << endl;

			// VIAEHL
			A_type A8 = copy_A(A);
			auto start_VIAEHL = high_resolution_clock::now();

			V_type VIAEHL_approx_solution_tuple = value_iteration_action_elimination_heaps_lower_bound_approxGS(S, R, A8, P, gamma, epsilon);
			vector<double> VIAEHL_approx_solution = get<0>(VIAEHL_approx_solution_tuple);

			auto stop_VIAEHL = high_resolution_clock::now();
			auto duration_VIAEHL = duration_cast<milliseconds>(stop_VIAEHL - start_VIAEHL);
			stringstream_VIAEHL << to_string(S) << " " << duration_VIAEHL.count() << endl;
			VI[6][k] += duration_VIAEHL.count();

			auto start_VIAEHLo = high_resolution_clock::now();

			V_type VIAEHL_approx_solution_tupleO = value_iteration_action_elimination_heaps_lower_bound_approx(S, R, A6, P, gamma, epsilon);
			vector<double> VIAEHL_approx_solutionO = get<0>(VIAEHL_approx_solution_tupleO);

			auto stop_VIAEHLo = high_resolution_clock::now();
			auto duration_VIAEHLo = duration_cast<milliseconds>(stop_VIAEHLo - start_VIAEHLo);
			stringstream_VIAEHL << to_string(S) << " " << duration_VIAEHLo.count() << endl;

			// BAO
			A_type A9 = copy_A(A);
			auto start_BAO = high_resolution_clock::now();

			// V_type BAO_approx_solution_tuple = value_iteration_BAO(S, R, A9, P, gamma, epsilon);
			// vector<double> BAO_approx_solution = get<0>(BAO_approx_solution_tuple);

			auto stop_BAO = high_resolution_clock::now();
			auto duration_BAO = duration_cast<milliseconds>(stop_BAO - start_BAO);

			stringstream_BAO << to_string(S) << " " << duration_BAO.count() << endl;
			VI[7][k] += duration_BAO.count();

			A_type A10 = copy_A(A);
			auto start_BAOSK = high_resolution_clock::now();

			V_type BAO_approx_solution_tuple1 = value_iteration_BAOSKGS(S, R, A10, P, gamma, epsilon);
			vector<double> BAO_approx_solution1 = get<0>(BAO_approx_solution_tuple1);
			auto stop_BAOSK = high_resolution_clock::now();

			auto duration_BAOSK = duration_cast<milliseconds>(stop_BAOSK - start_BAOSK);
			stringstream_BAON << to_string(S) << " " << duration_BAOSK.count() << endl;

			auto start_BAOSKo = high_resolution_clock::now();

			V_type BAO_approx_solution_tuple1O = value_iteration_BAOSK(S, R, A10, P, gamma, epsilon);
			vector<double> BAO_approx_solution1O = get<0>(BAO_approx_solution_tuple1O);
			auto stop_BAOSKo = high_resolution_clock::now();

			auto duration_BAOSKo = duration_cast<milliseconds>(stop_BAOSKo - start_BAOSKo);
			stringstream_BAON << to_string(S) << " " << duration_BAOSKo.count() << endl;
			VI[8][k] += duration_BAOSK.count();
			VI[7][k] += duration_BAOSKo.count();
			// They should in theory all be epsilon from true value, and therefore, at most be 2 * epsilon from each other

			if (abs_max_diff_vectors(V_approx_solution, V_approx_solutionO) > (2 * epsilon))
			{
				stringstream_BAON << "PROBLEM" << endl;
				cout << "prob" << endl;
			}
			if (abs_max_diff_vectors(V_heap_approxO, V_heap_approx) > (2 * epsilon))
			{
				stringstream_BAON << "PROBLEM" << endl;
				cout << "prob" << endl;
			}
			if (abs_max_diff_vectors(V_AE_approx_solution, V_AE_approx_solutionO) > (2 * epsilon))
			{
				stringstream_BAON << "PROBLEM" << endl;
				cout << "prob" << endl;
			}
			if (abs_max_diff_vectors(V_bounded_approx_solution, V_bounded_approx_solutionO) > (2 * epsilon))
			{
				stringstream_BAON << "PROBLEM" << endl;
				cout << "prob" << endl;
			}
			if (abs_max_diff_vectors(VIAEHL_approx_solution, VIAEHL_approx_solutionO) > (2 * epsilon))
			{
				stringstream_BAON << "PROBLEM" << endl;
				cout << "prob" << endl;
			}
			if (abs_max_diff_vectors(V_AE_H_approx_solutionO, V_AE_H_approx_solution) > (2 * epsilon))
			{
				stringstream_BAON << "PROBLEM" << endl;
				cout << "prob" << endl;
			}
			if (abs_max_diff_vectors(V_approx_solution_upperO, V_approx_solution_upper) > (2 * epsilon))
			{
				stringstream_BAON << "PROBLEM" << endl;
				cout << "prob" << endl;
			}
			if (abs_max_diff_vectors(BAO_approx_solution1O, BAO_approx_solution1) > (2 * epsilon))
			{
				stringstream_BAON << "PROBLEM" << endl;
				cout << "prob" << endl;
			}
			k++;
		}
	}
	for (int k = 0; k < 10; k++)
	{
		stringstream_VI << VI[0][k] / 5 << endl;
		stringstream_VIU << VI[1][k] / 5 << endl;
		stringstream_VIH << VI[2][k] / 5 << endl;
		stringstream_VIHN << VI[9][k] / 5 << endl;
		stringstream_BVI << VI[3][k] / 5 << endl;
		stringstream_VIAE << VI[4][k] / 5 << endl;
		stringstream_VIAEH << VI[5][k] / 5 << endl;
		stringstream_VIAEHL << VI[6][k] / 5 << endl;
		stringstream_BAO << VI[7][k] / 5 << endl;
		stringstream_BAON << VI[8][k] / 5 << endl;
		// cout<<"writeA"<<endl;
	}
	// WRITE ALL DATA TO THEIR RESPECTVIE FILES
	write_stringstream_to_file(stringstream_VI, output_stream_VI, file_name_VI);
	write_stringstream_to_file(stringstream_VIU, output_stream_VIU, file_name_VIU);
	write_stringstream_to_file(stringstream_BVI, output_stream_BVI, file_name_BVI);
	write_stringstream_to_file(stringstream_VIH, output_stream_VIH, file_name_VIH);
	write_stringstream_to_file(stringstream_VIHN, output_stream_VIHN, file_name_VIHN);
	write_stringstream_to_file(stringstream_VIAE, output_stream_VIAE, file_name_VIAE);
	write_stringstream_to_file(stringstream_VIAEH, output_stream_VIAEH, file_name_VIAEH);
	write_stringstream_to_file(stringstream_VIAEHL, output_stream_VIAEHL, file_name_VIAEHL);
	write_stringstream_to_file(stringstream_BAO, output_stream_BAO, file_name_BAO);
	write_stringstream_to_file(stringstream_BAON, output_stream_BAON, file_name_BAON);
	cout << "writeF" << endl;
}

void create_data_tables_number_GS(string filename, int expnum, int States, int Actions, int SS, int StartP, int endP, int IncP, double epsilon, double gamma, double upper_reward, double non_zero_transition)
{

	// the stringstreams to create the test for the files
	ostringstream stringstream_BVI;
	ostringstream stringstream_VI;
	ostringstream stringstream_VIU;
	ostringstream stringstream_VIH;
	ostringstream stringstream_VIHN;
	ostringstream stringstream_VIAE;
	ostringstream stringstream_VIAEH;
	ostringstream stringstream_VIAEHL;
	ostringstream stringstream_BAO;
	ostringstream stringstream_BAON;

	// the file output objects
	ofstream output_stream_BVI;
	ofstream output_stream_VI;
	ofstream output_stream_VIU;
	ofstream output_stream_VIH;
	ofstream output_stream_VIHN;
	ofstream output_stream_VIAE;
	ofstream output_stream_VIAEH;
	ofstream output_stream_VIAEHL;
	ofstream output_stream_BAO;
	ofstream output_stream_BAON;

	// set the name of the file to write to
	string file_name_BVI = "data_tables/number_of_states/" + filename + "GS_BVI.dat";
	string file_name_VI = "data_tables/number_of_states/" + filename + "GS_VI.dat";
	string file_name_VIU = "data_tables/number_of_states/" + filename + "GS_VIU.dat";
	string file_name_VIH = "data_tables/number_of_states/" + filename + "GS_VIH.dat";
	string file_name_VIHN = "data_tables/number_of_states/" + filename + "GS_VIHN.dat";
	string file_name_VIAE = "data_tables/number_of_states/" + filename + "GS_VIAE.dat";
	string file_name_VIAEH = "data_tables/number_of_states/" + filename + "GS_VIAEH.dat";
	string file_name_VIAEHL = "data_tables/number_of_states/" + filename + "GS_VIAEHL.dat";
	string file_name_BAO = "data_tables/number_of_states/" + filename + "GS_BAO.dat";
	string file_name_BAON = "data_tables/number_of_states/" + filename + "GS_BAON.dat";

	// The varying parameters
	int S_starting_value = 100;
	int S_finishing_value = endP;
	int S_increment = 100;
	// S_finishing_value = 2000;
	// hardcoded parameter
	double action_prob = 1.0;
	// A_num=100;
	// write meta data to all stringstreams as first in their respective files

	write_meta_data_to_dat_file_number_of_states(stringstream_VI, Actions, epsilon, gamma, upper_reward, non_zero_transition, action_prob, S_starting_value, S_finishing_value, S_increment);
	write_meta_data_to_dat_file_number_of_states(stringstream_VIU, Actions, epsilon, gamma, upper_reward, non_zero_transition, action_prob, S_starting_value, S_finishing_value, S_increment);
	write_meta_data_to_dat_file_number_of_states(stringstream_BVI, Actions, epsilon, gamma, upper_reward, non_zero_transition, action_prob, S_starting_value, S_finishing_value, S_increment);
	write_meta_data_to_dat_file_number_of_states(stringstream_VIH, Actions, epsilon, gamma, upper_reward, non_zero_transition, action_prob, S_starting_value, S_finishing_value, S_increment);
	write_meta_data_to_dat_file_number_of_states(stringstream_VIAE, Actions, epsilon, gamma, upper_reward, non_zero_transition, action_prob, S_starting_value, S_finishing_value, S_increment);
	write_meta_data_to_dat_file_number_of_states(stringstream_VIAEH, Actions, epsilon, gamma, upper_reward, non_zero_transition, action_prob, S_starting_value, S_finishing_value, S_increment);
	float VI[10][20];
	int S2 = 0;
	int k = 0;
	// int S=500;
	// int S=1000;
	int S = 500;
	// A_num=100;
	S = 500;
	S2 = S / 10;
	S2 = 100;
	// A_num=100;
	S = 100;
	S2 = 10;
	R_type R1;
	A_type Aa1;
	P_type P1;
	auto MDP = make_tuple(R1, Aa1, P1);
	for (int iters = 0; iters < 2; iters++)
	{
		k = 0;
		for (int ite = StartP; ite <= endP; ite = ite + IncP)
		{
			// printf("Beginning iteration %d  S2,  %d, A, S %d = %d\n",iters, S2,A_num,S);
			// auto MDP ;
			// GENERATE THE MDP
			int seed = time(0);
			// auto MDP = generate_random_MDP_exponential_distributed_rewards(S, A_num, 1.0, S, 0.02, seed);

			if (expnum == 1)
			{
				States = ite;
				MDP = generate_random_MDP_normal_distributed_rewards(ite, Actions, action_prob, SS, seed, 1000, 10);
			}
			else if (expnum == 2)
				MDP = generate_random_MDP_normal_distributed_rewards(States, ite, action_prob, SS, seed, 1000, 10);
			else
				MDP = generate_random_MDP_normal_distributed_rewards(States, Actions, action_prob, ite, seed, 1000, 10);

			//
			// int xs=90;
			// auto MDP=Maze(xs,xs,seed);
			// S=xs*xs+1;
			// auto MDP = generate_random_MDP_with_variable_parameters(S, A_num, action_prob, non_zero_transition, upper_reward, seed);
			R_type R = get<0>(MDP);
			A_type A = get<1>(MDP);
			P_type P = get<2>(MDP);
			int counter = 0;
			/*
			for (int t = 0; t < S; t++) {
					for (auto a : A[t]) {
							auto& [P_s_a, P_s_a_nonzero] = P[t][a];
								for (int k : P_s_a_nonzero) {
									//cout<<P_s_a[counter]<<"PSA"<<P_s_a_nonzero[counter]<<"P_s_a_nonzero"<< R[t][a]  <<"Rew"<<a<<"A"<<endl;
									counter++;
									}
									counter=0;

							}
					}*/
			// cout<<"MDP"<<endl;
			// VI testing
			// TODO: make method that takes arguments and performs the writing of data, takes filename, MDP etc. as argument
			A_type A1 = copy_A(A);
			auto start_VI = high_resolution_clock::now();
			// cout<<getValue()<<endl;
			V_type V_approx_solution_tuple = value_iterationGS(States, R, A1, P, gamma, epsilon);
			vector<double> V_approx_solution = get<0>(V_approx_solution_tuple);
			auto stop_VI = high_resolution_clock::now();
			auto duration_VI = duration_cast<milliseconds>(stop_VI - start_VI);

			stringstream_VI << to_string(S) << " " << duration_VI.count() << endl;
			VI[0][k] += duration_VI.count();
			// VIU testing
			A_type A6 = copy_A(A);
			auto start_VIU = high_resolution_clock::now();
			V_type V_approx_solution_upper_tuple = value_iteration_upperGS(States, R, A6, P, gamma, epsilon);
			vector<double> V_approx_solution_upper = get<0>(V_approx_solution_upper_tuple);
			auto stop_VIU = high_resolution_clock::now();
			auto duration_VIU = duration_cast<milliseconds>(stop_VIU - start_VIU);

			VI[1][k] += duration_VIU.count();
			cout << "VIU," << duration_VIU.count() << endl;
			stringstream_VIU << to_string(S) << " " << duration_VIU.count() << endl;

			// VIH testing
			A_type A2 = copy_A(A);
			auto start_VIH = high_resolution_clock::now();

			V_type V_heap_approx_tuple = value_iteration_with_heapGS(States, R, A2, P, gamma, epsilon);
			vector<double> V_heap_approx = get<0>(V_heap_approx_tuple);
			auto stop_VIH = high_resolution_clock::now();
			auto duration_VIH = duration_cast<milliseconds>(stop_VIH - start_VIH);
			VI[2][k] += duration_VIH.count();
			cout << "VIHGS," << duration_VIH.count() << endl;
			stringstream_VIH << to_string(S) << " " << duration_VIH.count() << endl;
			// A_type A12 = copy_A(A);
			auto start_VIHN = high_resolution_clock::now();

			// V_type V_heap_approx_tupleN = value_iteration_VIH_custom(S, R, A12, P, gamma, epsilon);
			// vector<double> V_heap_approxN = get<0>(V_heap_approx_tuple);

			auto stop_VIHN = high_resolution_clock::now();
			auto duration_VIHN = duration_cast<milliseconds>(stop_VIHN - start_VIHN);

			// stringstream_VIHN << to_string(A_num) << " " << duration_VIHN.count() << endl;
			VI[9][k] += duration_VIHN.count();

			// BVI
			A_type A3 = copy_A(A);
			auto start_BVI = high_resolution_clock::now();

			V_type V_bounded_approx_solution_tuple = bounded_value_iterationGS(States, R, A3, P, gamma, epsilon);
			vector<double> V_bounded_approx_solution = get<0>(V_bounded_approx_solution_tuple);
			auto stop_BVI = high_resolution_clock::now();
			auto duration_BVI = duration_cast<milliseconds>(stop_BVI - start_BVI);

			stringstream_BVI << to_string(S) << " " << duration_BVI.count() << endl;
			VI[3][k] += duration_BVI.count();
			A_type A4 = copy_A(A);
			auto start_VIAE = high_resolution_clock::now();
			V_type V_AE_approx_solution_tuple = value_iteration_action_eliminationGS(States, R, A4, P, gamma, epsilon);
			vector<double> V_AE_approx_solution = get<0>(V_AE_approx_solution_tuple);

			auto stop_VIAE = high_resolution_clock::now();
			auto duration_VIAE = duration_cast<milliseconds>(stop_VIAE - start_VIAE);

			stringstream_VIAE << to_string(S) << " " << duration_VIAE.count() << endl;
			VI[4][k] += duration_VIAE.count();
			// VIAEH
			A_type A5 = copy_A(A);
			auto start_VIAEH = high_resolution_clock::now();

			V_type V_AE_H_approx_solution_tuple = value_iteration_action_elimination_heapsGS(States, R, A5, P, gamma, epsilon);
			vector<double> V_AE_H_approx_solution = get<0>(V_AE_H_approx_solution_tuple);

			auto stop_VIAEH = high_resolution_clock::now();
			auto duration_VIAEH = duration_cast<milliseconds>(stop_VIAEH - start_VIAEH);

			stringstream_VIAEH << to_string(S) << " " << duration_VIAEH.count() << endl;
			VI[5][k] += duration_VIAEH.count();
			// VIAEHL
			A_type A8 = copy_A(A);
			auto start_VIAEHL = high_resolution_clock::now();

			V_type VIAEHL_approx_solution_tuple = value_iteration_action_elimination_heaps_lower_bound_approxGS(States, R, A8, P, gamma, epsilon);
			vector<double> VIAEHL_approx_solution = get<0>(VIAEHL_approx_solution_tuple);

			auto stop_VIAEHL = high_resolution_clock::now();
			auto duration_VIAEHL = duration_cast<milliseconds>(stop_VIAEHL - start_VIAEHL);
			stringstream_VIAEHL << to_string(S) << " " << duration_VIAEHL.count() << endl;
			VI[6][k] += duration_VIAEHL.count();
			// BAO
			A_type A9 = copy_A(A);
			auto start_BAO = high_resolution_clock::now();

			V_type BAO_approx_solution_tuple = value_iteration_BAOGS(States, R, A9, P, gamma, epsilon);
			vector<double> BAO_approx_solution = get<0>(BAO_approx_solution_tuple);

			auto stop_BAO = high_resolution_clock::now();
			auto duration_BAO = duration_cast<milliseconds>(stop_BAO - start_BAO);

			stringstream_BAO << to_string(S) << " " << duration_BAO.count() << endl;
			VI[7][k] += duration_BAO.count();

			A_type A10 = copy_A(A);
			auto start_BAOSK = high_resolution_clock::now();

			// V_type BAO_approx_solution_tuple1 = value_iteration_BAOSKGS(States, R, A10, P, gamma, epsilon);
			// vector<double> BAO_approx_solution1 = get<0>(BAO_approx_solution_tuple1);
			auto stop_BAOSK = high_resolution_clock::now();

			auto duration_BAOSK = duration_cast<milliseconds>(stop_BAOSK - start_BAOSK);
			stringstream_BAON << to_string(S) << " " << duration_BAOSK.count() << endl;
			VI[8][k] += duration_BAOSK.count();
			// They should in theory all be epsilon from true value, and therefore, at most be 2 * epsilon from each other

			if (abs_max_diff_vectors(V_approx_solution, BAO_approx_solution) > (2 * epsilon))
			{
				printf("DIFFERENCE1\n");
			}
			if (abs_max_diff_vectors(V_approx_solution, V_heap_approx) > (2 * epsilon))
			{
				printf("DIFFERENCE2\n");
			}

			if (abs_max_diff_vectors(V_approx_solution, V_AE_H_approx_solution) > (2 * epsilon))
			{
				printf("DIFFERENCE3\n");
			}
			if (abs_max_diff_vectors(V_approx_solution, V_approx_solution_upper) > (2 * epsilon))
			{
				printf("DIFFERENCE4\n");
			}
			if (abs_max_diff_vectors(V_approx_solution, V_bounded_approx_solution) > (2 * epsilon))
			{
				printf("DIFFERENCE5\n");
			}
			if (abs_max_diff_vectors(V_approx_solution, VIAEHL_approx_solution) > (2 * epsilon))
			{
				printf("DIFFERENCE6\n");
			}
			if (abs_max_diff_vectors(V_approx_solution, V_AE_approx_solution) > (2 * epsilon))
			{
				printf("DIFFERENCE7\n");
			}
			if (abs_max_diff_vectors(V_approx_solution, BAO_approx_solution) > (2 * epsilon))
			{
				printf("DIFFERENCE8\n");
			}

			k++;
		}
	}
	for (int k = 0; k < 20; k++)
	{
		stringstream_VI << VI[0][k] / 5 << endl;
		stringstream_VIU << VI[1][k] / 5 << endl;
		stringstream_VIH << VI[2][k] / 5 << endl;
		stringstream_VIHN << VI[9][k] / 5 << endl;
		stringstream_BVI << VI[3][k] / 5 << endl;
		stringstream_VIAE << VI[4][k] / 5 << endl;
		stringstream_VIAEH << VI[5][k] / 5 << endl;
		stringstream_VIAEHL << VI[6][k] / 5 << endl;
		stringstream_BAO << VI[7][k] / 5 << endl;
		stringstream_BAON << VI[8][k] / 5 << endl;
		// cout<<"writeA"<<endl;
	}
	// WRITE ALL DATA TO THEIR RESPECTVIE FILES
	write_stringstream_to_file(stringstream_VI, output_stream_VI, file_name_VI);
	write_stringstream_to_file(stringstream_VIU, output_stream_VIU, file_name_VIU);
	write_stringstream_to_file(stringstream_BVI, output_stream_BVI, file_name_BVI);
	write_stringstream_to_file(stringstream_VIH, output_stream_VIH, file_name_VIH);
	write_stringstream_to_file(stringstream_VIHN, output_stream_VIHN, file_name_VIHN);
	write_stringstream_to_file(stringstream_VIAE, output_stream_VIAE, file_name_VIAE);
	write_stringstream_to_file(stringstream_VIAEH, output_stream_VIAEH, file_name_VIAEH);
	write_stringstream_to_file(stringstream_VIAEHL, output_stream_VIAEHL, file_name_VIAEHL);
	write_stringstream_to_file(stringstream_BAO, output_stream_BAO, file_name_BAO);
	write_stringstream_to_file(stringstream_BAON, output_stream_BAON, file_name_BAON);
	// cout<<"writeF"<<endl;
}

void create_data_tables_number_GSBO(string filename, int expnum, int States, int Actions, int SS, int startP, int endP, int IncP, double epsilon, double gamma, double upper_reward, double non_zero_transition)
{

	// the stringstreams to create the test for the files

	ostringstream stringstream_VIH;
	ostringstream stringstream_VIAEHL;
	ostringstream stringstream_BAON;

	// the file output objects
	ofstream output_stream_VIH;
	ofstream output_stream_VIAEHL;
	ofstream output_stream_BAON;

	// set the name of the file to write to

	string file_name_VIH = "data_tables/number_of_states_best/" + filename + "GS_VIHBO.dat";
	string file_name_VIAEHL = "data_tables/number_of_states_best/" + filename + "GS_VIAEHLBO.dat";
	string file_name_BAON = "data_tables/number_of_states_best/" + filename + "GS_BAONBO.dat";

	// The varying parameters

	// hardcoded parameter
	double action_prob = 1.0;
	// A_num=100;
	// write meta data to all stringstreams as first in their respective files

	write_meta_data_to_dat_file_number_of_states(stringstream_VIH, Actions, epsilon, gamma, upper_reward, non_zero_transition, action_prob, startP, endP, IncP);
	write_meta_data_to_dat_file_number_of_states(stringstream_BAON, Actions, epsilon, gamma, upper_reward, non_zero_transition, action_prob, startP, endP, IncP);
	write_meta_data_to_dat_file_number_of_states(stringstream_VIAEHL, Actions, epsilon, gamma, upper_reward, non_zero_transition, action_prob, startP, endP, IncP);
	float VI[10][20];
	int k = 0;
	R_type R1;
	A_type Aa1;
	P_type P1;
	auto MDP = make_tuple(R1, Aa1, P1);
	for (int iters = 0; iters < 2; iters++)
	{
		k = 0;
		for (int ite = startP; ite <= endP; ite = ite + IncP)
		{
			// printf("Beginning iteration %d  S2,  %d, A, S %d = %d\n",iters, S2,A_num,S);

			// GENERATE THE MDP
			int seed = time(0);
			// auto MDP = generate_random_MDP_exponential_distributed_rewards(S, A_num, 1.0, S, 0.02, seed);
			if (expnum == 1)
			{
				MDP = generate_random_MDP_normal_distributed_rewards(ite, Actions, action_prob, SS, seed, 1000, 10);
				States = ite;
			}
			else if (expnum == 2)
				MDP = generate_random_MDP_normal_distributed_rewards(States, ite, action_prob, SS, seed, 1000, 10);
			else
				MDP = generate_random_MDP_normal_distributed_rewards(States, Actions, action_prob, ite, seed, 1000, 10);

			// auto MDP = RiverSwim(S);
			// int xs=90;
			// auto MDP=Maze(xs,xs,seed);
			// S=xs*xs+1;
			// auto MDP = generate_random_MDP_with_variable_parameters(S, A_num, action_prob, non_zero_transition, upper_reward, seed);
			R_type R = get<0>(MDP);
			A_type A = get<1>(MDP);
			P_type P = get<2>(MDP);
			int counter = 0;
			/*
			for (int t = 0; t < S; t++) {
					for (auto a : A[t]) {
							auto& [P_s_a, P_s_a_nonzero] = P[t][a];
								for (int k : P_s_a_nonzero) {
									//cout<<P_s_a[counter]<<"PSA"<<P_s_a_nonzero[counter]<<"P_s_a_nonzero"<< R[t][a]  <<"Rew"<<a<<"A"<<endl;
									counter++;
									}
									counter=0;

							}
					}*/
			cout << "MDP" << endl;
			// VI testing
			// TODO: make method that takes arguments and performs the writing of data, takes filename, MDP etc. as argument

			// VIH testing
			A_type A2 = copy_A(A);
			auto start_VIH = high_resolution_clock::now();
			V_type V_heap_approx_tuple = value_iteration_with_heapGS(States, R, A2, P, gamma, epsilon);
			vector<double> V_heap_approx = get<0>(V_heap_approx_tuple);
			auto stop_VIH = high_resolution_clock::now();

			auto duration_VIH = duration_cast<milliseconds>(stop_VIH - start_VIH);
			VI[2][k] += duration_VIH.count();
			cout << "VIHGS," << duration_VIH.count() << endl;
			stringstream_VIH << to_string(States) << " " << duration_VIH.count() << endl;
			A_type A8 = copy_A(A);

			auto start_VIAEHL = high_resolution_clock::now();

			V_type VIAEHL_approx_solution_tuple = value_iteration_action_elimination_heaps_lower_bound_approxGS(States, R, A8, P, gamma, epsilon);
			vector<double> VIAEHL_approx_solution = get<0>(VIAEHL_approx_solution_tuple);

			auto stop_VIAEHL = high_resolution_clock::now();
			auto duration_VIAEHL = duration_cast<milliseconds>(stop_VIAEHL - start_VIAEHL);
			stringstream_VIAEHL << to_string(States) << " " << duration_VIAEHL.count() << endl;
			VI[6][k] += duration_VIAEHL.count();
			// BAO

			A_type A10 = copy_A(A);
			auto start_BAOSK = high_resolution_clock::now();

			V_type BAO_approx_solution_tuple1 = value_iteration_BAOGS(States, R, A10, P, gamma, epsilon);
			vector<double> BAO_approx_solution1 = get<0>(BAO_approx_solution_tuple1);
			auto stop_BAOSK = high_resolution_clock::now();

			auto duration_BAOSK = duration_cast<milliseconds>(stop_BAOSK - start_BAOSK);
			stringstream_BAON << to_string(States) << " " << duration_BAOSK.count() << endl;
			VI[8][k] += duration_BAOSK.count();
			// They should in theory all be epsilon from true value, and therefore, at most be 2 * epsilon from each other

			if (abs_max_diff_vectors(V_heap_approx, VIAEHL_approx_solution) > (2 * epsilon))
			{
				printf("DIFFERENCE6\n");
			}
			if (abs_max_diff_vectors(BAO_approx_solution1, V_heap_approx) > (2 * epsilon))
			{
				printf("DIFFERENCE7\n");
			}

			k++;
		}
	}
	for (int k = 0; k < 20; k++)
	{
		stringstream_VIH << VI[2][k] / 5 << endl;
		stringstream_VIAEHL << VI[6][k] / 5 << endl;
		stringstream_BAON << VI[8][k] / 5 << endl;
		cout << "writeA" << endl;
	}
	// WRITE ALL DATA TO THEIR RESPECTVIE FILES

	write_stringstream_to_file(stringstream_VIH, output_stream_VIH, file_name_VIH);
	write_stringstream_to_file(stringstream_VIAEHL, output_stream_VIAEHL, file_name_VIAEHL);
	write_stringstream_to_file(stringstream_BAON, output_stream_BAON, file_name_BAON);
	cout << "writeF" << endl;
}

void create_data_tables_number_GSTM(string filename, int expnum, int States, int Actions, int SS, int StartP, int endP, int IncP, double epsilon, double gamma, double upper_reward, double non_zero_transition)
{

	// the stringstreams to create the test for the files
	ostringstream stringstream_BVI;
	ostringstream stringstream_VI;
	ostringstream stringstream_VIU;
	ostringstream stringstream_VIH;
	ostringstream stringstream_VIHN;
	ostringstream stringstream_VIAE;
	ostringstream stringstream_VIAEH;
	ostringstream stringstream_VIAEHL;
	ostringstream stringstream_BAO;
	ostringstream stringstream_BAON;

	// the file output objects
	ofstream output_stream_BVI;
	ofstream output_stream_VI;
	ofstream output_stream_VIU;
	ofstream output_stream_VIH;
	ofstream output_stream_VIHN;
	ofstream output_stream_VIAE;
	ofstream output_stream_VIAEH;
	ofstream output_stream_VIAEHL;
	ofstream output_stream_BAO;
	ofstream output_stream_BAON;

	// set the name of the file to write to
	string file_name_BVI = "data_tables/number_of_states/" + filename + "TMGS_BVI.dat";
	string file_name_VI = "data_tables/number_of_states/" + filename + "TMGS_VI.dat";
	string file_name_VIU = "data_tables/number_of_states/" + filename + "TMGS_VIU.dat";
	string file_name_VIH = "data_tables/number_of_states/" + filename + "TMGS_VIH.dat";
	string file_name_VIHN = "data_tables/number_of_states/" + filename + "TMGS_VIHN.dat";
	string file_name_VIAE = "data_tables/number_of_states/" + filename + "TMGS_VIAE.dat";
	string file_name_VIAEH = "data_tables/number_of_states/" + filename + "TMGS_VIAEH.dat";
	string file_name_VIAEHL = "data_tables/number_of_states/" + filename + "TMGS_VIAEHL.dat";
	string file_name_BAO = "data_tables/number_of_states/" + filename + "TMGS_BAO.dat";
	string file_name_BAON = "data_tables/number_of_states/" + filename + "TMGS_BAON.dat";

	// The varying parameters
	int S_starting_value = 100;
	int S_finishing_value = endP;
	int S_increment = 100;
	// S_finishing_value = 2000;
	// hardcoded parameter
	double action_prob = 1.0;
	// A_num=100;
	// write meta data to all stringstreams as first in their respective files

	write_meta_data_to_dat_file_number_of_states(stringstream_VI, Actions, epsilon, gamma, upper_reward, non_zero_transition, action_prob, S_starting_value, S_finishing_value, S_increment);
	write_meta_data_to_dat_file_number_of_states(stringstream_VIU, Actions, epsilon, gamma, upper_reward, non_zero_transition, action_prob, S_starting_value, S_finishing_value, S_increment);
	write_meta_data_to_dat_file_number_of_states(stringstream_BVI, Actions, epsilon, gamma, upper_reward, non_zero_transition, action_prob, S_starting_value, S_finishing_value, S_increment);
	write_meta_data_to_dat_file_number_of_states(stringstream_VIH, Actions, epsilon, gamma, upper_reward, non_zero_transition, action_prob, S_starting_value, S_finishing_value, S_increment);
	write_meta_data_to_dat_file_number_of_states(stringstream_VIAE, Actions, epsilon, gamma, upper_reward, non_zero_transition, action_prob, S_starting_value, S_finishing_value, S_increment);
	write_meta_data_to_dat_file_number_of_states(stringstream_VIAEH, Actions, epsilon, gamma, upper_reward, non_zero_transition, action_prob, S_starting_value, S_finishing_value, S_increment);
	int siIter = ((endP - StartP) / IncP) + 1;
	float VI[10][siIter];
	int k = 0;
	R_type R1;
	A_type Aa1;
	P_type P1;
	int S;
	auto MDP = make_tuple(R1, Aa1, P1);
	for (int iters = 0; iters < 10; iters++)
	{
		k = 0;
		for (int ite = StartP; ite <= endP; ite = ite + IncP)
		{
			// printf("Beginning iteration %d  S2,  %d, A, S %d = %d\n",iters, S2,A_num,S);
			// auto MDP ;
			// GENERATE THE MDP
			int seed = time(0);
			// auto MDP = generate_random_MDP_exponential_distributed_rewards(S, A_num, 1.0, S, 0.02, seed);
			cout << "ite" << ite << endl;
			cout << "iters" << iters << endl;
			if (expnum == 1)
			{

				// auto MDP=Maze(xs,xs,seed);
				MDP = Maze(ite, ite, seed);
				States = ite * ite + 1;
				S = States;
			}
			else if (expnum == 2)
			{
				MDP = Maze3d(ite, ite, ite, seed);
				States = ite * ite * ite + 1;
				S = States;
			}
			// auto MDP = generate_random_MDP_with_variable_parameters(S, A_num, action_prob, non_zero_transition, upper_reward, seed);
			R_type R = get<0>(MDP);
			A_type A = get<1>(MDP);
			P_type P = get<2>(MDP);
			int counter = 0;
			/*
			for (int t = 0; t < S; t++) {
					for (auto a : A[t]) {
							auto& [P_s_a, P_s_a_nonzero] = P[t][a];
								for (int k : P_s_a_nonzero) {
									//cout<<P_s_a[counter]<<"PSA"<<P_s_a_nonzero[counter]<<"P_s_a_nonzero"<< R[t][a]  <<"Rew"<<a<<"A"<<endl;
									counter++;
									}
									counter=0;

							}
					}*/
			// cout<<"MDP"<<endl;
			// VI testing
			// TODO: make method that takes arguments and performs the writing of data, takes filename, MDP etc. as argument
			A_type A1 = copy_A(A);
			auto start_VI = high_resolution_clock::now();
			// cout<<getValue()<<endl;
			V_type V_approx_solution_tuple = value_iterationGSTM(States, R, A1, P, gamma, epsilon, expnum);
			vector<double> V_approx_solution = get<0>(V_approx_solution_tuple);
			auto stop_VI = high_resolution_clock::now();
			auto duration_VI = duration_cast<milliseconds>(stop_VI - start_VI);

			stringstream_VI << to_string(States) << " " << duration_VI.count() << endl;
			VI[0][k] += duration_VI.count();
			// cout << "VI," << duration_VI.count() << endl;
			//  VIU testing
			A_type A6 = copy_A(A);
			auto start_VIU = high_resolution_clock::now();
			V_type V_approx_solution_upper_tuple = value_iteration_upperGSTM(States, R, A6, P, gamma, epsilon, expnum);
			vector<double> V_approx_solution_upper = get<0>(V_approx_solution_upper_tuple);
			auto stop_VIU = high_resolution_clock::now();
			auto duration_VIU = duration_cast<milliseconds>(stop_VIU - start_VIU);

			VI[1][k] += duration_VIU.count();
			// cout << "VIU," << duration_VIU.count() << endl;
			stringstream_VIU << to_string(States) << " " << duration_VIU.count() << endl;
			// VIH testing
			A_type A2 = copy_A(A);
			auto start_VIH = high_resolution_clock::now();

			V_type V_heap_approx_tuple = value_iteration_with_heapGSTM(States, R, A2, P, gamma, epsilon, expnum);
			vector<double> V_heap_approx = get<0>(V_heap_approx_tuple);
			auto stop_VIH = high_resolution_clock::now();
			auto duration_VIH = duration_cast<milliseconds>(stop_VIH - start_VIH);
			VI[2][k] += duration_VIH.count();
			// cout << "VIHGS," << duration_VIH.count() << endl;
			stringstream_VIH << to_string(States) << " " << duration_VIH.count() << endl;
			// A_type A12 = copy_A(A);
			auto start_VIHN = high_resolution_clock::now();

			// V_type V_heap_approx_tupleN = value_iteration_VIH_custom(S, R, A12, P, gamma, epsilon);
			// vector<double> V_heap_approxN = get<0>(V_heap_approx_tuple);

			auto stop_VIHN = high_resolution_clock::now();
			auto duration_VIHN = duration_cast<milliseconds>(stop_VIHN - start_VIHN);

			// stringstream_VIHN << to_string(A_num) << " " << duration_VIHN.count() << endl;
			VI[9][k] += duration_VIHN.count();

			// BVI
			A_type A3 = copy_A(A);
			auto start_BVI = high_resolution_clock::now();

			V_type V_bounded_approx_solution_tuple = bounded_value_iterationGSTM(States, R, A3, P, gamma, epsilon, expnum);
			vector<double> V_bounded_approx_solution = get<0>(V_bounded_approx_solution_tuple);
			auto stop_BVI = high_resolution_clock::now();
			auto duration_BVI = duration_cast<milliseconds>(stop_BVI - start_BVI);
			// cout << "BVI," << duration_BVI.count() << endl;
			stringstream_BVI << to_string(States) << " " << duration_BVI.count() << endl;
			VI[3][k] += duration_BVI.count();
			A_type A4 = copy_A(A);
			auto start_VIAE = high_resolution_clock::now();
			V_type V_AE_approx_solution_tuple = value_iteration_action_eliminationGSTM(States, R, A4, P, gamma, epsilon, expnum);
			vector<double> V_AE_approx_solution = get<0>(V_AE_approx_solution_tuple);

			auto stop_VIAE = high_resolution_clock::now();
			auto duration_VIAE = duration_cast<milliseconds>(stop_VIAE - start_VIAE);

			stringstream_VIAE << to_string(S) << " " << duration_VIAE.count() << endl;
			VI[4][k] += duration_VIAE.count();
			// cout << "VIAE," << duration_VIAE.count() << endl;
			//  VIAEH
			A_type A5 = copy_A(A);
			auto start_VIAEH = high_resolution_clock::now();

			V_type V_AE_H_approx_solution_tuple = value_iteration_action_elimination_heapsGSTM(States, R, A5, P, gamma, epsilon, expnum);
			vector<double> V_AE_H_approx_solution = get<0>(V_AE_H_approx_solution_tuple);

			auto stop_VIAEH = high_resolution_clock::now();
			auto duration_VIAEH = duration_cast<milliseconds>(stop_VIAEH - start_VIAEH);

			stringstream_VIAEH << to_string(S) << " " << duration_VIAEH.count() << endl;
			VI[5][k] += duration_VIAEH.count();
			// cout << "VIAEH," << duration_VIAEH.count() << endl;
			//  VIAEHL
			A_type A8 = copy_A(A);
			auto start_VIAEHL = high_resolution_clock::now();

			V_type VIAEHL_approx_solution_tuple = value_iteration_action_elimination_heaps_lower_bound_approxGSTM(States, R, A8, P, gamma, epsilon, expnum);
			vector<double> VIAEHL_approx_solution = get<0>(VIAEHL_approx_solution_tuple);

			auto stop_VIAEHL = high_resolution_clock::now();
			auto duration_VIAEHL = duration_cast<milliseconds>(stop_VIAEHL - start_VIAEHL);
			stringstream_VIAEHL << to_string(S) << " " << duration_VIAEHL.count() << endl;
			VI[6][k] += duration_VIAEHL.count();
			// cout << "VIAEHL," << duration_VIAEHL.count() << endl;
			//  BAO
			A_type A9 = copy_A(A);
			auto start_BAO = high_resolution_clock::now();

			V_type BAO_approx_solution_tuple = value_iteration_BAOGSTM(States, R, A9, P, gamma, epsilon, expnum);
			vector<double> BAO_approx_solution = get<0>(BAO_approx_solution_tuple);

			auto stop_BAO = high_resolution_clock::now();
			auto duration_BAO = duration_cast<milliseconds>(stop_BAO - start_BAO);

			stringstream_BAO << to_string(S) << " " << duration_BAO.count() << endl;
			// cout << "BAO," << duration_BAO.count() << endl;
			VI[7][k] += duration_BAO.count();

			A_type A10 = copy_A(A);
			auto start_BAOSK = high_resolution_clock::now();

			// V_type BAO_approx_solution_tuple1 = value_iteration_BAOSKGS(States, R, A10, P, gamma, epsilon);
			// vector<double> BAO_approx_solution1 = get<0>(BAO_approx_solution_tuple1);
			auto stop_BAOSK = high_resolution_clock::now();

			auto duration_BAOSK = duration_cast<milliseconds>(stop_BAOSK - start_BAOSK);
			stringstream_BAON << to_string(S) << " " << duration_BAOSK.count() << endl;
			VI[8][k] += duration_BAOSK.count();
			// They should in theory all be epsilon from true value, and therefore, at most be 2 * epsilon from each other

			if (abs_max_diff_vectors(V_approx_solution, BAO_approx_solution) > (2 * epsilon))
			{
				printf("DIFFERENCE1\n");
			}
			if (abs_max_diff_vectors(V_approx_solution, V_heap_approx) > (2 * epsilon))
			{
				printf("DIFFERENCE2\n");
			}

			if (abs_max_diff_vectors(V_approx_solution, V_AE_H_approx_solution) > (2 * epsilon))
			{
				printf("DIFFERENCE3\n");
			}
			if (abs_max_diff_vectors(V_approx_solution, V_approx_solution_upper) > (2 * epsilon))
			{
				printf("DIFFERENCE4\n");
			}
			if (abs_max_diff_vectors(V_approx_solution, V_bounded_approx_solution) > (2 * epsilon))
			{
				printf("DIFFERENCE5\n");
			}
			if (abs_max_diff_vectors(V_approx_solution, VIAEHL_approx_solution) > (2 * epsilon))
			{
				printf("DIFFERENCE6\n");
			}
			if (abs_max_diff_vectors(V_approx_solution, V_AE_approx_solution) > (2 * epsilon))
			{
				printf("DIFFERENCE7\n");
			}
			if (abs_max_diff_vectors(V_approx_solution, BAO_approx_solution) > (2 * epsilon))
			{
				printf("DIFFERENCE8\n");
			}

			k++;
		}
	}
	for (int k = 0; k < siIter; k++)
	{
		stringstream_VI << VI[0][k] / 10 << endl;
		stringstream_VIU << VI[1][k] / 10 << endl;
		stringstream_VIH << VI[2][k] / 10 << endl;
		stringstream_VIHN << VI[9][k] / 10 << endl;
		stringstream_BVI << VI[3][k] / 10 << endl;
		stringstream_VIAE << VI[4][k] / 10 << endl;
		stringstream_VIAEH << VI[5][k] / 10 << endl;
		stringstream_VIAEHL << VI[6][k] / 10 << endl;
		stringstream_BAO << VI[7][k] / 10 << endl;
		stringstream_BAON << VI[8][k] / 10 << endl;
		// cout<<"writeA"<<endl;
	}
	// WRITE ALL DATA TO THEIR RESPECTVIE FILES
	write_stringstream_to_file(stringstream_VI, output_stream_VI, file_name_VI);
	write_stringstream_to_file(stringstream_VIU, output_stream_VIU, file_name_VIU);
	write_stringstream_to_file(stringstream_BVI, output_stream_BVI, file_name_BVI);
	write_stringstream_to_file(stringstream_VIH, output_stream_VIH, file_name_VIH);
	write_stringstream_to_file(stringstream_VIHN, output_stream_VIHN, file_name_VIHN);
	write_stringstream_to_file(stringstream_VIAE, output_stream_VIAE, file_name_VIAE);
	write_stringstream_to_file(stringstream_VIAEH, output_stream_VIAEH, file_name_VIAEH);
	write_stringstream_to_file(stringstream_VIAEHL, output_stream_VIAEHL, file_name_VIAEHL);
	write_stringstream_to_file(stringstream_BAO, output_stream_BAO, file_name_BAO);
	write_stringstream_to_file(stringstream_BAON, output_stream_BAON, file_name_BAON);
	// cout<<"writeF"<<endl;
}