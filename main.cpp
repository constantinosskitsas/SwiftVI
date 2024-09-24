#include <algorithm>
#include <thread>
#include <queue>
#include <chrono>
#include <tuple>
#include <random>
#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <sstream>

// My header files
#include "MDP_type_definitions.h"
#include "pretty_printing_MDP.h"
#include "MDP_generation.h"
#include "VI_algorithms_helper_methods.h"
#include "VI_algorithm.h"
#include "VIU_algorithm.h"
#include "BVI_algorithm.h"
#include "VIAE_algorithm.h"
#include "VIAEH_algorithm.h"
#include "VIAEH_algorithm_no_pointers.h"
#include "VIAEH_algorithm_maxmin_heap.h"
#include "VIAEH_algorithm_lazy_update.h"
#include "VIAEH_algorithm_set.h"
#include "VIH_algorithm.h"
#include "experiments.h"
#include "stopping_criteria_plot.h"
#include "VIH_algorithm_custom_heaps.h"
#include "VIAE_algorithm_improved_bounds.h"
#include "VIAE_algorithm_old_bounds.h"
#include "VIAEH_algorithm_lower_bound_approx.h"
#include "top_action_change_plot.h"
#include "VIH_actions_touched.h"
#include "BAO_algorithm.h"

using namespace std;
using namespace std::chrono;
// terminal compilation and running: g++ -pthread -std=gnu++17 -o algo_test *.cpp && ./algo_test
int main(int argc, char *argv[])
{
	//std::cout << "running MBIE" << std::endl;
	//runMBIE();
	//std::cout << "Back to normal" << std::endl;
	time_t time_now = time(0);
	double epsilon = 0.05;
	double gamma = 0.99;
	int S = 100;
	int A_num = 50;
	double upper_reward = 1000.0;
	double action_prob = 1.0;		  // alpha
	double non_zero_transition = 0.5; // beta

	int expnum = 3;
	int NOFexp = 3;
	int States = 500;
	int Actions = 100;
	int SS = 50;
	int StartP = 100;  //7 for 40 S grid
	int EndP = 500;
	int IncP = 100;
	std::size_t pos;

	string file_prefix_number_of_states_best = "number_of_states_best";
	string file_prefix_actions_touched = "";
	string file_prefix_number_of_states = "";
	int number_of_transitions = 0;
	double mean = 0;
	double variance = 0;
	NOFexp = 50;
	if (NOFexp == 1 || NOFexp == 2)
	{
		VMS(NOFexp, epsilon, gamma);
	}
	else if (NOFexp >= 3 && NOFexp <= 8)
	{
		// expnum=3 States 4 BO, 5 Actions B0, 6 Supported States 7B0
		REXP(file_prefix_number_of_states, NOFexp, States, Actions, SS, StartP, EndP, IncP, epsilon, gamma, upper_reward, non_zero_transition);
	}
	// else if (NOFexp == 5)
	//	create_data_tables_actions_touched(file_prefix_actions_touched, S, A_num, epsilon, gamma, action_prob, number_of_transitions, mean, variance);
	else if (NOFexp == 9 || NOFexp == 10 || NOFexp == 11)
	{
		GSTM(file_prefix_number_of_states, NOFexp, States, Actions, SS, StartP, EndP, IncP, epsilon, gamma, upper_reward, non_zero_transition);
	}else if (NOFexp==12){
		RLRS(file_prefix_number_of_states, NOFexp, States, Actions, SS, StartP, EndP, IncP, epsilon, gamma, upper_reward, non_zero_transition);
	}else if (NOFexp>12){
		REXP_temp(file_prefix_number_of_states, NOFexp/10, States, Actions, SS, StartP, EndP, IncP, epsilon, gamma, upper_reward, non_zero_transition);
	}


}

int main_Emil(int argc, char *argv[])
{

	time_t time_now = time(0);

	double epsilon = 0.05;
	double gamma = 0.99;

	int S = 100;
	int A_num = 50;
	double upper_reward = 1000.0;
	double action_prob = 1.0;		  // alpha
	double non_zero_transition = 0.5; // beta

	// ACTION_PROB EXPERIMENTS
	S = 100;
	A_num = 50;
	upper_reward = 1000.0;
	action_prob = 1.0; // alpha
	non_zero_transition = 0.5;
	string file_prefix_action_prob = "action_prob";
	// create_data_tables_action_prob(file_prefix_action_prob, S, A_num, epsilon, gamma, upper_reward, non_zero_transition);
	// thread th24(create_data_tables_action_prob,file_prefix_action_prob, S, A_num, epsilon, gamma, upper_reward, non_zero_transition);

	// NUMBER OF STATES AND ACTIONS
	time_now = time(0);
	// cout << "experiment run at: " << ctime(&time_now);
	int S_A_max = 500;
	string file_prefix_number_of_states_and_actions = "number_of_states_and_actions";
	// create_data_tables_number_of_states_and_actions(file_prefix_number_of_states_and_actions, S_A_max, epsilon, gamma, upper_reward, non_zero_transition);

	// thread th23(create_data_tables_number_of_states_and_actions, file_prefix_number_of_states_and_actions, S_A_max, epsilon, gamma, upper_reward, non_zero_transition);

	// NUMBER OF STATES
	time_now = time(0);
	// cout << "experiment run at: " << ctime(&time_now);
	int S_max = 500;
	A_num = 100;
	upper_reward = 1000.0;
	non_zero_transition = 1;
	string file_prefix_number_of_states = "number_of_states";
	// create_data_tables_number_of_states(file_prefix_number_of_states, S_max, A_num, epsilon, gamma, upper_reward, non_zero_transition);
	// thread th22(create_data_tables_number_of_states,file_prefix_number_of_states, S_max, A_num, epsilon, gamma, upper_reward, non_zero_transition);

	// S_max = 500;
	A_num = 100;
	upper_reward = 1000.0;
	non_zero_transition = 1;
	S_max = 50;
	// file_prefix_number_of_states = "number_of_states";
	// create_data_tables_number_of_statesGS(file_prefix_number_of_states, S_max, A_num, epsilon, gamma, upper_reward, non_zero_transition);
	// create_data_tables_number_of_statesGSAll(file_prefix_number_of_states, S_max, A_num, epsilon, gamma, upper_reward, non_zero_transition);

	// NUMBER OF STATES - BEST IMPLEMENTATIONS
	time_now = time(0);
	// cout << "experiment run at: " << ctime(&time_now);
	S_max = 500;
	A_num = 50;
	upper_reward = 1000.0;
	non_zero_transition = 0.5;
	string file_prefix_number_of_states_best = "number_of_states_best";
	// create_data_tables_number_of_states_best_implementations(file_prefix_number_of_states_best, S_max, A_num, epsilon, gamma, upper_reward, non_zero_transition);
	// thread th1(create_data_tables_number_of_states, file_prefix_number_of_states, S_max, A_num, epsilon, gamma, upper_reward, non_zero_transition);

	// NUMER_OF_ACTIONS EXPERIMENTS
	time_now = time(0);
	// cout << "experiment run at: " << ctime(&time_now);
	int A_max = 2000;
	S = 1000;
	upper_reward = 1000.0;
	non_zero_transition = 1;
	string file_prefix_number_of_actions = "number_of_actions";
	// create_data_tables_number_of_actions(file_prefix_number_of_actions, S, A_max, epsilon, gamma, upper_reward, non_zero_transition);
	// thread th2(create_data_tables_number_of_actions,file_prefix_number_of_actions, S, A_max, epsilon, gamma, upper_reward, non_zero_transition);

	// NUMER_OF_ACTIONS BEST EXPERIMENTS
	time_now = time(0);
	// cout << "experiment run at: " << ctime(&time_now);
	A_max = 500;
	S = 100;
	upper_reward = 1000.0;
	non_zero_transition = 0.5;
	string file_prefix_number_of_actions_best = "number_of_actions_best";
	// create_data_tables_number_of_actions_best_implementations(file_prefix_number_of_actions_best, S, A_max, epsilon, gamma, upper_reward, non_zero_transition);
	// thread th3(create_data_tables_number_of_actions, file_prefix_number_of_actions, S, A_max, epsilon, gamma, upper_reward, non_zero_transition);

	// REWARD_DIST EXPERIMENTS
	time_now = time(0);
	// cout << "experiment run at: " << ctime(&time_now);
	double reward_factor = 1000.0;
	double reward_prob = 0.5;
	action_prob = 1.0;
	string file_prefix_reward_dist = "reward_dist";
	// create_data_tables_rewards(file_prefix_reward_dist, S, A_num, epsilon, gamma, reward_factor, reward_prob, upper_reward, action_prob, non_zero_transition);

	// thread th4(create_data_tables_rewards, file_prefix_reward_dist, S, A_num, epsilon, gamma, reward_factor, reward_prob, upper_reward, action_prob, non_zero_transition);

	// NUMER_OF_TRANSITIONS_PROB EXPERIMENTS
	time_now = time(0);
	// cout << "experiment run at: " << ctime(&time_now);
	S = 250;
	A_num = 100;
	upper_reward = 1000.0;
	action_prob = 1.0; // alpha
	string file_prefix_transition_prob = "transition_prob";
	// create_data_tables_transition_prob(file_prefix_transition_prob, S, A_num, epsilon, gamma, upper_reward);

	// thread th5(create_data_tables_transition_prob, file_prefix_transition_prob, S, A_num, epsilon, gamma, upper_reward);

	// NUMBER OF STATES - ITERATIONS
	time_now = time(0);
	// cout << "experiment run at: " << ctime(&time_now);
	int S_max_iterations = 500;
	A_num = 100;
	action_prob = 1.0;
	non_zero_transition = 1.0; // beta
	upper_reward = 1000.0;
	string file_prefix_number_of_states_iterations = "number_of_states_iterations";
	// create_data_tables_number_of_states_iterations(file_prefix_number_of_states_iterations, S_max_iterations, A_num, epsilon, gamma, upper_reward, non_zero_transition);

	// thread th6(create_data_tables_number_of_states_iterations, file_prefix_number_of_states_iterations, S_max_iterations, A_num, epsilon, gamma, upper_reward, non_zero_transition);

	// NUMBER OF TRANSITIONS STATES IN EACH STATE
	time_now = time(0);
	// cout << "experiment run at: " << ctime(&time_now);
	S = 200;
	A_num = 100;
	upper_reward = 1000.0;
	action_prob = 1.0;
	int max_transitions = 200;
	string file_prefix_number_of_transitions = "number_of_transitions";
	// create_data_tables_number_of_transitions(file_prefix_number_of_transitions, S, A_num, epsilon, gamma, upper_reward, action_prob, max_transitions);

	// thread th7(create_data_tables_number_of_transitions, file_prefix_number_of_transitions, S, A_num, epsilon, gamma, upper_reward, action_prob, max_transitions);

	// NUMER_OF_ACTIONS - ITERATIONS
	time_now = time(0);
	// cout << "experiment run at: " << ctime(&time_now);
	S = 100;
	A_max = 2000;
	upper_reward = 1000.0;
	non_zero_transition = 0.5; // beta
	string file_prefix_number_of_actions_iterations = "number_of_actions_iterations";
	// create_data_tables_number_of_actions_iterations(file_prefix_number_of_actions_iterations, S, A_max, epsilon, gamma, upper_reward, non_zero_transition);

	// thread th8(create_data_tables_number_of_actions_iterations, file_prefix_number_of_actions_iterations, S, A_max, epsilon, gamma, upper_reward, non_zero_transition);

	// MAX_REWARD
	time_now = time(0);
	// cout << "experiment run at: " << ctime(&time_now);
	S = 100;
	A_num = 100;
	action_prob = 1.0;
	double max_reward_finish = 10000.0;
	string file_prefix_max_reward = "max_reward";
	// create_data_tables_max_reward(file_prefix_max_reward, S, A_num, epsilon, gamma, action_prob, max_reward_finish, non_zero_transition);

	// thread th9(create_data_tables_max_reward, file_prefix_max_reward, S, A_num, epsilon, gamma, action_prob, max_reward_finish, non_zero_transition);

	// NUMER_OF_ACTIONS CONVERGENCE ITERATION EXPERIMENTS
	time_now = time(0);
	// cout << "experiment run at: " << ctime(&time_now);
	S = 100;
	A_max = 200;
	upper_reward = 1000.0;
	non_zero_transition = S;
	string file_prefix_number_of_actions_first_convergence_iteration = "number_of_actions_first_convergence_iteration";
	// create_data_tables_number_of_actions_first_convergence_iteration(file_prefix_number_of_actions_first_convergence_iteration, S, A_max, epsilon, gamma, upper_reward, non_zero_transition);

	// thread th10(create_data_tables_number_of_actions_first_convergence_iteration, file_prefix_number_of_actions_first_convergence_iteration, S, A_max, epsilon, gamma, upper_reward, non_zero_transition);

	// WORK PER ITERATION EXPERIMENT
	time_now = time(0);
	// cout << "experiment run at: " << ctime(&time_now);
	S = 100;
	A_num = 2000;
	action_prob = 1.0;
	double mean_work = 1000.0;
	double variance_work = 10.0;
	non_zero_transition = S;
	string file_prefix_work_per_iteration = "work_per_iteration";
	// create_data_tables_work_per_iteration(file_prefix_work_per_iteration, S, A_num, epsilon, gamma, action_prob, non_zero_transition, mean_work, variance_work);

	// thread th11(create_data_tables_work_per_iteration, file_prefix_work_per_iteration, S, A_num, epsilon, gamma, action_prob, non_zero_transition, mean_work, variance_work);

	// TOP ACTION CHANGE EXPERIMENT
	time_now = time(0);
	// cout << "experiment run at: " << ctime(&time_now);
	S = 100;
	A_num = 1000;
	action_prob = 1.0;
	int number_of_transitions = 100;

	upper_reward = 1000.0;
	double mean = 1000.0;
	double variance = 10.0;
	double lambda = 0.02;

	string file_prefix_top_action_change = "top_action_change";
	// create_data_tables_top_action_change(file_prefix_top_action_change, S, A_num, epsilon, gamma, upper_reward, action_prob, number_of_transitions, mean, variance, lambda);

	// thread th12(create_data_tables_top_action_change, file_prefix_top_action_change, S, A_num, epsilon, gamma, upper_reward, action_prob, number_of_transitions, mean, variance, lambda);

	// VARYING VARIANCE IN NORMAL REWARD DISTRIBUTION
	time_now = time(0);
	// cout << "experiment run at: " << ctime(&time_now);
	S = 100;
	A_num = 100;
	action_prob = 1.0;
	number_of_transitions = 100;
	mean = 1000.0;
	double min_variance = 0.1;
	double max_variance = 10.0;
	double variance_increment = 0.5;

	// string file_prefix_normal_dist_varying_variance = "normal_dist_varying_variance";
	// create_data_tables_normal_dist_varying_variance(file_prefix_normal_dist_varying_variance, S, A_num, epsilon, gamma, action_prob, number_of_transitions, mean, min_variance, max_variance, variance_increment);

	// thread th13(create_data_tables_normal_dist_varying_variance, file_prefix_normal_dist_varying_variance, S, A_num, epsilon, gamma, action_prob, number_of_transitions, mean, min_variance, max_variance, variance_increment);

	// VARYING VARIANCE IN NORMAL REWARD DISTRIBUTION - VERY SMALL VARIANCE
	time_now = time(0);
	// cout << "experiment run at: " << ctime(&time_now);
	S = 100;
	A_num = 100;
	action_prob = 1.0;
	number_of_transitions = 50;
	mean = 100.0;
	min_variance = 0.01;
	max_variance = 2.0;
	variance_increment = 0.05;

	string file_prefix_normal_dist_varying_variance_very_small = "normal_dist_varying_variance_very_small";
	// create_data_tables_normal_dist_varying_variance(file_prefix_normal_dist_varying_variance_very_small, S, A_num, epsilon, gamma, action_prob, number_of_transitions, mean, min_variance, max_variance, variance_increment);

	// thread th14(create_data_tables_normal_dist_varying_variance, file_prefix_normal_dist_varying_variance_very_small, S, A_num, epsilon, gamma, action_prob, number_of_transitions, mean, min_variance, max_variance, variance_increment);

	// VIH DIFFERENT REWARD DISTRIBUTIONS
	time_now = time(0);
	// cout << "experiment run at: " << ctime(&time_now);
	S = 200;
	A_num = 1000;
	action_prob = 1.0;
	number_of_transitions = 50;

	double mean_1 = 50.0;
	double mean_2 = 50.0;
	double mean_3 = 50.0;

	double variance_1 = 0.01;
	double variance_2 = 5.0;
	double variance_3 = 25.0;

	double lambda_1 = 0.02;
	double lambda_2 = 2.0;
	double lambda_3 = 20.0;

	double upper_reward_1 = 10.0;
	double upper_reward_2 = 1000.0;
	double upper_reward_3 = 100000.0;

	string file_prefix_VIH_distributions_iterations_work = "VIH_distributions_iterations";
	// create_data_tables_VIH_distributions_iterations_work(file_prefix_VIH_distributions_iterations_work, S, A_num, epsilon, gamma, action_prob, number_of_transitions, mean_1, mean_2, mean_3, variance_1, variance_2, variance_3, lambda_1, lambda_2, lambda_3, upper_reward_1, upper_reward_2, upper_reward_3);

	// thread th15(create_data_tables_VIH_distributions_iterations_work, file_prefix_VIH_distributions_iterations_work, S, A_num, epsilon, gamma, action_prob, number_of_transitions, mean_1, mean_2, mean_3, variance_1, variance_2, variance_3, lambda_1, lambda_2, lambda_3, upper_reward_1, upper_reward_2, upper_reward_3);

	// VARYING LAMBDA EXPERIMENT - EXPONENTIAL REWARD DISTRIBUTION
	time_now = time(0);
	// cout << "experiment run at: " << ctime(&time_now);
	S = 100;
	A_num = 100;
	action_prob = 1.0;
	int transition_list_size = 50;
	double min_lambda = 0.01;
	double max_lambda = 1.0;
	double lambda_increment = 0.01;

	string file_prefix_exponential_dist_varying_lambda = "exponential_dist_varying_lambda";
	// create_data_tables_exponential_dist_varying_lambda(file_prefix_exponential_dist_varying_lambda, S, A_num, epsilon, gamma, action_prob, transition_list_size, min_lambda, max_lambda, lambda_increment);

	// thread th16(create_data_tables_exponential_dist_varying_lambda, file_prefix_exponential_dist_varying_lambda, S, A_num, epsilon, gamma, action_prob, transition_list_size, min_lambda, max_lambda, lambda_increment);

	// WORK PER ITERATION EXPERIMENT OF VIAEH IMPLEMENTATIONS
	time_now = time(0);
	// cout << "experiment run at: " << ctime(&time_now);
	S = 100;
	A_num = 2000;
	action_prob = 1.0;
	number_of_transitions = 50;

	mean = 1000.0;
	variance = 10.0;

	string file_prefix_work_per_iteration_VIAEH_implementations = "work_per_iteration_VIAEH_implementations";
	// create_data_tables_work_per_iteration_VIAEH_implementations(file_prefix_work_per_iteration_VIAEH_implementations, S, A_num, epsilon, gamma, action_prob, number_of_transitions, mean, variance);

	// thread th17(create_data_tables_work_per_iteration_VIAEH_implementations, file_prefix_work_per_iteration_VIAEH_implementations, S, A_num, epsilon, gamma, action_prob, number_of_transitions, mean, variance);

	// BOUNDS EXPERIMENT
	S = 100;
	A_num = 1000;
	action_prob = 1.0;
	number_of_transitions = 50;

	mean = 100.0;
	variance = 10.0;

	string file_prefix_bounds_comparison = "bounds_comparison";
	// create_data_tables_bounds_comparisons(file_prefix_bounds_comparison, S, A_num, epsilon, gamma, action_prob, number_of_transitions, mean, variance);

	// thread th18(create_data_tables_bounds_comparisons,file_prefix_bounds_comparison, S, A_num, epsilon, gamma, action_prob, number_of_transitions, mean, variance);

	// ACTIONS TOUCHED AND IMPLICIT/EXPLICIT IMPLEMENTATION
	S = 100;
	A_num = 100;
	action_prob = 1.0;
	number_of_transitions = 50;

	mean = 100.0;
	variance = 10.0;

	string file_prefix_actions_touched = "actions_touched";
	// create_data_tables_actions_touched(file_prefix_actions_touched, S, A_num, epsilon, gamma, action_prob, number_of_transitions, mean, variance);

	// thread th19(create_data_tables_actions_touched,file_prefix_actions_touched, S, A_num, epsilon, gamma, action_prob, number_of_transitions, mean, variance);

	// WORK PER ITERATION EXPERIMENT OF BEST IMPLEMENTATIONS
	S = 100;
	A_num = 1000;
	action_prob = 1.0;
	number_of_transitions = 50;

	mean = 100.0;
	variance = 10.0;

	string file_prefix_work_per_iteration_BEST_implementations = "work_per_iteration_BEST_implementations";
	// create_data_tables_work_per_iteration_BEST_implementations(file_prefix_work_per_iteration_BEST_implementations, S, A_num, epsilon, gamma, action_prob, number_of_transitions, mean, variance);
	// thread th20(create_data_tables_work_per_iteration_BEST_implementations,file_prefix_work_per_iteration_BEST_implementations, S, A_num, epsilon, gamma, action_prob, number_of_transitions, mean, variance);

	// NUMER_OF_ACTIONS BEST EXPERIMENTS
	time_now = time(0);
	// cout << "experiment run at: " << ctime(&time_now);
	A_max = 10000;
	S = 100;
	upper_reward = 1000.0;
	non_zero_transition = 0.5;
	string file_prefix_number_of_actions_VIH_impl = "number_of_actions_VIH_impl";
	// create_data_tables_number_of_actions_VIH_implementations(file_prefix_number_of_actions_VIH_impl, S, A_max, epsilon, gamma, upper_reward, non_zero_transition);
	// thread th21(create_data_tablkes_number_of_actions_VIH_implementations,file_prefix_number_of_actions_VIH_impl, S, A_max, epsilon, gamma, upper_reward, non_zero_transition);
	epsilon = 0.05;
	gamma = 0.99;
	// string file_prefix_numberVMS = "numberVMS";
	string file_prefix_numberVMA = "numberVMA";
	string file_prefix_numberVMS = "numberVMS";
	// create_data_tables_VMS(file_prefix_numberVMS, epsilon, gamma);

	// create_data_tables_VMS(file_prefix_numberVMA, epsilon, gamma);
	// string file_prefix_numberVMSA = "numberVMA";
	// create_data_tables_VMA(file_prefix_numberVMSA, epsilon, gamma);
	// WAIT FOR ALL THREAD BEFORE EXITING - VERY IMPORTANT!!!!!
	int StartP = 10;
	int EndP = 15;
	int IncP = 5;
	int expnum = 2;
	int NOFexp = 6;
	int States = 100;
	int Actions = 50;
	int SS = 50;
	std::size_t pos;
	cout << "argc " << argc;
	if (argc > 1)
		NOFexp = std::stoi(argv[1], &pos);
	if (argc > 2)
	{
		expnum = std::stoi(argv[2], &pos);
		States = std::stoi(argv[3], &pos);
		Actions = std::stoi(argv[4], &pos);
		SS = std::stoi(argv[5], &pos);
		StartP = std::stoi(argv[6], &pos);
		EndP = std::stoi(argv[7], &pos);
		IncP = std::stoi(argv[8], &pos);
	}
	cout << "NOFexp " << NOFexp;
	if (NOFexp == 1)
	{
		create_data_tables_VMS(file_prefix_numberVMS, epsilon, gamma);
		cout << "end1" << endl;
	}
	else if (NOFexp == 2)
	{
		create_data_tables_VMS(file_prefix_numberVMA, epsilon, gamma);
		cout << "end2" << endl;
	}
	else if (NOFexp == 3)
	{
		// expnum=0 States, 1 Actions, 2 Supported States
		create_data_tables_number_GS(file_prefix_number_of_states, expnum, States, Actions, SS, StartP, EndP, IncP, epsilon, gamma, upper_reward, non_zero_transition);
		cout << "end3" << endl;
	}
	// create_data_tables_number_of_statesGS(file_prefix_number_of_states, S_max, A_num, epsilon, gamma, upper_reward, non_zero_transition);
	else if (NOFexp == 4)
	{
		create_data_tables_number_GSBO(file_prefix_number_of_states_best, expnum, States, Actions, SS, StartP, EndP, IncP, epsilon, gamma, upper_reward, non_zero_transition);
		cout << "end4" << endl;
	}
	else if (NOFexp == 5)
		create_data_tables_actions_touched(file_prefix_actions_touched, S, A_num, epsilon, gamma, action_prob, number_of_transitions, mean, variance);
	else if (NOFexp == 6)
	{
		create_data_tables_number_GSTM(file_prefix_number_of_states_best, expnum, States, Actions, SS, StartP, EndP, IncP, epsilon, gamma, upper_reward, non_zero_transition);
		cout << "end4" << endl;
	}
	cout << "NOFexp " << NOFexp;

	// th1.join();
	// th2.join();
	// th3.join();
	// th4.join();
	// th5.join();
	// th6.join();
	// th7.join();
	// th8.join();
	// th9.join();
	// th10.join();
	// th11.join();
	// th12.join();
	// th13.join();
	// th14.join();
	// th15.join();
	// th16.join();
	// th17.join();
	// th18.join();
	// th19.join();
	// th20.join();
	// th21.join();
	// th22.join();
	// th23.join();
	// th24.join();

	return 0;
}
