from Algorithms.value_iteration import (grid_problem_value_iteration,
                            cookie_problem_value_iteration, 
                            gambler_problem_value_iteration)
from Problems.GridProblem import GridProblem
from Problems.CookieProblem import CookieProblem
from Problems.GamblerProblem import GamblerProblem
from Algorithms.greedy_policy import choose_gready_action, sample_transition

def play_value_iteration_greed(tamano):
    problem = GridProblem(tamano)
    policy = grid_problem_value_iteration(tamano)
    state = problem.get_initial_state()
    done = False
    total_reward = 0.0
    while not done:
        problem.show(state)
        action = policy.get_action(state)
        print(action)
        transitions = problem.get_transitions(state, action)
        s_next, reward = sample_transition(transitions)
        done = problem.is_terminal(s_next)
        state = s_next
        total_reward += reward
    print("Done.")
    print(f"Total reward: {total_reward}")

def play_value_iteration_cookie(tamano):
    problem = CookieProblem(tamano)
    policy = cookie_problem_value_iteration(tamano)
    state = problem.get_initial_state()
    done = False
    total_reward = 0.0
    while not done:
        problem.show(state)
        action = policy.get_action(state)
        print(action)
        transitions = problem.get_transitions(state, action)
        s_next, reward = sample_transition(transitions)
        done = problem.is_terminal(s_next)
        state = s_next
        total_reward += reward
    print("Done.")
    print(f"Total reward: {total_reward}")

def play_value_iteration_gambler(prob):
    problem = GamblerProblem(prob)
    policy = gambler_problem_value_iteration(prob)
    state = problem.get_initial_state()
    done = False
    total_reward = 0.0
    while not done:
        problem.show(state)
        action = policy.get_action(state)
        print(action)
        transitions = problem.get_transitions(state, action)
        s_next, reward = sample_transition(transitions)
        done = problem.is_terminal(s_next)
        state = s_next
        total_reward += reward
    print("Done.")
    print(f"Total reward: {total_reward}")

