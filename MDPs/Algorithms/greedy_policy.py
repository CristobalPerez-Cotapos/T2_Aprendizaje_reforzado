from Algorithms.iterative_policy_evaluation import (iterative_policy_evaluation, 
                                                    grid_problem_values, cookie_problem_values, 
                                                    gambler_problem_values)
from Problems.CookieProblem import CookieProblem
from Problems.GridProblem import GridProblem
from Problems.GamblerProblem import GamblerProblem
import random

def choose_gready_action(problem, V, state):
    actions = problem.get_available_actions(state)
    best_action = None
    best_value = float('-inf')
    for action in actions:
        transitions = problem.get_transitions(state, action)
        value = sum(prob * (reward + V[s_next]) for prob, s_next, reward in transitions)
        if value > best_value:
            best_value = value
            best_action = action
    print(f"Best action: {best_action}")
    return best_action

def sample_transition(transitions):
    probs = [prob for prob, _, _ in transitions]
    transition = random.choices(population=transitions, weights=probs)[0]
    prob, s_next, reward = transition
    return s_next, reward

def play_greedy_greed(tamano):
    problem = GridProblem(tamano)
    V = grid_problem_values(tamano)
    state = problem.get_initial_state()
    done = False
    total_reward = 0.0
    while not done:
        problem.show(state)
        action = choose_gready_action(problem, V, state)
        transitions = problem.get_transitions(state, action)
        s_next, reward = sample_transition(transitions)
        done = problem.is_terminal(s_next)
        state = s_next
        total_reward += reward
    print("Done.")
    print(f"Total reward: {total_reward}")

def play_greedy_cookie(tamano):
    problem = CookieProblem(tamano)
    V = cookie_problem_values(tamano)
    state = problem.get_initial_state()
    done = False
    total_reward = 0.0
    while not done:
        problem.show(state)
        action = choose_gready_action(problem, V, state)
        transitions = problem.get_transitions(state, action)
        s_next, reward = sample_transition(transitions)
        done = problem.is_terminal(s_next)
        state = s_next
        total_reward += reward
    print("Done.")
    print(f"Total reward: {total_reward}")
    
def play_greedy_gambler(prob):
    problem = GamblerProblem(prob)
    V = gambler_problem_values(prob)
    state = problem.get_initial_state()
    done = False
    total_reward = 0.0
    while not done:
        problem.show(state)
        action = choose_gready_action(problem, V, state)
        transitions = problem.get_transitions(state, action)
        s_next, reward = sample_transition(transitions)
        done = problem.is_terminal(s_next)
        state = s_next
        total_reward += reward
    print("Done.")
    print(f"Total reward: {total_reward}")

