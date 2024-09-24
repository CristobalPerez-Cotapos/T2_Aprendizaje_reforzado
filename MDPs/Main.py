import random
from datetime import datetime

from Problems.CookieProblem import CookieProblem
from Problems.GridProblem import GridProblem
from Problems.GamblerProblem import GamblerProblem
from Algorithms.iterative_policy_evaluation import grid_problem_values, cookie_problem_values, gambler_problem_values
from Algorithms.greedy_policy import play_greedy_greed, play_greedy_cookie, play_greedy_gambler
from Algorithms.value_iteration import grid_problem_value_iteration, cookie_problem_value_iteration, gambler_problem_value_iteration
from Algorithms.value_iteration_player import play_value_iteration_greed, play_value_iteration_cookie, play_value_iteration_gambler


def get_action_from_user(actions):
    print("Valid actions:")
    for i in range(len(actions)):
        print(f"{i}. {actions[i]}")
    print("Please select an action:")
    selected_id = -1
    while not (0 <= selected_id < len(actions)):
        selected_id = int(input())
    return actions[selected_id]


def sample_transition(transitions):
    probs = [prob for prob, _, _ in transitions]
    transition = random.choices(population=transitions, weights=probs)[0]
    prob, s_next, reward = transition
    return s_next, reward


def play(problem):
    state = problem.get_initial_state()
    done = False
    total_reward = 0.0
    while not done:
        problem.show(state)
        actions = problem.get_available_actions(state)
        # action = get_action_from_user(actions)
        action = random.choice(actions)
        transitions = problem.get_transitions(state, action)
        s_next, reward = sample_transition(transitions)
        done = problem.is_terminal(s_next)
        state = s_next
        total_reward += reward
    print("Done.")
    print(f"Total reward: {total_reward}")


def play_gambler_problem():
    p = 0.4
    problem = GamblerProblem(p)
    play(problem)


def play_grid_problem():
    size = 4
    problem = GridProblem(size)
    play(problem)


def play_cookie_problem():
    size = 3
    problem = CookieProblem(size)
    play(problem)



if __name__ == '__main__':
    # play_grid_problem()
    # play_cookie_problem()
    # play_gambler_problem()


    #print("Iterative Policy Evaluation")
    # grid_problem_values(10)
    # cookie_problem_values()
    # gambler_problem_values(0.51)

    # play_greedy_greed(10)
    # play_greedy_cookie(10)
    # play_greedy_gambler(0.4)

    #print("Value Iteration")
    # grid_problem_value_iteration(10)
    # cookie_problem_value_iteration(3)
    # gambler_problem_value_iteration(0.51)

    # play_value_iteration_greed(10)
    # play_value_iteration_cookie(3)
    play_value_iteration_gambler(0.51)




