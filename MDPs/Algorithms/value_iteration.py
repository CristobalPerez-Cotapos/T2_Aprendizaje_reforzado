from datetime import datetime
from Problems.GridProblem import GridProblem
from Problems.CookieProblem import CookieProblem
from Problems.GamblerProblem import GamblerProblem

class Policy:

    def __init__(self, V, problem):
        self.problem = problem
        self.V = V

    def get_action(self, state):
        actions = self.problem.get_available_actions(state)
        best_action = None
        best_value = float('-inf')
        for action in actions:
            transitions = self.problem.get_transitions(state, action)
            value = sum(prob * (reward + self.V[s_next]) for prob, s_next, reward in transitions)
            if value > best_value:
                best_value = value
                best_action = action
        return best_action

def value_iteration(problem, gamma=0.9, tol=1e-3):
    initial_time = datetime.now()
    states = problem.states
    V = {s: 0 for s in states}
    while True:
        delta = 0
        for s in states:
            if problem.is_terminal(s):
                continue
            v = V[s]
            V[s] = max(sum(prob * (reward + gamma * V[s_next]) for prob, s_next, reward in problem.get_transitions(s, a)) for a in problem.get_available_actions(s))
            delta = max(delta, abs(v - V[s]))
        if delta < tol:
            break
    print(f"Time: {datetime.now() - initial_time}")
    return V, Policy(V, problem)

def grid_problem_value_iteration(i):
    print(f"Tamaño {i}")
    problem = GridProblem(i)
    V, policy = value_iteration(problem, gamma=1, tol=0.0000000001)
    print(problem.get_initial_state())
    print(V[problem.get_initial_state()])
    return policy

def cookie_problem_value_iteration(i):
    problem = CookieProblem(i)
    V, policy = value_iteration(problem, gamma=0.99, tol=0.0000000001)
    print(problem.get_initial_state())
    print(V[problem.get_initial_state()])
    return policy

def gambler_problem_value_iteration(prob):
    print(f"Probabilidad {prob}")
    problem = GamblerProblem(prob)
    V, policy = value_iteration(problem, gamma=1, tol=0.0000000001)
    print(problem.get_initial_state())
    print(V[problem.get_initial_state()])
    return policy
