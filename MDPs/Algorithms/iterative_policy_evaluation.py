from datetime import datetime
from Problems.CookieProblem import CookieProblem
from Problems.GridProblem import GridProblem
from Problems.GamblerProblem import GamblerProblem

def iterative_policy_evaluation(problem, policy, gamma=0.9, tol=1e-3):
    initial_time = datetime.now()
    states = problem.states
    V = {s: 0 for s in states}
    delta = 0.0
    while True:
        delta = 0
        for s in states:
            if problem.is_terminal(s):
                #print(s, V[s])
                continue
            v = V[s]
            V[s] = sum(policy[s][a] * sum(prob * (reward + gamma * V[s_next]) for prob, s_next, reward in problem.get_transitions(s, a)) for a in problem.get_available_actions(s))
            delta = max(delta, abs(v - V[s]))
            #print(s, V[s])
        if delta < tol:
            break
    print(f"Time: {datetime.now() - initial_time}")
    return V

def grid_problem_values(i):
    print(f"Tamaño {i}")
    
    problem = GridProblem(i)
    policy = {s: {a: 1 / len(problem.get_available_actions(s)) for a in problem.get_available_actions(s)} for s in problem.states}
    V = iterative_policy_evaluation(problem, policy, gamma=1, tol=0.0000000001)
    print(problem.get_initial_state())
    print(V[problem.get_initial_state()])
    return V

def cookie_problem_values(i):
    problem = CookieProblem(i)
    policy = {s: {a: 1 / len(problem.get_available_actions(s)) for a in problem.get_available_actions(s)} for s in problem.states}
    V = iterative_policy_evaluation(problem, policy, gamma=0.99, tol=0.0000000001)
    print(problem.get_initial_state())
    print(V[problem.get_initial_state()])
    return V

def gambler_problem_values(prob):
    print(f"Probabilidad {prob}")
    problem = GamblerProblem(prob)
    policy = {s: {a: 1 / len(problem.get_available_actions(s)) for a in problem.get_available_actions(s)} for s in problem.states}
    V = iterative_policy_evaluation(problem, policy, gamma=1, tol=0.0000000001)
    print(problem.get_initial_state())
    print(V[problem.get_initial_state()])
    return V


if __name__ == '__main__':
    # grid_problem_values()
    # cookie_problem_values()
    gambler_problem_values()


