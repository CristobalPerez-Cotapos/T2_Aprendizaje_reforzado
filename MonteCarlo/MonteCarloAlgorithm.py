import random

def save_policy(problem, policy, filename):
    with open(filename, 'w') as f:
        for s in problem.states:
            f.write(f"{s}: {policy[s]}\n")

def on_policy_every_visit_mc_control(problem, gamma=1, epsilon=0.01, episodes=10000000, evaluation_episodes=500000):
    Q = {(s, a): 0 for s in problem.states for a in problem.action_space}
    returns = {(s, a): {"N": 0} for s in problem.states for a in problem.action_space}
    policy = {s: {a: 1 / len(problem.action_space) for a in problem.action_space} for s in problem.states}
    
    for episodio in range(episodes):
        episode = []
        state = problem.reset()
        state_2 = state
        done = False
        while not done:
            action = random.choices(problem.action_space, weights=[policy[state][a] for a in problem.action_space])[0]
            s_next, reward, done = problem.step(action)
            episode.append((state, action, reward))
            state = s_next
        G = 0
        for t in range(len(episode) - 1, -1, -1):
            state, action, reward = episode[t]
            G = gamma * G + reward
            if not (state, action) in [(s, a) for s, a, _ in episode[:t]]:
                returns[(state, action)]["N"] += 1
                Q[(state, action)] = Q[(state, action)] + (G - Q[(state, action)]) / returns[(state, action)]["N"]
                best_action = max(problem.action_space, key=lambda a: Q[(state, a)])
                for a in problem.action_space:
                    policy[state][a] = 1 - epsilon + epsilon / len(problem.action_space) if a == best_action else epsilon / len(problem.action_space)

        if episodio % evaluation_episodes == 0 and episodio > 0:
            evaluation = evaluate_policy_greedy(problem, policy, episodes=1)
            print(f"Episodio: {episodio}")
            print(f"Evaluación: {evaluation}")
            pass

    save_policy(problem, policy, 'policy.txt')
    return policy

def evaluate_policy_greedy(problem, policy, episodes=100000, show=False):
    rewards_n = 0
    rewards_avg = 0
    for _ in range(episodes):
        state = problem.reset()
        done = False
        total_reward = 0
        while not done:
            if show:
                problem.show()
            action = max(problem.action_space, key=lambda a: policy[state][a])
            state, reward, done = problem.step(action)
            total_reward += reward
        rewards_n += 1
        rewards_avg += (total_reward - rewards_avg) / rewards_n
    return rewards_avg
