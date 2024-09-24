import matplotlib.pyplot as plt


def plot_optimal_values_graph(policy):
    action_values = []
    state_values = []

    for i in range(100):
        action = policy.get_action(i)
        action_values.append(action)
        state_values.append(i)
    
    plt.plot(state_values, action_values)
    plt.xlabel("Capital")
    plt.ylabel("Final policy (stake)")
    plt.show()






