from Environments.BlackjackEnv import BlackjackEnv
from Environments.CliffEnv import CliffEnv
from MonteCarloAlgorithm import on_policy_every_visit_mc_control


def get_action_from_user(actions):
    print("Valid actions:")
    for i in range(len(actions)):
        print(f"{i}. {actions[i]}")
    print("Please select an action:")
    selected_id = -1
    while not (0 <= selected_id < len(actions)):
        selected_id = int(input())
    return actions[selected_id]


def play(env):
    actions = env.action_space
    state = env.reset()
    total_reward = 0.0
    done = False
    while not done:
        env.show()
        action = get_action_from_user(actions)
        state, reward, done = env.step(action)
        total_reward += reward
    env.show()
    print("Done.")
    print(f"Total reward: {total_reward}")


def play_blackjack():
    env = BlackjackEnv()
    play(env)


def play_cliff():
    cliff_width = 6
    env = CliffEnv(cliff_width)
    play(env)


if __name__ == '__main__':
    # play_blackjack()
    # play_cliff()

    #policy = on_policy_every_visit_mc_control(BlackjackEnv())
    policy = on_policy_every_visit_mc_control(CliffEnv(), episodes=200000, evaluation_episodes=1000, epsilon=0.1)

