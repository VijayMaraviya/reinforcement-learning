import gym
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
import collections

# generate a episode using a given policy
def generate_episode(env, policy):
    episode = []
    obs = env.reset()
    done = False
    while not done:
        # choose action
        action, act_prob = policy(obs)
        # take action
        next_obs, reward, done, info = env.step(action)
        # append (S_t-1, A_t-1, R_t)
        episode.append((obs, action, reward, act_prob))
        obs = next_obs
    return episode

# create behaviour policy (e-greedy w.r.t target policy)
def behaviour_policy(pi, EPS, act_space, discretizer):
    # return action a and probability b(A|s)
    def e_greedy_policy(s):
        s = discretizer(*s)
        if np.random.random() < 1 - EPS:
            act = pi[s]
            prob = 1 - EPS
        else:
            act = np.random.choice(act_space)
            prob = EPS
        return act, prob
    return e_greedy_policy

# off-policy Monte-Carlo algorithm to learn target policy pi
def train(env, state_space, action_space, descritizer, max_episodes=100000, GAMMA=1.0, EPS=0.1):

    # intialize Q, C, pi
    Q = {}
    pi = {}
    C = {}
    for s in state_space:
        pi[s] = np.random.choice(action_space)
        for a in action_space:
            Q[(s, a)] = 0
            C[(s, a)] = 0

    # loop for max_episodes
    for n_eps in range(max_episodes):
        # behaviour policy
        b = behaviour_policy(pi, EPS, action_space, descritizer)
        # generate a episode using behaviour policy
        episode = generate_episode(env, b)
        # set return of terminal state (by defination)
        G = 0
        W = 1
        for S, A, R, act_prob in reversed(episode):
            S = descritizer(*S)
            # calculate return
            G = GAMMA*G + R
            # updtae C
            C[(S, A)] = C[(S, A)] + W
            # updtae Q
            Q[(S, A)] = Q[(S, A)] + (W/C[(S, A)])*(G - Q[(S, A)])

            # improve target policy
            act_vals = np.array([Q[(S, a)] for a in action_space])
            pi[S] = action_space[np.random.choice(
                np.where(act_vals == act_vals.max())[0])]

            if A != pi[S]:
                break

            # update W
            W = W*(1/act_prob)

        if n_eps % 10000 == 0:
            print("episode:", n_eps)

    return pi

# test a policy
def test(policy, num_episodes=1000):
    rewards = np.zeros(num_episodes)
    totalReward = 0
    wins = 0
    losses = 0
    draws = 0

    for i in range(num_episodes):
        observation = bj_env.reset()
        done = False
        while not done:
            if isinstance(policy, collections.Callable):
                action = policy(observation)
            else:
                action = policy[observation]
            observation_, reward, done, info = bj_env.step(action)
            observation = observation_
        totalReward += reward
        rewards[i] = totalReward

        if reward >= 1:
            wins += 1
        elif reward == 0:
            draws += 1
        elif reward == -1:
            losses += 1

    win_rate = wins/num_episodes
    loss_rate = losses/num_episodes
    draw_rate = draws/num_episodes

    return rewards, win_rate, loss_rate, draw_rate


# blackjack env
bj_env = gym.make('Blackjack-v0')

# discretize the state
def bj_discretizer(agent_sum, dealer_show_card, agent_ace):
    return (agent_sum, dealer_show_card, agent_ace)


agent_sum_space = [i for i in range(4, 22)]
dealer_show_card_space = [i+1 for i in range(10)]
agent_ace_space = [False, True]

# action space
bj_action_space = [0, 1]  # stick or hit

# state space
bj_state_space = []
for s in product(agent_sum_space, dealer_show_card_space, agent_ace_space):
    bj_state_space.append(s)


# learn policy
pi_blackjack = train(env=bj_env,
                     state_space=bj_state_space,
                     action_space=bj_action_space,
                     descritizer=bj_discretizer)

# random policy
def bj_random_policy(*args):
    return np.random.choice(bj_action_space)


# outcome random policy
rewards, win_rate, loss_rate, draw_rate = test(policy=bj_random_policy)
print(f"win percentage: {win_rate*100:.2f}%")
print(f"loss percentage: {loss_rate*100:.2f}%")
print(f"draw percentage: {draw_rate*100:.2f}%")

plt.plot(rewards)
plt.xlabel('episode number')
plt.ylabel('cumulative reward')
plt.show()

# outcome with optimal policy learned with Monte Carlo off-policy method
rewards, win_rate, loss_rate, draw_rate = test(policy=pi_blackjack)
print(f"win percentage: {win_rate*100:.2f}%")
print(f"loss percentage: {loss_rate*100:.2f}%")
print(f"draw percentage: {draw_rate*100:.2f}%")

plt.plot(rewards)
plt.xlabel('episode number')
plt.ylabel('cumulative reward')
plt.show()