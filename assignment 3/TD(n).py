import math
import numpy as np
import gym
import matplotlib.pyplot as plt
from math import inf
from itertools import product
from sklearn.preprocessing import KBinsDiscretizer


def e_greedy(Q, eps, S):
    # random action with probability eps
    if np.random.random() < eps:
        return np.random.choice([0, 1])

    # greedy action otherwise
    act_vals = np.array([Q[(S, a)] for a in [0, 1]])
    return np.random.choice(np.where(act_vals == act_vals.max())[0])


def decay_value(init_val, min_val, global_step, decay_step):
    new_val = init_val * np.power(0.9, (global_step/decay_step))
    if new_val < min_val:
        return min_val
    else:
        return new_val


def n_SARSA(env, state_space, action_space, descritizer, Q,
            max_episodes=10000, GAMMA=1.0,
            EPS=0.1, ALPHA=0.1, n=1):

    eps = EPS
    alpha = ALPHA

    # intialize Q
    # Q = {}
    # for s in state_space:
    #     for a in action_space:
    #         Q[(s,a)] = 0

    scores = []

    for episode in range(max_episodes):
        T = inf
        t = 0

        # storage
        states = ['s']*(n+1)
        actions = ['a']*(n+1)
        rewards = ['r']*(n+1)

        # initialize S and store
        obs = env.reset()
        S = descritizer(*obs)
        states[t % (n+1)] = S

        # choose A and store
        A = e_greedy(Q, eps, S)
        actions[t % (n+1)] = A

        score = 0
        while True:
            if t < T:
                # take action A, observe R and S_next
                obs, R, done, _ = env.step(A)
                S = descritizer(*obs)

                score += R

                # store R and S_next
                rewards[(t+1) % (n+1)] = R
                states[(t+1) % (n+1)] = S

                if done:
                    T = t + 1
                    # print('episode ends at step', t)
                else:
                    # choose and store A_next
                    A = e_greedy(Q, eps, S)
                    actions[(t+1) % (n+1)] = A

            tau = t - n + 1
            if tau >= 0:
                G = [GAMMA**(i-tau-1)*rewards[i % (n+1)]
                     for i in range(tau+1, min(tau+n, T) + 1)]
                G = np.sum(G)

                if tau + n < T:
                    s = states[(tau+n) % (n+1)]
                    a = actions[(tau+n) % (n+1)]
                    G += (GAMMA**n) * Q[(s, a)]

                s = states[tau % (n+1)]
                a = actions[tau % (n+1)]
                Q[(s, a)] += alpha*(G-Q[(s, a)])
                # print('tau ', tau, '| Q %.2f' % Q[states[tau%n], actions[tau%n]],
                #  'state: s', states[tau%n], 'action: a', actions[tau%n])

            t += 1
            if tau == T - 1:
                break

        # eps = decay_value(EPS, 0.1, episode, 5000)

        scores.append(score)
        avg_score = np.mean(scores[-1000:])

        if episode % 100 == 0:
            print('episode ', episode, 'avg_score %.1f' % avg_score)
            # print('Q:', Q)

    return Q, scores


cp_env = gym.make("CartPole-v0")
cp_env.reset()

# lower bounds of state space
lower_bounds = cp_env.observation_space.low
lower_bounds[1] = -0.5
lower_bounds[3] = -math.radians(50)

# upper bounds of state space
upper_bounds = cp_env.observation_space.high
upper_bounds[1] = 0.5
upper_bounds[3] = math.radians(50)

n_bins = (12, 12, 12, 12)

# discretize the state


def cp_discretizer(cart_position, cart_velocity, pole_angle, pole_velocity):
    est = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform')
    est.fit([lower_bounds, upper_bounds])
    return tuple(map(int, est.transform([[cart_position, cart_velocity, pole_angle, pole_velocity]])[0]))


# action_space
cp_action_space = [0, 1]

# discretized state_space
cp_state_space = []
for s in product(range(12), range(12), range(12), range(12)):
    cp_state_space.append(s)

# intialize Q
Q = {}
for s in cp_state_space:
    for a in cp_action_space:
        Q[(s, a)] = 0

Q_n_sarsa, rewards = n_SARSA(cp_env,
                             cp_state_space,
                             cp_action_space,
                             cp_discretizer,
                             max_episodes=10000, GAMMA=1.0, EPS=0.1, ALPHA=0.1, n=4,
                             Q=Q)
