import gym
import numpy as np
import math
import matplotlib.pyplot as plt
from IPython import display as ipythondisplay
from sklearn.preprocessing import KBinsDiscretizer
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

# cart-pole environment
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

n_bins = ( 12, 12, 12 , 12 )

# discretize the state
def cp_discretizer( cart_position, cart_velocity, pole_angle, pole_velocity):
    est = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform')
    est.fit([lower_bounds, upper_bounds ])
    return tuple(map(int, est.transform([[cart_position, cart_velocity, pole_angle, pole_velocity]])[0]))

# action_space
cp_action_space = [0,1]

# discretized state_space
cp_state_space = []
for s in product(range(12), range(12), range(12), range(12)):
    cp_state_space.append(s)

# learned policy
pi_cartpole = train(env = cp_env, 
                    state_space = cp_state_space,
                    action_space = cp_action_space,
                    descritizer = cp_discretizer, 
                    max_episodes = 100000, 
                    GAMMA = 1.0, 
                    EPS = 0.1)

# random policy
def cp_random_policy(*args):
    return np.random.choice(cp_action_space)

# function to test policy 
def test(policy, num_episodes = 1000):
    rewards = np.zeros(num_episodes)

    for i in range(num_episodes):
        totalReward = 0
        observation = cp_discretizer(*cp_env.reset())
        done = False
        while not done:
            if isinstance(policy, collections.Callable):
                action = policy(observation)
            else:
                action = policy[observation]
            observation_, reward, done, info = cp_env.step(action)            
            observation = cp_discretizer(*observation_)
            totalReward += reward
        rewards[i] = totalReward
    
    return rewards

# outcome with random policy
rewards = test(policy = cp_random_policy)
print(f"Average reward over 1000 episodes: {np.average(rewards):.2f}")
print(f"number of successes (reward >=200) in 1000 episodes: {np.sum(np.where(rewards >= 200, 1, 0))}")
                                               
plt.plot(rewards)
plt.xlabel('episode number')
plt.ylabel('reward')
plt.show()

# outcome with optimal policy learned with Monte Carlo off-policy method
rewards = test(policy = pi_cartpole)
print(f"Average reward over 1000 episodes: {np.average(rewards):.2f}")
print(f"number of successes (reward >=200) in 1000 episodes: {np.sum(np.where(rewards >= 200, 1, 0))}")
plt.plot(rewards)
plt.xlabel('episode number')
plt.ylabel('reward')
plt.show()