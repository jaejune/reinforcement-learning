import gym
from itsdangerous import exc
import numpy as np
import matplotlib.pyplot as plt
import time

class PolicyIteration:
    def __init__(self, env, state_size, action_size, episodes, envaluations, render):
        self.env = env
        self.state_size = state_size
        self.action_size = action_size
        self.episodes = episodes
        self.render = render
        self.envaluations = envaluations

        # Hyperparameters
        self.gamma = 0.99
        self.trial = 500
        self.max_itr = 30
        self.sample = 1e6

    def policy_evaluation(self, policy, value, trans_prob, reward):
        counter = 0

        while counter < self.max_itr:
            counter += 1
            for s in range(self.state_size):
                score = 0
                for s_new in range(self.state_size):
                    score += trans_prob[s][policy[s]][s_new] * (
                        reward[s][policy[s]][s_new] + self.gamma * value[s_new]
                    )
                value[s] = score
        return value

    def policy_improvement(self, policy, value, trans_prob, reward):
        policy_stable = True

        for state in range(self.state_size):
            old_action = policy[state]
            val = value[state]
            for action in range(self.action_size):
                score = 0
                for s_new in range(self.state_size):
                    score += trans_prob[state][action][s_new] * (
                        reward[state][action][s_new] + self.gamma * value[s_new]
                    )
                if score > val:
                    policy[state] = action
                    val = score
            if policy[state] != old_action:
                policy_stable = False

        return policy, policy_stable

    def policy_iteration(self, trans_prob, reward, stop_if_stable=False):
        reward_list = []
        num_state = trans_prob.shape[0]

        # init policy and value function
        policy = np.zeros(num_state, dtype=int)
        value = np.zeros(num_state)

        counter = 0
        for episode in range(self.episodes):
            counter += 1
            value = self.policy_evaluation(policy, value, trans_prob, reward)
            policy, stable = self.policy_improvement(policy, value, trans_prob, reward)

            # test the policy for each iteration
            score = self.test_policy(policy)
            reward_list.append(score)

            print(f'Episode: {episode+1:4}/{self.episodes} Reward: {score:4}')

            if stable and stop_if_stable:
                print("policy is stable at {} iteration".format(counter))
                break

        return policy, reward_list


    def test_policy(self, policy):
        state = self.env.reset()
        score = 0
        for _ in range(self.trial):
            self.env.render(render=self.render)
            action = policy[state]
            
            state, reward, done, info = self.env.step(action)

            score += reward
            if done:
                break

        return score


    def get_transitional_probability(self):
        trans_prob = np.zeros((self.state_size, self.action_size, self.state_size))
        reward = np.zeros((self.state_size, self.action_size, self.state_size))
        counter_map = np.zeros((self.state_size, self.action_size, self.state_size))

        counter = 0
        while counter < self.sample:
            state = self.env.reset()
            done = False
            score = 0
            while not done:
                random_action = self.env.action_space.sample()
                new_state, r, done, _ = self.env.step(random_action)
                
                score += r
                trans_prob[state][random_action][new_state] += 1
                reward[state][random_action][new_state] += score

                state = new_state
                counter += 1
                
        # normalization
        for i in range(trans_prob.shape[0]):
            for j in range(trans_prob.shape[1]):
                norm_coeff = np.sum(trans_prob[i, j, :])
                if norm_coeff:
                    trans_prob[i, j, :] /= norm_coeff

        counter_map[counter_map == 0] = 1  # avoid invalid division
        reward /= counter_map

        return trans_prob, reward
    
    def learn(self):
        trans_prob, reward = self.get_transitional_probability()
        policy, reward_list = self.policy_iteration(trans_prob, reward)
        return policy, reward_list
    
    def envaluation(self, policy):
        for episode in range(self.envaluations):
            done = False
            state = self.env.reset()
            score = 0
            while not done:
                self.env.render()
                action = policy[state]
                
                state, reward, done, info = self.env.step(action)

                score += reward
                if done:
                    break
        print(f'Episode: {episode+1:4}/{self.envaluations} Reward: {score:4}')
        return score
