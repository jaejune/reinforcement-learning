from time import sleep
import numpy as np
import gym
import time

class QLearning:
    def __init__(self, env, state_size, action_size, episodes, envaluations, render):
        self.env = env
        self.action_size = action_size
        self.state_size = state_size
        self.episodes = episodes
        self.envaluations = envaluations
        self.render = render
        # Init self.Q-Table
        self.Q = {}
        for i in range(state_size):
            self.Q[i] = np.random.rand(action_size)
        # Hyperparameters
        self.lr = 1.0
        self.lrMin = 0.001
        self.lrDecay = 0.999
        self.gamma = 1.0
        self.epsilon = 1.0
        self.epsilonMin = 0.001
        self.epsilonDecay = 0.999

    def learn(self):
        reward_list = list()
        for episode in range(self.episodes):
            state = self.env.reset()
            done = False
            score = 0
            while not done:
                self.env.render(self.render) 
                if np.random.random() < self.epsilon:
                    action = np.random.randint(self.action_size)
                else:
                    action = np.argmax(self.Q[state])

                new_state, reward, done, info = self.env.step(action)
                score += reward
                self.Q[state][action] = self.Q[state][action] + self.lr * (score + self.gamma * np.max(self.Q[new_state]) - self.Q[state][action])
                state = new_state

            if self.lr > self.lrMin:
                self.lr *= self.lrDecay
                
            if self.epsilon > self.epsilonMin:
                self.epsilon *= self.epsilonDecay
                
            print(f'Episode: {episode+1:4}/{self.episodes} Reward: {score:4}')

            reward_list.append(score)

        return reward_list, self.Q
    
    def envaluation(self, Q):
        reward_list = list()
        for episode in range(self.envaluations):
            state = self.env.reset()
            done = False
            score = 0
            while not done:
                self.env.render(True)
                action = np.argmax(Q[state])

                new_state, reward, done, info = self.env.step(action)
                score += reward
                state = new_state

                if done:
                    break
                
            print(f'Episode: {episode+1:4}/{self.envaluations} Reward: {score:4}')

            reward_list.append(score)

        return reward_list, self.Q


