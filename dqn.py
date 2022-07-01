from hashlib import new
import gym
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib
import warnings
warnings.filterwarnings("ignore")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from collections import deque


class DQN:
    def __init__(self, env, state_size, action_size, episodes, envaluations, render):
        self.env = env
        self.action_size = action_size
        self.state_size = state_size
        self.episodes = episodes
        self.envaluations = envaluations
        self.render = render

        # Hyperparameters
        self.memory = deque(maxlen=2500)
        self.learning_rate=0.001
        self.epsilon = 1
        self.max_eps = 1
        self.min_eps = 0.01
        self.eps_decay = 0.001/3
        self.gamma = 0.9
        self.state_size= state_size
        self.action_size= action_size
        self.epsilon_lst=[]
        self.model = self.buildmodel()
        self.batch_size=32


    def buildmodel(self):
        model=Sequential()
        model.add(Dense(10, input_dim=self.state_size, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def add_memory(self, new_state, reward, done, state, action):
        self.memory.append((new_state, reward, done, state, action))

    def get_action(self, state, action_size):
        if np.random.rand() < self.epsilon:
            return np.random.randint(action_size)
        return np.argmax(self.model.predict(state))

    def pred(self, state):
        return np.argmax(self.model.predict(state))

    def replay(self, batch_size, episode):
        minibatch=random.sample(self.memory, batch_size)
        for new_state, reward, done, state, action in minibatch:
            target= reward
            if not done:
                target=reward + self.gamma* np.amax(self.model.predict(new_state))
            target_f= self.model.predict(state)
            target_f[0][action]= target
            self.model.fit(state, target_f, epochs=1, verbose=0)

        if self.epsilon > self.min_eps:
            self.epsilon=(self.max_eps - self.min_eps) * np.exp(-self.eps_decay*episode) + self.min_eps

        self.epsilon_lst.append(self.epsilon)

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

    def learn(self):
        reward_list = list()
        for episode in range(self.episodes):
            state = self.env.reset()
            state_array = np.zeros(self.state_size)
            state_array[state] = 1
            state = np.reshape(state_array, [1, self.state_size])
            score = 0
            done = False
            while not done:
                self.env.render(self.render)
                action = self.get_action(state, self.action_size)
                
                new_state, reward, done, info = self.env.step(action)
                score += reward
                new_state_array = np.zeros(self.state_size)
                new_state_array[new_state] = 1
                new_state = np.reshape(new_state_array, [1, self.state_size])
                self.add_memory(new_state, score, done, state, action)
                state = new_state

                if done:
                    break

            print(f'Episode: {episode+1:4}/{self.episodes} Reward: {score:4}')

            reward_list.append(score)
            
            if len(self.memory) > self.batch_size:
                self.replay(self.batch_size, episode)

        return reward_list, self.model

    def envaluation(self, model):
        for episode in range(self.envaluations):
            state = self.env.reset()
            state_array = np.zeros(self.state_size)
            state_array[state] = 1
            state = np.reshape(state_array, [1, self.state_size])
            score = 0
            done = False
            while not done:
                self.env.render(True)
                action = np.argmax(model.predict(state))
                
                new_state, reward, done, info = self.env.step(action)
                score += reward
                new_state_array = np.zeros(self.state_size)
                new_state_array[new_state] = 1
                new_state = np.reshape(new_state_array, [1, self.state_size])
                state = new_state

                if done:
                    break

            print(f'Episode: {episode+1:4}/{self.envaluations} Reward: {score:4}')


# if __name__ == '__main__':
#     agent = DQN(state_size, action_size)

#     reward_lst=[]
#     for episode in range(train_episodes):
#         state= env.reset()
#         state_arr=np.zeros(state_size)
#         state_arr[state] = 1
#         state= np.reshape(state_arr, [1, state_size])
#         reward = 
#         done = False
#         for t in range(max_steps):
#             # env.render()
#             # action = agent.action(state, self.action_si)
#             new_state, reward, done, info = env.step(action)
#             new_state_arr = np.zeros(state_size)
#             new_state_arr[new_state] = 1
#             new_state = np.reshape(new_state_arr, [1, state_size])
#             agent.add_memory(new_state, reward, done, state, action)
#             state= new_state

#             if done:
#                 print(f'Episode: {episode:4}/{train_episodes} Reward: {reward:4}')
#                 break

#         reward_lst.append(reward)

#         if len(agent.memory)> batch_size:
#             agent.replay(batch_size)

#     print(' Train mean % score= ', round(100*np.mean(reward_lst),1))

#     # test
#     test_wins=[]
#     for episode in range(test_episodes):
#         state = env.reset()
#         state_arr=np.zeros(state_size)
#         state_arr[state] = 1
#         state= np.reshape(state_arr, [1, state_size])
#         done = False
#         reward=0
#         state_lst = []
#         state_lst.append(state)
#         print('******* EPISODE ',episode, ' *******')

#         for step in range(max_steps):
#             action = agent.pred(state)
#             new_state, reward, done, info = env.step(action)
#             new_state_arr = np.zeros(state_size)
#             new_state_arr[new_state] = 1
#             new_state = np.reshape(new_state_arr, [1, state_size])
#             state = new_state
#             state_lst.append(state)
#             if done:
#                 print(reward)
#                 # env.render()
#                 break

#         test_wins.append(reward)
#     env.close()

#     print(' Test mean % score= ', int(100*np.mean(test_wins)))

#     fig=plt.figure(figsize=(10,12))
#     matplotlib.rcParams.clear()
#     matplotlib.rcParams.update({'font.size': 22})
#     plt.subplot(311)
#     plt.scatter(list(range(len(reward_lst))), reward_lst, s=0.2)
#     plt.title('4x4 Frozen Lake Result(DQN) \n \nTrain Score')
#     plt.ylabel('Score')
#     plt.xlabel('Episode')

#     plt.subplot(312)
#     plt.scatter(list(range(len(agent.epsilon_lst))), agent.epsilon_lst, s=0.2)
#     plt.title('Epsilon')
#     plt.ylabel('Epsilon')
#     plt.xlabel('Episode')

#     plt.subplot(313)
#     plt.scatter(list(range(len(test_wins))), test_wins, s=0.5)
#     plt.title('Test Score')
#     plt.ylabel('Score')
#     plt.xlabel('Episode')
#     plt.ylim((0,1.1))
#     plt.savefig('result.png',dpi=300)
#     plt.show()