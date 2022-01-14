import gym
import time
from env import CustomEnv
from dqn import DQN
from qlearning import QLearning
from policy_iteration import PolicyIteration
from matplotlib import pyplot as plt

# 훈련 횟수
EPISODES = 30

# 검증 횟수
ENVALUATIONS = 5

# 시각화를 위한 차트 준비
plt.figure()

# 훈련 알고리즘
algos = ['QLearning', 'PolicyIteration', 'DQN']

# 시뮬레이션 정의 
env = CustomEnv('6x6')

# 알고리즘 정의
dqn = DQN(env, env.state_size, env.action_space.n, episodes=EPISODES, envaluations=ENVALUATIONS,render=False)
qlearning = QLearning(env, env.state_size, env.action_space.n, episodes=EPISODES, envaluations=ENVALUATIONS, render=False)
policy_iteration = PolicyIteration(env, env.state_size, env.action_space.n, episodes=EPISODES, envaluations=ENVALUATIONS, render=False)

# 훈련 시작
for algo in algos:
    print(f'Learning with {algo}')
    time.sleep(1)
    if algo == 'QLearning':
        reward_list, Q = qlearning.learn()
        qlearning.envaluation(Q)
        plt.plot(reward_list, label=algo)

    elif algo == 'PolicyIteration':
        policy, reward_list = policy_iteration.learn()
        policy_iteration.envaluation(policy)
        plt.plot(reward_list, label=algo)

    elif algo == 'DQN':
        reward_list, model = dqn.learn()
        dqn.envaluation(model)

        plt.plot(reward_list, label=algo)
    else:
        pass

# 결과 시각화
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.legend(loc='best')
plt.show()
    