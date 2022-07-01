import gym
import time
from env import CustomEnv
from dqn import DQN
from qlearning import QLearning
from policy_iteration import PolicyIteration
from matplotlib import pyplot as plt


# 훈련 횟수
EPISODES = 100

# 검증 횟수
ENVALUATIONS = 5

# 시뮬레이션 시각화
RENDER = False

# 시각화를 위한 차트 준비
plt.figure()

# 훈련 알고리즘
algos = ['QLearning', 'PolicyIteration', 'DQN']

# 시뮬레이션 정의 
env = CustomEnv('map')

# 환경 테스트 코드 (삭제 예정)
# ================================================
# for i in range(10):
#     done = False
#     env.reset()
#     while not done:
#         env.render()
#         action = env.action_space.sample()
#         state, _, done, _ = env.step(action)
# ================================================

    
# 알고리즘 정의
dqn = DQN(env, env.state_size, env.action_space.n, episodes=EPISODES, envaluations=ENVALUATIONS,render=RENDER)
qlearning = QLearning(env, env.state_size, env.action_space.n, episodes=EPISODES, envaluations=ENVALUATIONS, render=RENDER)
policy_iteration = PolicyIteration(env, env.state_size, env.action_space.n, episodes=EPISODES, envaluations=ENVALUATIONS, render=RENDER)

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
    