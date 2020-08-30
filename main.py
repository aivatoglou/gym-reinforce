import gym
import torch
from collections import Counter
import matplotlib.pyplot as plt
from torch.distributions import Categorical
from policy import Policy

# hyperparameters
learning_rate = 0.003
gamma         = 0.99
hidden        = 128
episodes      = 10000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Device: ', device)

env = gym.make('MountainCar-v0') # load classic game
env._max_episode_steps = 250 # change steps limiit

# Get the obersvation space and the number of actions from the game.
observation_space = env.observation_space.shape[0]
action_space = env.action_space.n

# Define the model architecture
agent = Policy(observation_space, action_space, hidden, learning_rate, gamma).to(device)
scores = []
losses = []
actions = []

high_score = 0
for n_epi in range(episodes):
	observation = env.reset()
	done = False
	score = 0.0

	while not done: 
	# Game exits on steps limit.
		
		# run for 250 steps
		prob = agent(torch.from_numpy(observation).float().to(device))
		m = Categorical(prob)
		action = m.sample()
		observation, reward, done, info = env.step(action.item())
		new_reward = agent.get_reward(observation[0])
		agent.put_data((new_reward,prob[action]))
		score += new_reward
		actions.append(action.item())

		# render the game.
		if n_epi % 100 == 0 :
			env.render()

	if score > high_score:
		print("*** New highscore! ***")
		high_score = score

	if done:
		scores.append(score)
		if info['TimeLimit.truncated'] == True:
			response = 'Step limit maxed.'
		print("# of episode: {}, score: {}".format(n_epi, score))
		if observation[0] >= 0.5: # 0.5 is the point of the flag!
			print('Success!')
			break;
	losses.append(agent.train(device))

print('Completed!')
fig, axs = plt.subplots(3)
fig.suptitle('Results.')

counter = Counter(actions)
axs[0].bar(counter.keys(), counter.values())
axs[0].set_title('Actions preferred')

axs[1].plot(scores)
axs[1].set_title('Reward-Episodes')

axs[2].plot(losses)
axs[2].set_title('Loss-Episodes')

plt.show()
env.close()
