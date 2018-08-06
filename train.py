from agent.agent import Agent
from functions import *
import sys
from termcolor import colored

try:
	if len(sys.argv) != 4:
		print(colored("Usage: python train.py [stock] [window] [episodes]", 'red'))
		exit()

	stock_name, window_size, episode_count = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])

	agent = Agent(window_size)
	data = getStockDataVec(stock_name)
	l = len(data) - 1
	batch_size = 32

	for e in range(episode_count + 1):
		print(colored("Episode {}/{}\n".format(str(e), str(episode_count)), 'yellow'))
		state = getState(data, 0, window_size + 1)

		total_profit = 0
		agent.inventory = []

		for t in range(l):
			action = agent.act(state)

			# sit
			next_state = getState(data, t + 1, window_size + 1)
			reward = 0

			if action == 1: # buy
				agent.inventory.append(data[t])
				print(colored("> {0:.7f}".format(data[t]), 'cyan'))

			elif action == 2 and len(agent.inventory) > 0: # sell
				bought_price = agent.inventory.pop(0)
				reward = max(data[t] - bought_price, 0)
				total_profit += data[t] - bought_price
				print(colored("> {0:.7f} |".format(data[t]), 'yellow'), formatPrice(data[t] - bought_price))

			done = True if t == l - 1 else False
			agent.memory.append((state, action, reward, next_state, done))
			state = next_state

			if done:
				print(colored("\n------TOTAL PROFIT------", 'magenta'))
				print('     '+ formatPrice(total_profit))
				print(colored("------------------------\n",'magenta'))

			if len(agent.memory) > batch_size:
				agent.expReplay(batch_size)

		if e % 10 == 0:
			agent.model.save("models/{}_model_ep{}_w{}_e{}".format(stock_name, str(e), window_size, episode_count))

finally:
	exit()
