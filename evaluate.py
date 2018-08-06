import sys
from termcolor import colored

try:
	if len(sys.argv) != 3:
		print(colored("Usage: python evaluate.py [stock] [model]", 'red', attrs=['bold']))
		exit()
	import keras
	from keras.models import load_model
	from agent.agent import Agent
	from functions import *

	stock_name, model_name = sys.argv[1], sys.argv[2]
	model = load_model("models/" + model_name)
	window_size = model.layers[0].input.shape.as_list()[1]

	agent = Agent(window_size, True, model_name)
	data = getStockDataVec(stock_name)
	l = len(data) - 1
	batch_size = 32

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
			print(colored("> BTC {0:.7f}".format(data[t]), 'cyan'))


		elif action == 2 and len(agent.inventory) > 0: # sell
			bought_price = agent.inventory.pop(0)
			reward = max(data[t] - bought_price, 0)
			total_profit += data[t] - bought_price
			print(colored("> BTC {0:.7f} |".format(data[t]), 'yellow'), formatPrice(data[t] - bought_price))


		done = True if t == l - 1 else False
		agent.memory.append((state, action, reward, next_state, done))
		state = next_state

		if done:
			print(colored("\n\n\n\n\n------TOTAL PROFIT------\n", 'cyan'))
			print(stock_name + formatPrice(total_profit))
			print(colored("\n------------------------\n",'cyan'))

		if len(agent.memory) > batch_size:
			agent.expReplay(batch_size)

finally:
	print(colored('               D O N E              ', 'white', 'on_green', attrs=['bold']))
	exit()