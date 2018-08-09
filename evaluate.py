import sys
from termcolor import colored

if len(sys.argv) != 3:
	print(colored("Usage: python evaluate.py [stock] [model]", 'red', attrs=['bold']))
	exit()
import keras, os
from keras.models import load_model
from agent.agent import Agent
from functions import *
from my_vars import days, currency

width = os.get_terminal_size().columns
asset_name, model_name = sys.argv[1], sys.argv[2]
model = load_model("models/" + model_name)
window_size = model.layers[0].input.shape.as_list()[1]
asset = '{}_d{}_{}'.format(asset_name, days, currency)	

agent = Agent(window_size, True, model_name)
data = getStockDataVec(asset)
l = len(data) - 1
batch_size = 32
state = getState(data, 0, window_size + 1)
total_profit = 0
agent.inventory = []

print(chr(27) + "[2J")
print(colored('{}/{}'.format(asset_name.upper(), currency.upper()).center(width), 'blue'))
print(colored('Model {}\n\n'.format(model_name).center(width), 'blue'))
print(colored('From 1 ~~> {} {:.7f}'.format(currency.upper(),data[0]).center(width), 'magenta'))	
print(colored('To {} ~~> {} {:.7f}\n\n'.format(l,currency.upper(),data[-1]).center(width), 'magenta'))	

for t in range(l):
	action = agent.act(state)
	next_state = getState(data, t + 1, window_size + 1)
	reward = 0
	print("> {} BTC {:.7f}".format(t, data[t]), end='\r') #hold
	if action == 1: # buy
		agent.inventory.append(data[t])
		print(colored("> {} BTC {:.7f} |".format(t, data[t]), 'cyan'), formatPrice(total_profit), end='\r')
	elif action == 2 and len(agent.inventory) > 0: # sell
		bought_price = agent.inventory.pop(0)
		reward = max(data[t] - bought_price, 0)
		total_profit += data[t] - bought_price
		print(colored("> {} BTC {:.7f} |".format(t, data[t]), 'yellow'), formatPrice(total_profit), end='\r')
	done = True if t == l - 1 else False
	agent.memory.append((state, action, reward, next_state, done))
	state = next_state
	if done:
		print('\n\n')
		print(colored("-----------------------------".center(width), 'cyan'))
		print(formatPrice(total_profit).center(width))
		print(colored("-----------------------------".center(width),'cyan'))
	if len(agent.memory) > batch_size:
		agent.expReplay(batch_size)
print(colored('D O N E'.center(width),'white','on_green',attrs=['bold']))