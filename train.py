import sys
from termcolor import colored

if len(sys.argv) != 4:
	print(colored("Usage: python train.py [asset] [window] [episodes]", 'red'))
	exit()
import os
from agent.agent import Agent
from functions import getState, getStockDataVec, sigmoid
from my_vars import days, currency




asset_name = sys.argv[1]
window_size, episode_count = int(sys.argv[2]), int(sys.argv[3])

width = os.get_terminal_size().columns
asset = '{}_d{}_{}'.format(asset_name, days, currency)	

agent = Agent(window_size)
data = getStockDataVec(asset)
l = len(data) - 1
batch_size = 32

# prints formatted price
def formatPrice(n):
	if n < 0:
		return colored('Total profit: -BTC {0:.6f}'.format(abs(n)), 'red', attrs=['bold'])
	else:
		return colored('Total profit: BTC {0:.7f}'.format(abs(n)), 'green', attrs=['bold'])
	# return ("-BTC " if n < 0 else "BTC ") + "{0:.7f}".format(abs(n))



print(chr(27) + "[2J")
print(colored('{}/{}'.format(asset_name.upper(), currency.upper()).center(width), 'blue'))
print(colored('Trainning \n\n'.center(width), 'blue'))
print(colored('From 1 ~~> {} {:.7f}'.format(currency.upper(),data[0]).center(width), 'magenta'))	
print(colored('To {} ~~> {} {:.7f}\n\n'.format(l,currency.upper(),data[-1]).center(width), 'magenta'))	

for e in range(episode_count+1):
    print(colored("      {}/BTC".format(asset_name).center(width), 'yellow', attrs=['bold']))
    print(colored("> Episode {}/{}\n".format(str(e), str(episode_count)).center(width), 'yellow',  attrs=['bold']))
    state = getState(data, 0, window_size + 1)
    total_profit = 0
    agent.inventory = []
    for t in range(l):
        action = agent.act(state)
        next_state = getState(data, t + 1, window_size + 1)
        reward = 0
        print("> {} BTC {:.7f}".format(t, data[t]), end='\r') #hold
        if action == 1: # buy
            agent.inventory.append(data[t])
            print(colored("> {} BTC {:.7f} |".format(t, data[t]), 'cyan'), end='\r')
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
    if e % 10 == 0:
        agent.model.save("models/{}_d{}_e{}_w{}_c{}".format(days, asset_name, str(e), window_size, episode_count))
print(colored('D O N E'.center(width),'white','on_green',attrs=['bold']))