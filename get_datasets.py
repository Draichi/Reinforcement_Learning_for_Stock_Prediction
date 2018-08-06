import threading, requests, time, os, datetime
import pandas as pd
from termcolor import cprint
from my_vars import coins, currency, days
#------------------------------------------------------------->
start = time.time()

#------------------------------------------------------------->
def fetch_url(coin):
    if not (os.path.exists('data/{}_{}-days_{}.csv'.format(coin, days, currency))):
        cprint('> downloading {}, {} days dataset, this will take a while'.format(coin, days), 'yellow')
        url = "https://api.coingecko.com/api/v3/coins/{}/market_chart?vs_currency={}&days={}".format(coin, currency, days)
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers)
        df = pd.DataFrame(response.json())
        for i, item in enumerate(df['prices']):
            str_item = str(item)
            current_item = str_item.replace('[', '').replace(']', '').split(',')
            date = current_item[0]
            price = current_item[1]
            dt = datetime.datetime.fromtimestamp(int(date)/1000).strftime('%Y-%m-%d %H:%M:%S')
            df.loc[i, 'date'] = dt
            df.loc[i, coin] = price
        df = df[['date', coin]]
        df.set_index('date', inplace=True)
        df.to_csv('data/{}_{}-days_{}.csv'.format(coin, days, currency))
        cprint("> {} fetched and cached in {} seconds".format(coin, (time.time() - start)), 'green')
    else:
        cprint('> loading {} from cache'.format(coin), 'green')
#------------------------------------------------------------->
threads = [threading.Thread(target=fetch_url, args=(coin,)) for coin in coins]
for thread in threads:
    thread.start()
for thread in threads:
    thread.join()
