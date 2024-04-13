# -*- coding: utf-8 -*-

# Candidate Number:AC07188
# Do not enter Name

#%% QUESTION2: Backtest RSI Strategy



#%% Import libraries
import yfinance as yf 
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

#%% Set dir and path
path='D:\\program code\\2nd year\\individual project 2022\\'
os.chdir(path)
os.getcwd()
#%% a) Download FB between 2020-01-01 and 2022-01-01 using yahoo finance library.
# Alternatively, load the FB data from the excel file provided in folder Exercise2_Data.
tickers=['AAPL']
start='2020-01-01'
end='2022-01-01'
FB_data=yf.download(tickers, start=start,end=end)




#%% b) Download SPY benchmark for same dates as above in part a using yahoo finance library.
# Alternatively, load the SPY data from the excel file provided in folder Exercise2_Data
tickers=['SPY']
start='2020-01-01'
end='2022-01-01'
SPY_data=yf.download(tickers, start=start,end=end)

#%%  c) Extract FB Adjusted Close and create a new DataFrame called close.
close=FB_data['Adj Close'].to_frame()
close.head()

#%% d) Write a function to calculate Wilder’s smoothing RSI on the FB Adjusted Close (See Screenshot in slides for mathematics). Save these results to the DataFrame called close.

N=14 # deflaut setting 
def RSI_avg_up(df,N,colunm_name='Adj Close'):
    close_up=pd.DataFrame(index=df.index)
    close_up['avg_up']=0
    close_up['up_close']=np.where(df[colunm_name]>df[colunm_name].shift(1),df[colunm_name]-df[colunm_name].shift(1),
                                  np.where(close_up['avg_up'].index==close_up['avg_up'].index[0],np.nan,0))
    #the obove process is to maitain the first value is np.nan and make it easy in rebasing when calculate the average up 
    close_up.dropna(inplace=True)# rebasing for calculate the 1st simple moving average
    for i in range(len(close_up.index)):
        if i<N-1:
            close_up['avg_up'].iloc[i]=np.nan
        if i==N-1:
            close_up['avg_up'].iloc[N-1]=close_up['up_close'].iloc[:N].mean()
        if i>=N:
            close_up['avg_up'].iloc[i]=((close_up['avg_up'].iloc[i-1]*(N-1))+close_up['up_close'].iloc[i])/N
    return close_up['avg_up']
    
def RSI_avg_down(df,N,colunm_name='Adj Close'):
    close_down=pd.DataFrame(index=df.index)
    close_down['avg_down']=0
    close_down['down_close']=np.where(df[colunm_name]<df[colunm_name].shift(1),df[colunm_name].shift(1)-df[colunm_name],
                                      np.where(close_down['avg_down'].index==close_down['avg_down'].index[0],np.nan,0))
    close_down.dropna(inplace=True)        
    for i in range(len(close_down.index)):
        if i<N-1:
            close_down['avg_down'].iloc[i]=np.nan
        if i==N-1:
            close_down['avg_down'].iloc[N-1]=close_down['down_close'].iloc[:N].mean()
        if i>=N:
            close_down['avg_down'].iloc[i]=((close_down['avg_down'].iloc[i-1]*(N-1))+close_down['down_close'].iloc[i])/N
    return close_down['avg_down']
    
def RSI(df,window,colunm_name='Adj Close'):
    RSI_df=pd.DataFrame(index=df.index)
    ratio=(RSI_avg_up(df,N=window)/RSI_av)g_down(df,N=window)).to_frame()
    RSI_df['RSI']=ratio.apply(lambda x: 100-(100/(1+x))
    return RSI_df['RSI']

close['RSI']=RSI(close,window=N)

#%% e) Calculate the signals based off the below condition: (4 marks)
# RSI < 30 = BUY
# RSI > 70 = SELL
# *Note: 30 & 70 are the default parameters.
# N = 14 (setting default window)

    
def RSI_signal(dataframe,buy,sell,columns='RSI'):
    new_df=pd.DataFrame(index=dataframe.index)
    new_df['signal']=np.where(dataframe['RSI']<buy,"BUY",np.where(dataframe['RSI']>sell,'SELL',np.nan))                       
    return new_df['signal']

buy=30
sell=70
close['signal']=RSI_signal(close,buy=buy,sell=sell)

close['position']=np.where(close['signal']=='BUY',1,np.where(close['signal']=='SELL',-1,np.nan))
close['position'].ffill(inplace=True)
close['position'].fillna(0,inplace=True)

close['diff']=close['position']-close['position'].shift(1)
close['b_Price']=np.where(close['position'].diff()>0,close['Adj Close'],np.nan)
close['s_Price']=np.where(close['position'].diff()<0,close['Adj Close'],np.nan)



#%% f) Plot RSI signal and graph adjusted stock close price in separate plots. Save graph.





plt.figure(figsize=(12, 8))
plt.subplot(211)
plt.plot(close['Adj Close'], 'b', label = "price")
plt.legend(loc=4)
plt.grid(True)
plt.xlabel('Date')
plt.ylabel('close_price')
plt.title('close_price_of_FB')
plt.plot(close.index,close['b_Price'], marker='^', color='green', label = "buy")
plt.plot(close.index,close['s_Price'], marker='v', color='r', label = "sell")


plt.subplot(212)
plt.plot(close['position'], 'r', label = "signal") 
plt.legend(loc=4)
plt.grid(True)
# Put on axis
plt.xlabel('Date')
plt.ylabel('signal')
# Put on title
plt.title('signal_of_RSI_70_30')
plt.plot(close.index,np.where(close['b_Price']>0,close['position'],np.nan), marker='^', color='green', label = "buy")
plt.plot(close.index,np.where(close['s_Price']>0,close['position'],np.nan), marker='v', color='r', label = "sell")
#Save graph

#Save graph
plt.savefig('2f_results')



#%% g)  Calculate the log returns for adjusted close for the stock and the benchmark
def log_ret(dataframe,columns='Adj Close'):
    new_df=pd.DataFrame(index=dataframe.index)
    new_df['log_ret']=np.log(dataframe[columns] / dataframe[columns].shift(1)) #np.log() has already set up as calculaing natural log return
    return new_df['log_ret']
close['log_ret']=log_ret(close)
SPY_data['log_ret']=log_ret(SPY_data)
    
    



#%% h) Calculate the strategy returns
"""
The basic idea is that the algorithm can only set up a position in the  stock given today’s
market data (e.g., just before the close). The position then earns tomorrow’s return.
"""


close['strategy_ret']=close['position'].shift(1)*close['log_ret']




#%% i) Calculate cumualtive returns for buy and hold the stock, the strategy and the benchmark
# Double check your result with various approaches.
close['log_ret_cumsum']=np.exp(close['log_ret'].cumsum())
buy_hold_stock=close['log_ret_cumsum'].iloc[-1]
close['strategy_ret_cumsum']=np.exp(close['strategy_ret'].cumsum())
buy_hold_strategy=close['strategy_ret_cumsum'].iloc[-1]
SPY_data['log_ret_cumsum']=np.exp(SPY_data['log_ret'].cumsum())
buy_hold_benchmark=SPY_data['log_ret_cumsum'].iloc[-1]


#check

test1=np.exp(close['log_ret'].sum())
test2=np.exp(close['strategy_ret'].sum())
test3=np.exp(SPY_data['log_ret'].sum())




print('cumualtive returns for buy and hold of the stock is:',buy_hold_stock,'\ntest:',test1)
print(' cumualtive returns for buy and hold of the strategy is:',buy_hold_strategy,'\ntest:',test2)
print(' cumualtive returns for buy and hold of the benchmark is:',buy_hold_benchmark,'\ntest:',test3)



#%% j) Plot cumulative returns from the log returns for buy and hold the stock,
# the strategy and the benchmark



plt.figure(figsize=(12, 8))
plt.plot(close['log_ret_cumsum'], 'b', label = "stock_b_h_ret")
plt.plot(close['strategy_ret_cumsum'], 'r', label = "stra_b_h_ret")
plt.plot(SPY_data['log_ret_cumsum'], 'g', label = "bench_b_h_ret")
plt.legend(loc=4)
plt.grid(True)
# Put on axis
plt.xlabel('Date')
plt.ylabel('return')
# Put on title
plt.title('return_comparement')
#Save graph
plt.savefig('2j_results')



#%% k) Calculate descriptive statistics on the stock, the strategy and benchmark returns.
# Save to a DataFrame.
stats_df=pd.DataFrame()
stats_df['stock_ret']=(close['log_ret_cumsum']).describe()
stats_df['strat_ret']=(close['strategy_ret_cumsum']).describe()
stats_df['bench_ret']=(SPY_data['log_ret_cumsum']).describe()

stats_df['stock_ret2']=(close['log_ret']).describe()
stats_df['strat_ret2']=(close['strategy_ret']).describe()
stats_df['bench_ret2']=(SPY_data['log_ret']).describe()



#%% l) Optimise the RSI with the below condition ranges:

    
"""
Optimise the RSI with the below condition ranges: (6 marks)
rsi_buy between 0 and 30 with increment 1
rsi_sell between 70 and 100 with increment 1
n_window between 2 and 21 with increment 1
Hint: Due to computational time, test optimal parameters with increment 10 first.
Time the optimisation in seconds and minutes and print to screen.
The optimised results should generate a DataFrame showing the:
RSI Buy, RSI Sell, N Window, market returns, strategy returns and outperformance.
Note: Outperformance is Strategy Returns – Market Returns

"""
from itertools import product
range_buy=range(0,30,1)
range_sell=range(70,100,1)

range_window=range(2,21,1)
results = pd.DataFrame()
import time
start = time.time()
## ideally extracting un-change column out of the loop would reduce the time for running the loop.
df_optimize=pd.DataFrame()
df_optimize['log_ret']=log_ret(close)
df_optimize['market_log_ret']=log_ret(SPY_data)
market_total_ret=np.exp(df_optimize['market_log_ret'].sum())

for BUY,SELL,WINDOW in product(range_buy,range_sell,range_window):
    df_optimize['RSI']=RSI(close,window=WINDOW)
    df_optimize['signal']=RSI_signal(df_optimize,buy=BUY,sell=SELL)
    df_optimize['position']=np.where(df_optimize['signal']=='BUY',1,np.where(df_optimize['signal']=='SELL',-1,np.nan))
    df_optimize['position'].ffill(inplace=True)
    df_optimize['position'].fillna(0,inplace=True)
    df_optimize['strategy_ret']=df_optimize['position'].shift(1)*df_optimize['log_ret']
    strat_total_ret=np.exp(df_optimize['strategy_ret'].sum())
    results = results.append(pd.DataFrame(
        {'BUY': BUY, 'SELL': SELL,'WINDOW':WINDOW,
         'MARKET': market_total_ret,
         'STRATEGY': strat_total_ret,
         'OUTPERFORMANCE': strat_total_ret - market_total_ret},
        index=[0]), ignore_index=True)  # Records the vectorized backtesting results in a DataFrame object.
output = "%.2f" % (time.time() - start)
output = float(output)

print("Optimising parameters has taken:", output, "seconds")
print("Optimising parameters has taken:", round(output / 60,2), "minutes")




'''
Optimising parameters has taken: 2446.84 seconds
Optimising parameters has taken: 40.78 minutes
'''

#%% m) Sort the optimised parameter results on outperformance
results.sort_values(by=['OUTPERFORMANCE'],ascending=False,inplace=True)


#%% n) Extract the optimal parameters

best_parameters=results.iloc[0].to_frame()
print(best_parameters)

#%% o)  Rerun the optimal parameter strategy.

# Plot the RSI and signals and cumulative return graphs.

parametet_list=best_parameters.index.tolist()

# convert it to int so that is can be recognized by the function when using iloc,if it is a float it would not the recogized.
buy=int(best_parameters.iloc[parametet_list.index('BUY'),0])
sell=int(best_parameters.iloc[parametet_list.index('SELL'),0])
N=int(best_parameters.iloc[parametet_list.index('WINDOW'),0]) 

best_strat=pd.DataFrame(index=close.index)
best_strat['RSI']=RSI(close,window=N)
best_strat['signal']=RSI_signal(best_strat,buy=buy,sell=sell)

best_strat['position']=np.where(best_strat['signal']=='BUY',1,np.where(best_strat['signal']=='SELL',-1,np.nan))
best_strat['position'].ffill(inplace=True)
best_strat['position'].fillna(0,inplace=True)

best_strat['b_Price']=np.where(best_strat['position'].diff()>0,close['Adj Close'],np.nan)
best_strat['s_Price']=np.where(best_strat['position'].diff()<0,close['Adj Close'],np.nan)


best_strat['log_ret']=log_ret(close)

best_strat['strategy_ret']=best_strat['position'].shift(1)*best_strat['log_ret']

# Re-calculate the cumulative performances using the optimal parameters.

best_strat['stock_b&h_ret']=np.exp(best_strat['log_ret'].cumsum())
best_strat['strategy_ret_cumsum']=np.exp(best_strat['strategy_ret'].cumsum())

## plt 



plt.figure(figsize=(12, 10))
plt.subplot(211)
plt.plot(best_strat['strategy_ret_cumsum'], 'b', label = "price")
plt.legend(loc=4)
plt.grid(True)
plt.xlabel('Date')
plt.ylabel('strat_ret')
plt.title('strat_ret')
plt.plot(best_strat.index,np.where(best_strat['b_Price']>0,best_strat['strategy_ret_cumsum'],np.nan),
         marker='^', color='green', label ="buy")
plt.plot(best_strat.index,np.where(best_strat['s_Price']>0,best_strat['strategy_ret_cumsum'],np.nan),
         marker='v', color='red', label ="sell")


plt.subplot(212)
plt.plot(best_strat['RSI'], 'b', label = "signal") 
plt.legend(loc=4)
plt.grid(True)
plt.xlabel('Date')
plt.ylabel('signal')
plt.title('RSI_window_best')

plt.plot(best_strat.index,np.where(best_strat['b_Price']>0,best_strat['RSI'],np.nan), marker='^', color='green', label ="buy")
plt.plot(best_strat.index,np.where(best_strat['s_Price']>0,best_strat['RSI'],np.nan), marker='v', color='red', label ="sell")
plt.savefig('2o_results')







#%% p)  Isolate the optimal strategy returns and calculate the below performance statistics on this
# strategy: Assume rf = 0and 252 days. Format to 2 decimal places (dp).
# Write functions and store all results in a DataFrame and save to excel.
# Do not use a library for calcualtions.


#deflaut setting 
rf=0
n=252
strat=best_strat['strategy_ret'].to_frame()
def sharpe (df,rf,n):
    return_strat=df['strategy_ret'].mean()*n
    stand_dev=df['strategy_ret'].std()*(n**0.5)
    return (return_strat-rf)/stand_dev

# i) Sharpe Ratio
sharpe_strat=round(sharpe(strat,rf=rf,n=n),2)
# i) Sharpe Ratio

# ii) Sortino Ratio

def sortino (df,rf):
    return_strat=df['strategy_ret'].sum()
    down_df=np.where(df['strategy_ret']<0,df['strategy_ret'],0)
    std_down=down_df.std()
    return (return_strat-rf)/std_down

sortino_ratio=round(sortino(strat, rf=rf),2)

# iii) Compound Annual Growth Rate  (CAGR)

def CAGR(df,n=252):
    return_strat=df['strategy_ret'].sum()
    num_years=len(strat.index)/n
    return (return_strat**(1/num_years)-1)*100
 
cagr_strat=round(CAGR(strat),2)


# iv) Annual Volatility
def annual_vol (df,n=252):
    var_strat=df['strategy_ret'].var()
    return (var_strat**1/2)*(n**1/2)
annual_vol_strat=round(annual_vol(strat),2)


# vi) Maximum Drawdown
def max_drawdown(df,n):
    max_ret=df['strategy_ret'].cumsum().rolling(n,min_periods=1).max()
    
    drawdown=(df['strategy_ret'].cumsum()-max_ret)
    max_drawdown=(drawdown.rolling(n,min_periods=1).min()).min()
    return max_drawdown
max_dd_strat=round(max_drawdown(strat, n=n),2)






# v) Calmar Ratio
def calmar_ratio(df,rf,n):
    return_strat=df['strategy_ret'].sum()
    max_dd=max_drawdown(df, n=n)
    return (return_strat-rf)/max_dd
calmar_strat=round(calmar_ratio(strat,rf=rf,n=n),2)




# vii) Skewness (4dp)

def skewness (df):
    mean=df['strategy_ret'].mean()
    
    std=df['strategy_ret'].std()
    up=(((df['strategy_ret']-mean)/std)**3).sum()
    n=len(df['strategy_ret'])
    k=n/((n-1)*(n-2))
    return up*k
skew_strat=round(skewness(strat),2)
# viii) Kurtosis  (4dp)

def kurtosis (df):
    mean=df['strategy_ret'].mean()
    up=((df['strategy_ret']-mean)**4).sum()
    down=((df['strategy_ret']-mean**2)**2).sum()
    n=len(df['strategy_ret'])
    k=(n*(n+1))/((n-1)*(n-2)*(n-3))
    m=(3*(n-1)**2)/((n-2)*(n-3))
    return k*up/down-m
kurt_strat=round(kurtosis(strat),2)


perf_stats=pd.DataFrame()
perf_stats = perf_stats.append(pd.DataFrame(
        {'Sharpe_Ratio': sharpe_strat, 'Sortino Ratio': sortino_ratio,'CAGR':cagr_strat,
         'Annual_Volatility': annual_vol_strat,
         'Calmar_Ratio': calmar_strat,
         'Skewness': skew_strat,
         'Kurtosis':kurt_strat}, 
        index=[0]), ignore_index=True)
print(perf_stats)

perf_stats.to_excel('2p_results.xlsx')

#%%  q) Calculate the number of total trades, long trades and short trades for optimal strategy.
# Save as a DataFrame.

# Total Trades
buy_count=best_strat['b_Price'].notnull().sum()
sell_count=best_strat['s_Price'].notnull().sum()

total=buy_count+sell_count

# Total Longs

buy_count=best_strat['b_Price'].notnull().sum()




# Total Shorts
sell_count=best_strat['s_Price'].notnull().sum()

data={'Total_Trades':total,'Total_Longs':buy_count,'Total_Shorts':sell_count}


start_trade_sum=pd.DataFrame(data,index=[0])



#%% r) Plot a histogram optimal strategy returns vs benchmark returns del


his_df=pd.DataFrame(index=best_strat.index)
his_df['strat_ret']=best_strat['strategy_ret']
his_df['market_ret']=SPY_data['log_ret']



plt.figure(figsize=(10, 6))
plt.hist(his_df, label=['strat', 'market'], color=['b', 'g'],stacked=True, bins=20, alpha=0.5)
plt.legend(loc=0)
plt.xlabel('returns of strat and market')
plt.ylabel('frequency')
plt.title('Histogram')
plt.savefig('2r_results') 









#%%