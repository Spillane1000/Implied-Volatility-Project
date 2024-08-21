import yfinance as yf
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as si
from mpl_toolkits.mplot3d import Axes3D

def get_options(Ticker):

    # - Downlowads current price and option chain data from Yahoo finance 
    # - Option data is on European put and call options for all available future expiration dates
    # - Expiration is taken as the end of the relevant day and is given as days to expiration 
    # - Option price taken as last traded price for option with same strike/expiration 

    stock_data = yf.Ticker(Ticker) # Download data for relevant Ticker 
    current_price = (stock_data.info["bid"] + stock_data.info["ask"])/2 # current price as midpoint of bid/ask

    expirations = stock_data.options # all available expiration dates 
    today = datetime.datetime.today()
    call_options = []
    put_options = []
    for exp in expirations:
        calls = pd.DataFrame(stock_data.option_chain(exp).calls) # call option chain
        puts = pd.DataFrame(stock_data.option_chain(exp).puts) # put option chain 
        expiry = datetime.datetime.strptime(exp, "%Y-%m-%d")
        days_to_expiry = (expiry.date() - today.date()).days +1 # add 1 for end of day 

        # create array containing option info
        [call_options.append([days_to_expiry  ,calls["strike"].iloc[i],calls["lastPrice"].iloc[i]]) for i in range(calls.shape[0])]
        [put_options.append([days_to_expiry,puts["strike"].iloc[i],puts["lastPrice"].iloc[i]]) for i in range(puts.shape[0])]

    cols = ["days_to_expiry","strike","last_price"] 
    return pd.DataFrame(call_options,columns = cols),pd.DataFrame(put_options,columns = cols),current_price


def black_scholes(S, K, r, T, sigma):

    # Calculate the Black-Scholes option prices and Vega (Derivative of option price w.r.t the volatility sigma)
    # S - current underlying asset price
    # K - option Strike
    # r - risk free rate
    # T - maturity
    # sigma - volatilty parameter 
    # The Vega is used to calculate the implied volatility in a later step 

    d1 = (np.log(S/K) + (r + (sigma**2)/2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    call_price = si.norm.cdf(d1) * S - si.norm.cdf(d2) * K * np.exp(-r * T)
    put_price = call_price + K * np.exp(-r * T) - S

    N_deriv_d1 = np.exp((-1 * (d1**2)) / 2) / (np.sqrt(2 * np.pi))
    vega = S * N_deriv_d1 * np.sqrt(T)

    return call_price, put_price, vega

def implied_vol(S, K, T, r, market_price, tol=1e-5, max_iters=300,call = True):

    # Uses bisection method to find the implied volatility i.e root of C_0 - BS(S,K,T,r,sigma(K,T)) = 0
    # Upper and lower bound on implied volatility to avoid value errors in algorithm 

    lower_bound = 0.0001
    upper_bound = 5
    sigma = (upper_bound + lower_bound) / 2.0 # initial guess for implied volatility 

    for i in range(max_iters):

        if call:
            price, _, _ = black_scholes(S, K, T, r, sigma) # Black Scholes call price for current sigma

        _, price, _ = black_scholes(S, K, T, r, sigma) # Black-Scholes put price for current sigma

        if abs(price - market_price) < tol:
            return sigma
        
        # if difference between market and BS price is negative take midpoint between upper an lower bounds as new lower bound
        # if positive take midpoint as upper bound
        if price < market_price:
            lower_bound = sigma
        else:
            upper_bound = sigma

        sigma = (upper_bound + lower_bound) / 2.0 # new guess is midpoint of new bounds 

    return sigma 


### Change Ticker as neccessary
apple_calls, apple_puts,price = get_options("AAPL") 

# initialize DataFrame for each implied volatility 
implied_vol_surf = pd.DataFrame(np.zeros(apple_calls.shape),columns = ["days_to_expiry","strike","implied_volatility"])

# loop through all option combinations of strike/maturity (calls in this case) 
for i in range(apple_calls.shape[0]):
    expiry = apple_calls.iloc[i,0]
    strike = apple_calls.iloc[i,1]

    ## Note: risk free rate assumed to be 5% - method for estimating risk free rate (via US treasury bonds) to be included in later version 
    implied_volatility = implied_vol(price,strike,expiry/365, 0.05,apple_calls.iloc[i,2]) #calculates implied vol
    implied_vol_surf.iloc[i,:] = [expiry,strike,implied_volatility]

print(implied_vol_surf)

# remove extreme small values as these are likely 0 and are incossistent (usually very low strikes) 
implied_vol_surf = implied_vol_surf.loc[implied_vol_surf["implied_volatility"] >= 0.0002] 
implied_vol_surf = implied_vol_surf.loc[implied_vol_surf["implied_volatility"] <= 3] 

print(implied_vol_surf)


# 3D scatter plot of implied volatility surface 
fig = plt.figure()
ax1 = fig.add_subplot(111, projection='3d')
ax1.set_xlabel("T")
ax1.set_ylabel("K")
ax1.set_zlabel("Implied Volatility")
ax1.set_title("Implied Volatility Surface ")


Axes3D.scatter(xs = implied_vol_surf.iloc[:,0],ys = implied_vol_surf.iloc[:,1],zs = implied_vol_surf.iloc[:,2],ax = ax1)
#Axes3D.plot_trisurf(X = implied_vol_surf.iloc[:,0],Y = implied_vol_surf.iloc[:,1],Z = implied_vol_surf.iloc[:,2],ax = ax1)
plt.show()


#print(apple_calls)



