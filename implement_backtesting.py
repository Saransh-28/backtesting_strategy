import pandas as pd
from datetime import datetime, timedelta 
import warnings
import numpy as np
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

df = pd.read_csv('data.csv')
df.CreatedOn = pd.to_datetime(df.CreatedOn)

df.dropna(axis=1 , inplace=True)

def clean_data(df , ticker , start_date , end_date):
#     Get the data of the given ticker
    df = df[df['InstrumentIdentifier'] == ticker]
    df.dropna(axis=1,inplace=True)
    
    
#    define a function to round up the seconds to nearest minute
    def round_seconds_to_minutes(dt):
        rounded_minute = dt.minute
        if dt.second >= 30:
            rounded_minute += 1
        return datetime(dt.year, dt.month, dt.day, dt.hour, rounded_minute)
    
    
#   apply the funciton 
    df.CreatedOn = df['CreatedOn'].apply(lambda x: round_seconds_to_minutes(x))
    
    
# drop the duplicates that are generated after rounding 
    df = df.drop_duplicates(subset='CreatedOn')
    
# define function to fill missing value
    def fill_missing_value(df):
        df['CreatedOn'] = pd.to_datetime(df['CreatedOn'])
        df['Date'] = df['CreatedOn'].dt.date
        date_list = list(set(df['Date']))
        final_df = pd.DataFrame(columns = df.columns)
    
#         iterate every date to fill missing value at that particular date and store in a dataframe
        for date in date_list:
            temp = df[df['Date'] == date].copy()
            mi = pd.Timestamp(f'{date} 09:00:00')
            ma = pd.Timestamp(f'{date} 15:30:00')
            time_range = pd.date_range(start=mi, end=ma, freq='1T')
            temp = temp.set_index('CreatedOn').reindex(time_range).interpolate().reset_index()
            final_df = final_df.append(temp)
        return final_df
    
    df = fill_missing_value(df)
    df['CreatedOn'] = df['index']
    df.sort_values(by=['CreatedOn'])
    return df



def strat(df , freq , stop_loss , target_profit , window, std):
    
    ans = pd.DataFrame(columns = ['Date' , 'number of trades' , 'number of winning trades' , 'number of lossing trades' , 'Average gain (winning)' , 'Average loss (lossing)' , 'cumulative returns'])
    df = df[[ 'CreatedOn' , 'CloseValue' , 'OpenValue' ]]
    df['CreatedOn'] = pd.to_datetime(df['CreatedOn'])
    df['Date'] = df['CreatedOn'].dt.date
    date_list = list(set(df['Date']))
    
    
    for date in date_list:
        x = pd.DataFrame()
        temp = df[df['Date'] == date].copy()
        mi = pd.Timestamp(f'{date} 09:00:00')
        ma = pd.Timestamp(f'{date} 15:30:00')
        time_range = pd.DataFrame({'CreatedOn':pd.date_range(start=mi, end=ma, freq=f'{freq[:-1]}T')})
        x = pd.merge(temp, time_range, on="CreatedOn", how="inner")
        
        x['sma'] = x.CloseValue.rolling(window=window).mean()
        x['stddev'] = x.CloseValue.rolling(window=window).std()
        x['Upper'] = x.sma + x.stddev*std
        x['Lower'] = x.sma - x.stddev*std
        
        x['buy'] = np.where(x.Lower > x.CloseValue , True , False)
        x['sell'] = np.where(x.Upper < x.CloseValue , True , False)
        x.dropna(inplace=True)
        summary = pd.DataFrame(columns=['start_time', 'end_time' ,'reason'])
        start_time = []
        end_time = []

        reason = []
        buys = []
        sells = []
        pos = False
        last_buy_price =0
        
        
        for i in range(len(x)):
            if x.Lower.iloc[i] > x.CloseValue.iloc[i] and str(x.CreatedOn.iloc[i]) !=  str(time_range.iloc[-2]) :
                if pos == False:
                    buys.append(i)
                    pos= True
                    last_buy_price = x.CloseValue.iloc[i]
                    start_time.append(str(x.CreatedOn.iloc[i]))
                    
            elif (x.CloseValue.iloc[i] <= (1-stop_loss)*last_buy_price):
                 if pos:
                    sells.append(i)
                    pos= False      
                    reason.append('Stop loss')
                    end_time.append(str(x.CreatedOn.iloc[i]))
            elif (last_buy_price*(1+target_profit) <= x.CloseValue.iloc[i]):
                if pos:
                    sells.append(i)
                    pos= False
                    reason.append('Achieved Target Profit')
                    end_time.append(str(x.CreatedOn.iloc[i]))
            elif  (x.Upper.iloc[i] < x.CloseValue.iloc[i]) :
                if pos:
                    sells.append(i)
                    pos= False
                    reason.append('Upper band')
                    end_time.append(str(x.CreatedOn.iloc[i]))
                    
        if len(buys) != len(sells):
            sells.append(i)
            reason.append('End of Day')
            end_time.append(str(x.CreatedOn.iloc[i]))
        
        
        n_trades = len(buys)
        a = list(x.iloc[buys].CloseValue) 
        b = list(x.iloc[sells].CloseValue)
        c = pd.DataFrame({'buy':a , 'sell':b})
        c['profit'] =100*(c.sell - c.buy)/c.buy
        profits = list(c.profit)
        n_wins = sum(1 for value in profits if value > 0)       
        n_loss = n_trades - n_wins
        max_loss = min(profits) if len(profits) >0 else 0
        avg_gain = sum(value for value in profits if value > 0)/(n_wins)  if n_wins > 0 else 0
        avg_loss = sum(value for value in profits if value < 0)/(n_loss)  if n_loss > 0 else 0
        
        
        m = 1
        for p in profits:
            m *= (1 + p)
        c_return = m
      
        ans = ans.append( { 'Date':date,
          'number of trades':n_trades , 
         'number of winning trades' :n_wins,
         'number of lossing trades' :n_loss,
         'Average gain (winning)':avg_gain , 
         'Average loss (lossing)' :avg_loss,
         'cumulative returns':c_return},ignore_index=True)
        summary['start_time'] = start_time
        summary['end_time'] = end_time
        summary['reason'] = reason
        
        print('-'*20+f'summary for {date} '+'-'*20)
        print(summary)
        
        
        plt.figure(figsize=(10,5))
        plt.title('daily plot with entries and exits shown')
        plt.plot(x[['CloseValue' , 'Upper' , 'Lower']])
        plt.fill_between(x.index , x.Upper , x.Lower , color='grey' , alpha=0.5)
        plt.scatter(x.iloc[buys].index , x.iloc[buys].CloseValue , marker='^' , color='g')
        plt.scatter(x.iloc[sells].index , x.iloc[sells].CloseValue , marker='^' , color='r')
        plt.show()
    return ans


def run(data , ticker , start_date , end_date , freq , stop_loss , target_profit , window, std ):
    clean_dataset = clean_data(df , ticker , start_date, end_date)
    stats = strat(clean_dataset , freq , stop_loss , target_profit  , window, std)
    profits = list(stats['cumulative returns'])
    pnl = []
    temp = 10000
    for i in profits:
        pnl.append(temp*i)
        temp = temp*i

    plt.figure(figsize=(10,5))
    plt.title('Total PnL')
    plt.plot(pnl, marker='o')
    plt.show()
    print('-'*20 +'Stats' + '-'*20)
    print(stats)
    
    
run(df , 'ADANIENT' , '' , '' , '1M' , 0.005 , 0.005 , 20 , 1)
