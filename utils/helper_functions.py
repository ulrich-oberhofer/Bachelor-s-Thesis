import pandas as pd
import numpy as np
import h5py

from utils import settings as s


def get_frequency_data(country: str) -> pd.DataFrame:
    """
    Returns frequency data for a given country without NaN values and only those hours
    that have no missing measurements
    """
    if country == 'AUS':
        #df = pd.read_hdf('../data/AUS_cleansed_frequency_data.h5') #!!! this is replaced
        df = pd.read_pickle('../data/Mainland_AUS_frequency_data_2021-2024.pkl') #!!! this is added
        df = pd.DataFrame({'frequency': df})
        df = df.loc['2023-02-12':'2024-03-24']
    elif country == 'CE':
        # ce_data = h5py.File(f'../data/CE_cleansed_2015-01-01_to_2019-12-31.h5', 'r')['df']
        # df = pd.DataFrame({'timestamp': ce_data['index'], 'frequency': ce_data['values']})
        ce_data = pd.read_hdf(f'../data/cleansed_2015-01-01_to_2019-12-31.h5')#, key='df') #!!! this is added	
        #df = pd.DataFrame({'timestamp': ce_data.index, 'frequency': ce_data.values}) #!!! this is added	
        #df = ce_data[['timestamp', 'frequency']] #!!! this is added
        #df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = pd.DataFrame({'frequency': ce_data})
        #df.dropna(how='any', inplace=True)
        #df = df.dropna()
        # remove data before dip in drift dip at 2017-01-25 04:00:00,
        # but some missing data until 2017-03-13 00:00:00
        
        #df = df[df['timestamp'] >= '2017-03-13 00:00:00']
        df = df[df.index >= '2017-03-13 00:00:00']

    else:
        print('Invalid country')
        return pd.DataFrame()
    #df['timestamp'] = df.index
    
    # add datetime rounded down to hour to each datapoint
    #df['hour'] = df['timestamp'].dt.floor('h', ambiguous=True) #!!! ambiguous=True is added; with this method, we lose at the moment the hours of time changing (DST) !!!

    # Group by date and hour and filter out hours that don't have enough data points # !!! replaced !!!
    # hourly_amount = df.groupby('hour').size()
    # uninterrupted_hours_index = hourly_amount[hourly_amount == s.settings[country]['values per hour']].index
    # df = df[df['hour'].isin(uninterrupted_hours_index)]

    # Alternatively:
    # df.set_index('timestamp', inplace=True)
    # df_grouped = df.groupby('hour')
    # df = df_grouped.filter(lambda x: len(x) == 3600)
    #df['timestamp'] = df.index

    sample = df.resample('h')
    df_full_hours = []
    for t,i in sample:   
        if i['frequency'].count() == s.settings[country]['values per hour']:
            df_full_hours.append(i)
    df_full_hours = pd.concat(df_full_hours, axis=0)
    df_full_hours.to_pickle(f'../data/{country}_frequency_full_hours.pkl')
    print('Frequency processing finished')
    return df_full_hours


def to_angular_freq(df: pd.DataFrame, country: str) -> pd.DataFrame:
    df['frequency'] = 2 * np.pi * (df['frequency'] - s.settings[country]['reference frequency'])

    return df
