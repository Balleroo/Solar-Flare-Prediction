import numpy as np
import pandas as pd
from datetime import datetime



def parse_date(sdatex, stimex):
    """Goes through our string date data and strips them to create a fluid datetime setup for each time a solar flare starts, peaks, and ends

    Args:
        sdatex (str): contains the date in a string format
        stimex (str): contains the time in a string format

    Returns:
        datetime:  creates a datetime type data that is easy to use for our uses
    """   

    datex = datetime.strptime(sdatex, '%Y-%m-%d')
    timex = datetime.strptime(stimex, '%H:%M:%S')
    return datetime(datex.year,datex.month,datex.day,timex.hour,timex.minute,timex.second)


def preprocess(fname):
    """This preprocesses our data by going through and getting rid of values and modifying them to allow for a smooth machine learning process

    Args:
        fname (str): the name of the csv file containing our solar flare information

    Returns:
        solar_df (pandas.core.frame.DataFrame): the data frame that contains solar flare data after processings
    """    
    solar_df = pd.read_csv(fname, sep=',', index_col=0)

    ''' Drop the flag columns'''
    solar_df = solar_df.drop(["active.region.ar", "flag.1", "flag.2", "flag.3", "flag.4", "flag.5"], axis=1)

    ''' Some Radial Values are beyond 2000'''
    solar_df = solar_df[solar_df["radial"] <960]

    # Adding year, month, day, start date, peak date, end date and dropping earlier columns
    solar_df['dt.start'] = solar_df[['start.date','start.time']].apply(lambda x: parse_date(x[0],x[1]), axis=1)
    solar_df['dt.peak'] = solar_df[['start.date','peak']].apply(lambda x: parse_date(x[0],x[1]), axis=1)
    solar_df['dt.end'] = solar_df[['start.date','end']].apply(lambda x: parse_date(x[0],x[1]), axis=1)

    solar_df.drop(['start.date','start.time','peak','end'], axis=1, inplace=True)

    # add new columns
    solar_df['year'] = solar_df['dt.start'].apply(lambda col: col.year)
    solar_df['month'] = solar_df['dt.start'].apply(lambda col: col.month)
    solar_df['day'] = solar_df['dt.start'].apply(lambda col: col.day)


    solar_df = solar_df.rename(columns={'duration.s': 'duration', 'peak.c/s': 'peak_c_s', 'total.counts': 'total_counts', 
                                    'energy.kev': 'energy_kev', 'x.pos.asec': 'x_pos', 'y.pos.asec': 'y_pos', 
                                    'dt.start': 'date_start', 'dt.peak':'date_peak', 'dt.end': 'date_end'})



    # Enumerating energy range values from str to category
    category = {'3-6': 0, '6-12': 1, '12-25': 2, '25-50': 3, '50-100': 4, '100-300': 5, '300-800': 6, '800-7000': 7, '7000-20000': 8}
    solar_df['energy_kev'] = solar_df['energy_kev'].map(category)

    solar_df['duration'] = np.log1p(solar_df['duration'])
    solar_df = solar_df.drop(['date_start', 'date_peak', 'date_end'], axis=1)

    return solar_df

def preprocess_plots(fname):
    '''
    Process the data for the plots
    :param fname: The name of the csv file containing our solar flare information
    :return: solar_df (pandas.core.frame.DataFrame): The data frame that contains solar flare data after processing
    '''
    assert isinstance(fname,str)
    solar_df = pd.read_csv(fname, parse_dates=["start.date", "start.time", "peak", "end"],
                           dayfirst=True, infer_datetime_format=True)
    solar_df = solar_df.drop(["flare", "flag.1", "flag.2", "flag.3", "flag.4", "flag.5", "active.region.ar"], axis=1)
    solar_df = solar_df[solar_df["radial"] < 960]
    energy_bands = ["3-6", "6-12", "12-25", "25-50", "50-100", "100-300", "300-800", "800-7000", "7000-20000"]
    solar_df['energy.kev'] = pd.Categorical(solar_df['energy.kev'], categories=energy_bands, ordered=True)
    solar_df["start.time"] = solar_df["start.time"].dt.time
    solar_df["peak"] = solar_df["peak"].dt.time
    solar_df["end"] = solar_df["end"].dt.time
    solar_df["year"] = solar_df["start.date"].dt.year
    solar_df["month"] = solar_df["start.date"].dt.month
    solar_df['day'] = solar_df["start.date"].dt.day
    return solar_df


