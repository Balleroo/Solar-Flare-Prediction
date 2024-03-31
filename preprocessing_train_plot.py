import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
import time

def tic():
  return time.time()
def toc(tstart, name="Operation"):
  print('%s took: %s sec.\n' % (name,(time.time() - tstart)))

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


def filter_energy(data,filtered_eng = 0, filtered_rad_range = 99):
    '''
    The function for filter the solar event by energy_kev and radial

    '''
    try:
        assert isinstance(filtered_eng,int)
        assert isinstance(filtered_rad_range,int or float)
        assert filtered_eng>=0 and filtered_eng<= 8
        assert filtered_rad_range >= 0 and filtered_rad_range<=100

        #filter the data with energy
        lenght1 = len(data)
        data = data[data['energy_kev']!= filtered_eng]
        lenght2 = len(data)
    

        #filter the data with radial
        radial = data['radial'].values
        lenght1 = len(data)
        data = data[data['radial']<=np.percentile(radial,filtered_rad_range)]
        lenght2 = len(data)
        print('Filtering successful!')
        return data

    except(AssertionError):
        print('Filtering failed! range is wrong!')


def filter_month(data,start_month,end_month):
    '''
    This is the function for filtering the month to plot the sunspot
    '''
    data = data[data['month']<= end_month]
    data = data[data['month']>= start_month]

    return data


def ploting_predicted_sunspot(energy,x_pos,y_pos):
    '''
    The function for sunspot plotting using the filtered data
    
    '''
    ts = tic()
 
    colors = plt.cm.turbo(np.linspace(0,1,8))
    plt.style.use('dark_background')

    # build figure object
    fig, ax = plt.subplots(figsize=(10,10))
    # loop over energy ranges
    label_eng = ['3-6', '6-12','12-25','25-50','50-100','100-300','300-800','800-7000','7000-20000']

    for i,irange in enumerate(np.arange(0,9)):

        plt.scatter(x_pos[energy==i],y_pos[energy==i],color=colors[i],label='%s Kev'%label_eng[i])
        plt.legend(loc='best',fontsize=9,shadow=True)

    plt.grid( linestyle = '--', linewidth = 0.35)
 
    plt.title('SUNSPOTS per Energy in prediciton')
   
    plt.show()

    toc(ts,"Sunspot Drawing")


def solar_train(solar_data,test_size = 0.5):
    '''
    The training model
    input: solar_data: training solar data
    input: test_size: the probabilty of test-size in range of 0-1
    return: print the accuracy and each prediction
    '''
    ts = tic()
    pd_df = solar_data.copy(deep=True)

    X_train, X_test = train_test_split(pd_df, test_size=0.2)
    y_train_energy = X_train['energy_kev']
    X_train_energy = X_train.drop(['energy_kev', 'duration'], axis=1)
    y_test_energy = X_test['energy_kev']
    X_test_energy = X_test.drop(['energy_kev', 'duration'], axis=1)
 
    random_forest_classifier = RandomForestClassifier(n_jobs=-1).fit(X_train_energy, y_train_energy)
    random_forest_energy_predictions = random_forest_classifier.predict(X_test_energy)
    random_forest_energy_score = random_forest_classifier.score(X_test_energy, y_test_energy)
    
    y_train_xpos = X_train['x_pos']
    X_train_xpos = X_train.drop(['x_pos', 'y_pos'], axis=1)
    y_test_xpos = X_test['x_pos']
    X_test_xpos = X_test.drop(['x_pos', 'y_pos'], axis=1)

    random_forest_x_regressor = RandomForestRegressor(n_jobs=-1).fit(X_train_xpos, y_train_xpos)
    random_forest_x_predictions = random_forest_x_regressor.predict(X_test_xpos)
    random_forest_x_score = random_forest_x_regressor.score(X_test_xpos, y_test_xpos)

    y_train_ypos = X_train['y_pos']
    X_train_ypos = X_train.drop(['x_pos', 'y_pos'], axis=1)
    y_test_ypos = X_test['y_pos']
    X_test_ypos = X_test.drop(['x_pos', 'y_pos'], axis=1)

    random_forest_y_regressor = RandomForestRegressor(n_jobs=-1).fit(X_train_ypos, y_train_ypos)
    random_forest_y_predictions = random_forest_y_regressor.predict(X_test_ypos)
    random_forest_y_score = random_forest_y_regressor.score(X_test_ypos, y_test_ypos)
 
    print("accuarcy of energy prediction:",random_forest_energy_score)
    print("accuarcy of x pos prediction:",random_forest_x_score)
    print("accuracy of y pos prediction:",random_forest_y_score)
    toc(ts,"Training and prediction")

    return random_forest_energy_predictions,random_forest_x_predictions,random_forest_y_predictions


def ploting_sunspot(data,start_month,end_month):
    '''
    The function for sunspot plotting using the filtered data
    
    '''
    ts = tic()
    assert isinstance(start_month,int)
    assert isinstance(end_month,int)
    assert start_month>=1 and end_month<=12
    assert start_month<=end_month

    colors = plt.cm.turbo(np.linspace(-0.2,1,9))
    plt.style.use('dark_background')
    plt.rcParams.update({'font.size':12})
   
    data = filter_month(data,start_month,end_month)

    # build figure object
    fig, ax = plt.subplots(figsize=(10,10))
    # loop over energy ranges
    label_eng = ['3-6', '6-12','12-25','25-50','50-100','100-300','300-800','800-7000','7000-20000']
    month_map = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

    for i,irange in enumerate(np.arange(0,9)):
        
        AUX_data = data[data['energy_kev']==irange][['x_pos','y_pos']]
        # scatter plot to plot flare
        plt.scatter(AUX_data['x_pos'].values,AUX_data['y_pos'].values,color=colors[i],label='%s Kev'%label_eng[i])
        plt.legend(loc='lower right',fontsize=10,shadow=True)
        
    # set title to plot
    plt.grid( linestyle = '--', linewidth = 0.35)
    if end_month!=start_month:
        plt.title('Sunspots per Energy from '+month_map[start_month-1]+' to '+month_map[end_month-1]+' 2002-2018')
    else:
        plt.title('Sunspots per Energy in '+ month_map[start_month-1]+ ' 2002-2018')

    plt.xlabel('x_pos.asec')
    plt.ylabel('y_pos.asec')
    plt.xlim([-1200,1200])
    plt.ylim([-1200,1200])

    plt.show()
    
    toc(ts,"Sunspot Drawing")

if __name__ == '__main__':
    filename = 'hessi.solar.flare.UP_To_2018.csv'
    solar_data = preprocess(filename)
    # the return data gives us the processed data of solar flare 
    # trans the energy range to label
    # trans the years month and days to single columns
    filtered_data = filter_energy(solar_data,filtered_eng=0,filtered_rad_range=99)
    for i in range(0,12):
        start_month = i+1
        end_month = start_month
        ploting_sunspot(solar_data,start_month,end_month)
    
    eng_prd,x_prd,y_prd = solar_train(solar_data,test_size=1)
    ploting_predicted_sunspot(energy=eng_prd,x_pos=x_prd,y_pos=y_prd)
