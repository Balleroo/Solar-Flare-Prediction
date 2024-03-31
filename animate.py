import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
import datetime
from dateutil.relativedelta import relativedelta



def generate_time_lapse():
    def date_gen():
        current_date = datetime.datetime(2002, 2, 1)
        #super slow so change 2018 to... something smaller, idk, 2014?
        while (current_date.year != 2018):
        #for i in range(0,3):
            #time_change = datetime.timedelta(hours=24)
            time_change = relativedelta(months=1)
            current_date = current_date + time_change
            #print('gen ran, today is  \n' +str(current_date))
            yield current_date
        print('gen is done')
    
    def update(current_date):
        # Get a date index to track what points to plot next
        
        indices=solar_df.index[(solar_df['start.year']==current_date.year) & (solar_df['start.month']==current_date.month)].tolist()    #print('found indiced \n' + str(indices))
        solar_df.loc[indices,'alpha'] = 1.0
        #solar_df['alpha'] = solar_df['alpha'].apply(lambda x: x-0.1 if x>0 else 0)
        #solar_df.assign(solar_df.alpha=(solar_df.alpha - 0.1).where(solar_df.alpha!=0, 0)
        title.set_text("Current Day:"+str((current_date.strftime("%Y, %m"))))
        scat.set_alpha(solar_df['alpha'])
        #scat.set_label('')
        # Show points of the desired time (set their alpha values to 1).
        #solar_df.loc[solar_df['alpha'] > 0, 'alpha'] = solar_df['alpha'] - 0.1
        #lazy way to make fade out any points
        #Make all points more transparent (0 transparent points are not affected).
        solar_df.loc[solar_df['alpha'] == 0.3, 'alpha'] = 0
        solar_df.loc[solar_df['alpha'] == 0.6, 'alpha'] = 0.3
        #solar_df.loc[solar_df['alpha'] == 0.6, 'alpha'] = 0.4
        #solar_df.loc[solar_df['alpha'] == 0.8, 'alpha'] = 0.6
        solar_df.loc[solar_df['alpha'] == 1.0, 'alpha'] = 0.6
        
        #adding legend? Automatic doesn't seem to work
        #ax.legend(solar_df['energy.kev'], solar_df['energy.kev'].unique().tolist(),loc=1)
        #ax.legend([scat])
        
        
    
    solar_df = pd.read_csv("hessi.solar.flare.UP_To_2018.csv", parse_dates=["start.date", "start.time", "peak", "end"],
                           dayfirst=True, infer_datetime_format=True)
    #solar_df = pd.read_csv("hessi.solar.flare.UP_To_2018.csv")
    #duration_vs_counts(solar_df)
    #solar_timelapse(solar_df)
    #b = solar_df['energy.kev']
    #split_range = lambda b : [[int(y) for y in x.split('-')] if len(x.split('-')) == 2 else [int(x.split('+')[0])] for x in b]
    #energy_mean = pd.Series([sum(i)/len(i) for i in split_range(b)])
    #solar_df['energy.kev'] = energy_mean
    '''
    Takes x and y arc seconds, energy band, time, and duration, and makes a time
    lapse plot of the solar flares. Script remains in it's own file do to complexity. 
    Integrating it with the other plots as a function would be more difficult
    due to the nature of animation (need a lot of global variables).'
    '''
    '''Plotting all the Flares from the filtered dataframe'''
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_axes([0, 0, 1, 1], frameon=False)
    
    
        
    'Work to add legend'
    #hlegend = ax.legend(loc='upper right')
    #legend1 = ax.legend(*scat.legend_elements(),
    #                    loc="lower left", title="Classes")
    #ax.add_artist(legend1)
    #ax = fig.add_axes()
    #ax.legend()
    #colors = plt.cm.jet(np.linspace(0,1,len(solar_df['energy.kev'].values)))
    '''Take off any outlier in the x and y position. Leave rest of the columns unchanged.'''
    #solar_df=solar_df[solar_df["x.pos.asec"].between(-1000, 1000, inclusive=False) & solar_df["y.pos.asec"].between(-1000, 1000, inclusive=False)]
    solar_df = solar_df[solar_df["radial"] <960]
    n_points= len(solar_df)
    '''Add alpha for transparency during animation, fades out old flares.'''
    solar_df['alpha'] = float(0)
    solar_df['start.year'] = pd.DatetimeIndex(solar_df['start.date']).year
    solar_df['start.month'] = pd.DatetimeIndex(solar_df['start.date']).month
    '''Big TO DO: Add color column (based on ev bands) and size column (based on duration.
    These columns will be fed into the scatter and displayed. Only thing changing
    during the animation will be the transparency.'''
    title = ax.text(0.5,0.9, "", bbox={'facecolor':'w', 'alpha':0.5, 'pad':5},
                    transform=ax.transAxes, ha="center")
    
    'work to add specific colormap, really unoptimized I know'
    colormap = plt.cm.turbo(np.linspace(0,1,9))
    #colormap = np.array(['r','g','b', 'salmon', 'peru', 'orange', 'lime', 'green', 'navy'])
    #solar_df['colormap'] = list(0,0,0,0)
    solar_df=solar_df.reset_index(drop=True)
    color_df = pd.DataFrame(np.zeros((len(solar_df),4)))
    #df.apply(lambda r: tuple(r), axis=1).apply(np.array)
    catagories = solar_df['energy.kev'].unique().tolist()
    #atest = solar_df[solar_df['energy.kev']=='3-6'].index
    color_df.iloc[solar_df[solar_df['energy.kev']=='3-6'].index.values] = colormap[0]
    color_df.iloc[solar_df[solar_df['energy.kev']=='6-12'].index.values] = colormap[1]
    color_df.iloc[solar_df[solar_df['energy.kev']=='12-25'].index.values] = colormap[2]
    color_df.iloc[solar_df[solar_df['energy.kev']=='25-50'].index.values] = colormap[3]
    color_df.iloc[solar_df[solar_df['energy.kev']=='50-100'].index.values] = colormap[4]
    color_df.iloc[solar_df[solar_df['energy.kev']=='100-300'].index.values] = colormap[5]
    color_df.iloc[solar_df[solar_df['energy.kev']=='300-800'].index.values] = colormap[6]
    color_df.iloc[solar_df[solar_df['energy.kev']=='800-7000'].index.values] = colormap[7]
    color_df.iloc[solar_df[solar_df['energy.kev']=='7000-20000'].index.values] = colormap[8]
    
    
    
    scat = ax.scatter(solar_df["x.pos.asec"],solar_df["y.pos.asec"], alpha = solar_df['alpha'], s = solar_df['duration.s']/5, c=color_df)
    #plt.legend(solar_df['energy.kev'].unique().tolist())
    ''' Work to add circle grid to the animation'''
    circle = plt.Circle((0,0), 1000, color='r', fill=False)
    #ax.add_patch(circle)
    ax.grid(linestyle='--',linewidth=0.35)
    #scat.xlabel('x_pos.asec')
    #scat.ylabel('y_pos.asec')
    #scat.xlim([-1200,1200])
    #scat.ylim([-1200,1200])
   
    #plt.style.use(['dark_background'])
    #solar_df['alpha'] = float(0)
    #scat.set_alpha(solar_df['alpha'])
    
    
    
    #solar_df.loc[solar_df['energy.kev'] == '3-6', 'energy.kev'] = colormap[0]
    #solar_df.loc[solar_df['energy.kev'] == '6-12', 'energy.kev'] = colormap[1]
    #solar_df.loc[solar_df['energy.kev'] == '12-25', 'energy.kev'] = colormap[2]
    #solar_df.loc[solar_df['energy.kev'] == '25-50', 'energy.kev'] = colormap[3]
    #solar_df.loc[solar_df['energy.kev'] == '50-100', 'energy.kev'] = colormap[4]
    #solar_df.loc[solar_df['energy.kev'] == '100-300', 'energy.kev'] = colormap[5]
    #solar_df.loc[solar_df['energy.kev'] == '300-800', 'energy.kev'] = colormap[6]
    #solar_df.loc[solar_df['energy.kev'] == '800-7000', 'energy.kev'] = colormap[7]
    #solar_df.loc[solar_df['energy.kev'] == '7000-20000', 'energy.kev'] = colormap[8]
    # for j in range(0, len(catagories)):
    #     for i in range(0, len(solar_df)):
    #         if solar_df['energy.kev'][i] == catagories[j]:
    #             solar_df['energy.kev'][i] = j
    
    'Actual animation process'
    ani = animation.FuncAnimation(fig, update, date_gen, interval=120, repeat=False, blit=False)
    writergif = animation.PillowWriter(fps=1)
    #ani.save('filename2.gif',writer=writergif)
    plt.show()
    'Adding dark background'
    ani.save('filename2.gif',writer=writergif, savefig_kwargs={'facecolor':'black'})
    

