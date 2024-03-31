import numpy as np
import pandas as pd
from preprocessing import preprocess_plots
from matplotlib import pyplot as plt
import seaborn as sns



def plots(solar_df):
    '''
    Takes solar data frame and generates the plots for the presentation
    :input: solar_df (DataFrame): our main dataframe before any cleaning
    '''
    assert isinstance(solar_df, pd.DataFrame)

    month_order = ["", "January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
                   "November", "December"]
    duration = solar_df['duration.s']
    radial = solar_df['radial']

    # Adjust plot styles
    plt.style.use("dark_background")
    plt.rcParams.update({'font.size': 12})
    color = (32 / 255, 100 / 255, 170 / 255, 255 / 255)

    # Mask to cover the symmetric portion of the heat map
    corr = solar_df.corr()
    mask = np.triu(np.ones_like(corr)) - np.eye(corr.shape[0])

    f, ax = plt.subplots(figsize=(11, 9))
    cmap = sns.color_palette("RdBu", as_cmap=True)
    sns.heatmap(corr, mask=mask, cmap=cmap, center=0,
                square=True, linewidths=0, cbar_kws={"shrink": .5})
    plt.title("Data Correlation Heat Map")
    plt.xticks(rotation=45)
    plt.show()

    # Total Number of Flares grouped by duration
    df = solar_df.loc[:, ['duration.s', 'total.counts']]
    df_2 = df.groupby(pd.cut(df["duration.s"], range(0, max(duration), 500))).sum()
    ax = df_2["total.counts"].plot(kind='bar', title="Total Number of Flares Grouped by Duration", color=color)
    ax.set_xlabel("Duration (seconds)")
    ax.set_ylabel("Total Number of Flares")
    plt.xticks(rotation=0)
    plt.show()

    # Frequency of flare events given in a particular duration
    df_2 = df.groupby(pd.cut(df["duration.s"], range(0, max(duration), 500))).count()
    ax = df_2["total.counts"].plot(kind='bar', title="Frequency of Flare Events Grouped by Duration", color=color)
    ax.set_xlabel("Duration (seconds)")
    ax.set_ylabel("Total Number of Flares")
    plt.xticks(rotation=0)
    plt.show()

    # Mean Number of Flares in the given duration
    df_2 = df.groupby(pd.cut(df["duration.s"], range(0, max(duration), 500))).mean()
    ax = df_2["total.counts"].plot(kind='bar', title="Mean Number of Flares per Event Grouped by Duration", color=color)
    ax.set_xlabel("Duration (seconds)")
    ax.set_ylabel("Mean Number of Flares")
    plt.xticks(rotation=0)
    plt.show()

    # Number of flare events per year
    # Correlated with the 11 year cycle of flares
    df = solar_df.loc[:, ['year', 'total.counts']]
    df = df.groupby(df["year"]).count()
    ax = df["total.counts"].plot(title="Number of Flare Events per Year", color=color)
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of Flare Events")
    plt.show()

    # Number of flare event across all years per month
    df = solar_df.loc[:, ['month', 'total.counts']]

    df = df["total.counts"].groupby(df["month"]).count()
    ax = df.plot(title="Number of Flare Events per Month Across All Years", color=color)
    ax.set_xlabel("Month")
    ax.set_ylabel("Number of Flares Events")
    plt.xticks(range(13), month_order)
    plt.show()

    # Energy Band vs Radial Distance
    df = solar_df.loc[:, ['energy.kev', 'radial']]
    df = df[df['energy.kev'] != "800-7000"]
    df = df[df['energy.kev'] != "7000-20000"]
    df = df.groupby(['energy.kev'])['radial'].mean()
    ax = df.plot(title="Radial Distance vs Energy Band", color=color)
    ax.set_xlabel("Energy Band (kev)")
    ax.set_ylabel("Radial Distance (arcseconds)")
    plt.show()


    # Energy Band vs Duration
    # Showing that the energy band increases with radial distance
    df = solar_df.loc[:, ['energy.kev', 'duration.s']]
    df = df[df['energy.kev'] != "800-7000"]
    df = df[df['energy.kev'] != "7000-20000"]
    df = df.groupby(['energy.kev'])['duration.s'].mean()
    ax = df.plot(title="Duration vs Energy Band", color=color)
    ax.set_xlabel("Energy Band (kev)")
    ax.set_ylabel("Duration (in seconds)")
    plt.show()

    # Radial Distance vs Year
    df = solar_df.loc[:, ['year', 'radial']]
    df = df.groupby(['year'])['radial'].mean()
    ax = df.plot(title="Radial Distance vs Year", color=color)
    ax.set_xlabel("Year")
    ax.set_ylabel("Radial Distance (arcseconds)")
    plt.show()

if __name__ == '__main__':
    solar_df = preprocess_plots("hessi.solar.flare.UP_To_2018.csv")
    plots(solar_df)
