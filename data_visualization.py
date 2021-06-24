"""
Sneh Gupta
CSE 163: Brazil Forest Fire Project
Data visualization of forest fires in Brazil
Plotted the total fires in each state over two decades
and made a video of a time lapse of forest fires per month over two decades,
by state
fires refers to forest fire dataset, data refers to geospatial dataset,
and merged is the combination of the two
"""

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl
from matplotlib.artist import Artist
from unidecode import unidecode
import calendar


def process(url):
    """
    Takes url for forest fire dataset
    Reads in dataset, normalizes state names, converts date to datetime
    format and returns processed dataset
    """
    fires = pd.read_csv(url)
    fires = fires.rename(columns={'fires': 'number'})
    fires['state'] = fires['state'].apply(unidecode)
    fires['date'] = pd.to_datetime(fires['date'], format='%Y-%m-%d')
    return fires


def merge_new(fires, states):
    """
    Takes fires data and geometry data
    Normalizes data so that they mathc and merges the two datasets
    Returns the merged data and normalized geometry data
    """
    states['nome'] = states['nome'].apply(unidecode)
    states['nome'] = states['nome'].str.title()
    states = states[['nome', 'geometry']]
    states = states.dissolve(by='nome', aggfunc='sum')
    fires['state'] = fires['state'].str.title()
    merged = fires.merge(states, left_on='state', right_on='nome', how='inner')
    merged = gpd.GeoDataFrame(merged, geometry='geometry')
    return (merged, states)


def graph_tot(merged, states):
    """
    take merged data set and geometry data
    plot Brazil with visualization of the total number of forest fires
    in each state from 1998 to 2017
    """
    fig, ax = plt.subplots(1, figsize=(15, 7))
    states.plot(color='#EEEEEE', ax=ax)
    merged = merged[['state', 'geometry', 'number']]
    merged = merged.dissolve(by='state', aggfunc='sum')
    merged.plot(column='number', legend=True, ax=ax)
    plt.title('Number of Total Forest Fires in Brazil from 1998 to 2017')
    plt.savefig('tot_fires.png')


def time_lapse(merged, states):
    """
    take merged data and geometry data
    makes a video containing time lapse of the brazil forest fires at each
    month from 1998 to 2017
    """
    # replace counts for months with name of month
    merged['month'] = merged['month'].apply(lambda x: calendar.month_name[x])
    fig, ax = plt.subplots(1, figsize=(20, 10))
    states.plot(color='#EEEEEE', ax=ax)
    # sort and make legend
    merged = merged.sort_values(by='date')
    norm = mpl.colors.Normalize(vmin=merged['number'].min(),
                                vmax=merged['number'].max())
    cmap = cm.viridis
    fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap))
    ax.location = 'right'
    ax.set_title('Number of forest fires per month', fontsize=16)
    count = 0
    # loop over each month and create and image for forest fire visualization
    # for that month
    for month in merged['date'].unique():
        count += 1
        merged_month = merged[merged['date'] == month]
        # colors.Normalize(merged['number'].min(), merged['number'].max())
        merged_month.plot(column='number', ax=ax, vmin=merged['number'].min(),
                          vmax=merged['number'].max())
        text = ax.text(x=-45, y=3, s=str(merged_month['year'].unique()[0]) +
                       ' ' + str(merged_month['month'].unique()[0]),
                       fontsize=20)
        fig.savefig(f'D:/Textbooks/cse163/Project/frames/frame_{count:03d}',
                    bbox_inches='tight')
        Artist.remove(text)
        plt.close()
    # putting together the images in a video
    # ffmpeg -framerate 5 -i D:/Textbooks/cse163/Project/frames/frame_%03d.png\
    # output.mp4
    # D:/Textbooks/cse163/Project/ffmpeg/bin/ffmpeg.exe -framerate 5 -i\
    # D:/Textbooks/cse163/Project/frames/frame_%03d.png output.mp4
