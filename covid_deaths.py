#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Script to plot Johns Hopkins Covid-19 data, in order to
   get an understanding of what is going on in each country.

   Feel free to use and copy. Contact me if you have any issues.
   Maximilian Warner (maximilian.warner@gmail.com)
   https://github.com/maximilianwarner

   Data is taken from their Github. https://github.com/CSSEGISandData/COVID-19
"""

import pandas as pd
import matplotlib.pyplot as plt

# import the data, direct from Johns Hopkins github
url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/' \
      'csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Deaths.csv'

df_raw = pd.read_csv(url)
df_raw.drop(['Province/State', 'Lat', 'Long'], axis=1, inplace=True)
df = df_raw.groupby(['Country/Region']).sum().transpose().copy()
df.index = pd.to_datetime(df.index)

# countries of interest
countries = {'Italy':{'country_key':'Italy', 'first_death':1},
             'Spain':{'country_key':'Spain', 'first_death':1},
             'United Kingdom':{'country_key':'U.K.', 'first_death':1},
             'Korea, South':{'country_key':'S. Korea', 'first_death':1},
             'Germany':{'country_key':'Germany', 'first_death':2}}

# set the figure
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12,5))

# plot each country
for country, info in countries.items():
    # extract the key and first death number
    key = info['country_key']
    first = info['first_death']

    # get the location of the end of each line
    x_last = df.index[-1].dayofyear
    y_last = df.iloc[-1][country]

    # get the day of the first death
    idx_first = df.index[df[country] == first]

    # set colours
    if key == 'U.K.':
        line_colour = 'red'
        line_zorder = 2
    else:
        line_colour = 'dimgray'
        line_zorder = 1

    # plot the number of deaths versus the days since first death
    axes[0].plot(df.index.dayofyear - idx_first[0].dayofyear, df[country],
               color=line_colour, zorder=line_zorder, alpha=0.4)

    # plot the number of deaths versus the days in the year
    axes[1].plot(df.index.dayofyear, df[country],
               color=line_colour, zorder=line_zorder, alpha=0.4)

    # annotate the lines with the country names
    axes[0].text(x_last - idx_first[0].dayofyear + 0.5, y_last, key, color=line_colour, alpha=1.0)

    # set an offset for countries with similar end values
    if key == 'Germany':
        offset = -10
    else:
        offset = 0

    axes[1].text(x_last + 0.5, y_last + offset, key, color=line_colour, alpha=1.0)

# set the title and scales
axes[0].set_title('Number of Deaths, versus days since first death', loc='left', alpha=0.7)
axes[0].set_xlim([0,40])
axes[0].set_xticks([0,15,30])
axes[0].set_ylim([1,10000])
axes[0].set_yticks([1,10,100, 1000, 10000])

axes[1].set_title('Number of Deaths, versus day of year', loc='left', alpha=0.7)
axes[1].set_xlim([50,95])
axes[1].set_xticks([50,65,80])
axes[1].set_ylim([1,10000])
axes[1].set_yticks([1,10,100, 1000, 10000])

# get rid of top and right spines, remove ticks, and set log scale
for ax in axes:
    for spine in ['left', 'right', 'bottom', 'top']:
        ax.spines[spine].set_visible(False)
    ax.set_yscale('log')
    ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False)

# render the plot
plt.show()


