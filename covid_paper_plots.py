#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Script to plot Johns Hopkins Covid-19 data, in order to
   get an understanding of what is going on in each country.

   Feel free to use and copy. Contact me if you have any issues.
   Maximilian Warner (maximilian.warner@gmail.com)
   https://github.com/maximilianwarner

   Data is taken from their Github. https://github.com/CSSEGISandData/COVID-19
"""
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model

def get_countries():
    countries = {'Austria': {'country_key': 'AT',
                             'D_01': 1,
                             'Plot_line': False,
                             'D_10': 8,
                             'marker': 'dimgray',
                             'pop': 83783942},
                 'Belgium': {'country_key': 'BE',
                             'D_01': 3,
                             'Plot_line': False,
                             'D_10': 10,
                             'marker': 'dimgray',
                             'pop': 11589623},
                 'Canada': {'country_key': 'CA',
                            'D_01': 1,
                            'Plot_line': False,
                            'D_10': 12,
                            'marker': 'dimgray',
                            'pop': 37742154},
                 'Denmark': {'country_key': 'DK',
                             'D_01': 1,
                             'Plot_line': False,
                             'D_10': 9,
                             'marker': 'dimgray',
                             'pop': 83783942},
                 'France': {'country_key': 'FR',
                            'D_01': 1,
                            'Plot_line': False,
                            'D_10': 11,
                            'marker': 'dimgray',
                            'pop': 65273511},
                 'Germany': {'country_key': 'DE',
                             'D_01': 2,
                             'Plot_line': True,
                             'D_10': 9,
                             'marker': 'dimgray',
                             'pop': 5792202},
                 'Greece': {'country_key': 'GR',
                            'D_01': 1,
                            'Plot_line': False,
                            'D_10': 13,
                            'marker': 'dimgray',
                            'pop': 10423054},
                 'Ireland': {'country_key': 'IE',
                             'D_01': 1,
                             'Plot_line': False,
                             'D_10': 9,
                             'marker': 'dimgray',
                             'pop': 4937786},
                 'Italy': {'country_key': 'IT',
                           'D_01': 1,
                           'Plot_line': True,
                           'D_10': 10,
                           'marker': 'red',
                           'pop': 60461826},
                 'Japan': {'country_key': 'JP',
                           'D_01': 1,
                           'Plot_line': True,
                           'D_10': 19,
                           'marker': 'green',
                           'pop': 126476461},
                 'Korea, South': {'country_key': 'KR',
                                  'D_01': 1,
                                  'Plot_line': True,
                                  'D_10': 10,
                                  'marker': 'black',
                                  'pop': 51269185},
                 'Netherlands': {'country_key': 'NL',
                                 'D_01': 1,
                                 'Plot_line': False,
                                 'D_10': 10,
                                 'marker': 'dimgray',
                                 'pop': 17134872},
                 'Norway': {'country_key': 'NO',
                            'D_01': 3,
                            'Plot_line': True,
                            'D_10': 10,
                            'marker': 'Purple',
                            'pop': 5421241},
                 'Portugal': {'country_key': 'PT',
                              'D_01': 1,
                              'Plot_line': False,
                              'D_10': 12,
                              'marker': 'dimgray',
                              'pop': 83783942},
                 'Spain': {'country_key': 'ES',
                           'D_01': 1,
                           'Plot_line': True,
                           'D_10': 1,
                           'marker': 'dimgray',
                           'pop': 46754778},
                 'Sweden': {'country_key': 'SE',
                            'D_01': 1,
                            'Plot_line': True,
                            'D_10': 16,
                            'marker': 'dimgray',
                            'pop': 10099265},
                 'Switzerland': {'country_key': 'CH',
                                 'D_01': 1,
                                 'Plot_line': False,
                                 'D_10': 14,
                                 'marker': 'dimgray',
                                 'pop': 8654622},
                 'United Kingdom': {'country_key': 'UK',
                                    'D_01': 1,
                                    'Plot_line': True,
                                    'D_10': 8,
                                    'marker': 'dimgray',
                                    'pop': 67886011},
                 'US': {'country_key': 'USA',
                        'D_01': 1,
                        'Plot_line': True,
                        'D_10': 21,
                        'marker': 'blue',
                        'pop': 331002651},
                 }

    return countries


def get_data():
    # import the data, direct from Johns Hopkins github
    url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/' \
          'csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv'

    df_raw = pd.read_csv(url)
    df_raw.drop(['Province/State', 'Lat', 'Long'], axis=1, inplace=True)
    df = df_raw.groupby(['Country/Region']).sum().transpose().copy()
    df.index = pd.to_datetime(df.index)

    return df


def x_y(df, country, d_num=None, pop=None):

    if d_num is None:
        x = df.index.dayofyear
    else:
        idx_0 = df.index[df[country] == d_num]
        x = (df.index.dayofyear - idx_0[0].dayofyear).to_numpy()

    # if there isn't a division by population
    if pop is None:
        y = df[country].to_numpy()
    else:
        y = (df[country]/pop).to_numpy()

    # just keep the positive time data
    y = y[x > 0]
    x = x[x > 0]

    return x, y


def line_plot_country_labels(ax, x, y, key, colour):
    # get the location of the end of each line
    x_last = x[-1]
    y_last = y[-1]

    # annotate the lines with the country names
    ax.text(x_last + 0.5, y_last, key, color=colour, alpha=1.0)


def plot_best_fit_poly(ax, df, country, d_num, lower_fit_range, plot_params):
    # get the x and y data
    x, y = x_y(df, country, d_num=d_num, pop=None)

    # set the upper fit range to be 90% of the
    upper_fit_range = plot_params['x_lim'][1]*0.9
    log_y = np.log(y)
    log_x = np.log(x)

    # extract only the data above the threshold
    log_y_cut = log_y[log_x > np.log(lower_fit_range)].copy()
    log_x_cut = log_x[log_x > np.log(lower_fit_range)].copy()

    # do a linear regression
    regr = linear_model.LinearRegression()
    regr.fit(log_x_cut.reshape(-1, 1), log_y_cut.reshape(-1, 1))

    x_fit = np.linspace(lower_fit_range, upper_fit_range,1000)
    m_log = regr.coef_[0][0]
    c_log = regr.intercept_[0]
    print('{}:{}x^{}'.format(country, np.round(np.exp(c_log), 2), np.round(m_log, 2)))
    y_log_fit = np.exp(c_log) * (x_fit ** m_log)

    ax.plot(x_fit, y_log_fit, 'k--', alpha=0.8)


def plot_best_fit_exp(ax, df, countries, d_01_type, best_fit_exp, plot_params):
    upper_fit_range = plot_params['x_lim'][1] * 0.9
    y_log_sum = np.array([], dtype=np.int64).reshape(0,)
    x_sum = np.array([], dtype=np.int64).reshape(0,)

    for country, info in countries.items():
        # extract the keys
        d_10 = info['D_10']
        d_01 = info['D_01']

        if d_01_type is True:
            d_num = d_01
        else:
            d_num = d_10

        # get the x and y data
        x, y = x_y(df, country, d_num=d_num, pop=None)
        y_log = np.log(y)

        if country in best_fit_exp.keys():
            if best_fit_exp[country] != 'ALL':
                x_upper_lim = best_fit_exp[country]
                y_log = y_log[x < x_upper_lim]
                x = x[x < x_upper_lim]
                x_sum = np.concatenate((x_sum, x))
                y_log_sum = np.concatenate((y_log_sum, y_log))

        else:
            x_sum = np.concatenate((x_sum, x))
            y_log_sum = np.concatenate((y_log_sum, y_log))

    regr = linear_model.LinearRegression()
    regr.fit(x_sum.reshape(-1, 1), y_log_sum.reshape(-1, 1))
    x_fit = np.linspace(0.1, upper_fit_range, 1000)
    m_log = regr.coef_[0][0]
    c_log = regr.intercept_[0]
    print('{}exp^{}x'.format(np.round(np.exp(c_log), 2), np.round(np.exp(m_log), 2)))
    y_log_fit = np.exp(c_log) * np.exp(x_fit * m_log)

    ax.plot(x_fit, y_log_fit, 'k--', alpha=0.8)


def remove_data_ink(ax, plot_params):
    # set the plot title
    ax.set_title(plot_params['title'], loc='left', alpha=0.7)

    # set the limits and ticks for the graph
    ax.set_xlim(plot_params['x_lim'])
    ax.set_xticks(plot_params['x_ticks'])
    ax.set_ylim(plot_params['y_lim'])
    ax.set_yticks(plot_params['y_ticks'])

    # make y grid
    ax.grid(True, 'major', 'y', ls='--', lw=.5, c='k', alpha=.3)

    # get rid of top, left and right spines, , set log scale, and make y grid
    for spine in ['left', 'right', 'top']:
        ax.spines[spine].set_visible(False)

    # scale the y axis
    ax.set_yscale('log')

    # remove ticks
    ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False)


def line_plot(ax, x, y, colour):
    # plot a line plot
    ax.plot(x, y,
            color=colour,
            alpha=0.4)


def scatter_plot(ax, x, y, colour):
    # plot a scatter plot
    ax.plot(x, y,
            marker="o",
            linestyle='None',
            markeredgecolor=colour,
            markerfacecolor=colour,
            alpha=0.7)


def plot_func(df, countries, plot_params, line_style=True, time_abs=True, d_01_type=True, pop_type=False,
              best_fit_poly=None, best_fit_exp=None):
    # set the figure
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7,5))

    # plot each country
    for country, info in countries.items():
        # extract the keys
        key = info['country_key']
        inc_line_plot = info['Plot_line']
        d_10 = info['D_10']
        d_01 = info['D_01']
        colour = info['marker']
        population = info['pop']

        if time_abs is True:
            d_num = None
        else:
            if d_01_type is True:
                d_num = d_01
            else:
                d_num = d_10

        if pop_type is False:
            pop = None
        else:
            pop = population

        # get the x and y data
        x, y = x_y(df, country, d_num=d_num, pop=pop)

        # choose what type of data to plot
        if line_style is True:
            # check if the line should be plotted
            if inc_line_plot is True:
                line_plot(ax, x, y, colour)
                # add the labels
                if time_abs is False:
                    line_plot_country_labels(ax, x, y, key, colour)
        else:
            scatter_plot(ax, x, y, colour)
            if best_fit_poly is not None and country in best_fit_poly.keys():
                lower_fit_range = best_fit_poly[country]
                plot_best_fit_poly(ax, df, country, d_num, lower_fit_range, plot_params)

    #
    if best_fit_exp is not None and line_style is False:
        plot_best_fit_exp(ax, df, countries, d_01_type, best_fit_exp, plot_params)

    # fix up the axes and cosmetics
    remove_data_ink(ax, plot_params)

    # same the image
    save_name = plot_params['save_name']
    save_time = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    file_name = 'fig_' + save_time + '_' + save_name + '.pdf'
    plt.savefig(file_name)

    # render the plot
    plt.show()


def main():
    df = get_data()
    countries = get_countries()

    # plot 1: Line plot of Number of deaths versus day in year
    plot_params_1 = {'x_lim':[50,95],
                   'y_lim':[1,30000],
                   'x_ticks':[50,65,80],
                   'y_ticks':[1,10,100, 1000, 10000],
                   'title':'Number of Deaths, versus days in year',
                   'save_name': 'Day_in_year_abs'}


    plot_func(df, countries, plot_params_1, line_style=True, time_abs=True, d_01_type=True, pop_type=False,
              best_fit_poly=None, best_fit_exp=None)

    # plot 2: Line plot of Number of deaths versus days since first death
    plot_params_2 = {'x_lim':[0,60],
                   'y_lim':[1,30000],
                   'x_ticks':[0,15,30,45],
                   'y_ticks':[1,10,100, 1000, 10000],
                   'title':'Number of Deaths, versus days since first death',
                   'save_name': 'Days_since_first'}

    plot_func(df, countries, plot_params_2, line_style=True, time_abs=False, d_01_type=True, pop_type=False,
              best_fit_poly=None, best_fit_exp=None)

    # plot 3: Number of deaths versus days since 10th death
    plot_params_3 = {'x_lim':[0,40],
                   'y_lim':[1,30000],
                   'x_ticks':[0,15,30],
                   'y_ticks':[1,10,100, 1000, 10000],
                   'title':'Number of Deaths, versus days since tenth death',
                   'save_name': 'Days_since_tenth'}

    best_fit_poly_3 = {'Korea, South':10,
                       'Italy':17}

    best_fit_exp_3 = {'Korea, South': 'ALL',
                      'Norway': 'ALL',
                      'Japan': 'ALL',
                      'Italy': 17,
                      'Spain': 20}

    plot_func(df, countries, plot_params_3, line_style=False, time_abs=False, d_01_type=False, pop_type=False,
              best_fit_poly=best_fit_poly_3, best_fit_exp=best_fit_exp_3)

    # plot 4: Number of deaths versus days since 10th death scaled to population
    plot_params_4 = {'x_lim':[0,40],
                   'y_lim':[10**-8, 3*10**-4],
                   'x_ticks':[0,15,30],
                   'y_ticks':[10**-8,10**-7,10**-6, 10**-5, 10**-4],
                   'title':'Number of Deaths per Capita, versus days since tenth death',
                   'save_name': 'Per_cap_rel_tenth'}

    plot_func(df, countries, plot_params_4, line_style=False, time_abs=False, d_01_type=False, pop_type=True,
              best_fit_poly=None, best_fit_exp=None)

    # plot 5: Line plot of Number of deaths versus days since first death
    plot_params_5 = {'x_lim':[0,40],
                   'y_lim':[1,30000],
                   'x_ticks':[0,15,30],
                   'y_ticks':[1,10,100, 1000, 10000],
                   'title':'Number of Deaths, versus days since tenth death',
                   'save_name': 'Days_since_tenth_line'}

    plot_func(df, countries, plot_params_5, line_style=True, time_abs=False, d_01_type=False, pop_type=False,
              best_fit_poly=None, best_fit_exp=None)

if __name__ == "__main__":
    main()

