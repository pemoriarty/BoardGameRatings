#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 15:45:43 2018

@author: pamela
@purpose: visualize data to work on feature selection
"""
df_info2.columns
#Index(['id', 'name', 'year', 'minplayers', 'maxplayers', 'mintime', 'maxtime',
#       'playtime', 'age', 'description', 'users_rated', 'average_rating',
#       'bayes_rating', 'sd_rating', 'complexity', 'num_comp', 'categories',
#       'subdomains', 'mechanics', 'publisher', 'all_pub'],
      dtype='object', name='attribute')

plt.hist(df_info2['minplayers'])
plt.scatter(df_info2['minplayers'], df_info2['complexity'])

fig = plt.figure(figsize=(12,9))

signal_axes = fig.add_subplot(211)
signal_axes.plot(xs,rawsignal)

fft_axes = fig.add_subplot(212)
fft_axes.set_title("FFT")
fft_axes.set_autoscaley_on(False)
fft_axes.set_ylim([0,1000])
fft = scipy.fft(rawsignal)
fft_axes.plot(abs(fft))

plt.show()

plt.figure()
plt.scatter(df_info2['playtime'], df_info2['complexity'])
plt.xlim(0,200)

player_range = df_info2['maxplayers'] - df_info2['minplayers']
plt.scatter(player_range, df_info2['complexity'])

