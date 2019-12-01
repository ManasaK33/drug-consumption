    # -*- coding: utf-8 -*-

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
import matplotlib as mpl
import math


def is_drug_user(row):
    #row = row.drop(['Alcohol', 'Nicotine','Caff'])
    row = row['Amphet':]
    num_zeros = (row == 0).astype(bool).sum()
    if num_zeros == row.size:
        return False
    return True

def gender_map(x):
    if math.isclose(x, 0.48246):
        return 'Female'
    else:
        return 'Male'
    
def country_map(x):
    countries = [(-0.09765, 'Australia'), (0.24923,'Canada'), 
               (-0.46841,'New Zealand'), (0.21128,'Republic of Ireland'), 
               (0.96082,'UK'), (-0.57009,'USA'), (-0.28519,'Other')]
    for (value, country) in countries:
        if math.isclose(x, value):
            return country;
    return None
    


def create_plot(ax, data, color='#99004f', yticks_on=True):
    usage = data.apply(pd.Series.value_counts, axis=0)
    drug = usage.iloc[:,0].name
    usages = usage.sort_index().iloc[:,0].values
    width = 0.05
    xpos = np.arange(len(usage.index),dtype='float64')
    xpos *= 0.10
    ax.bar(xpos, usages, width=width, color=color)
    rects = ax.patches
    for rect, usage in zip(rects,usages):
        x = rect.get_x() + rect.get_width()/2
        y = rect.get_height() + 0.5
        ax.text(x, y, usage, ha='center',va='bottom', fontsize=5)
    
    usage_texts = ['Never Used','Decade Ago','Last Decade', 
                   'Last Year', 'Last Month', 
                   'Last Week', 'Last Day']
    ax.set_ylim(bottom=0, top=1800)
    ax.set_xticks(xpos)
    ax.set_xticklabels(usage_texts)
    ax.tick_params(axis='y', labelsize=4)
    ax.tick_params(axis ='x', labelrotation=15, labelsize=4, width=0.7)
    ax.text(0.5, 0.9, drug, horizontalalignment='center', 
            transform=ax.transAxes, fontsize=9)




def create_usage_subplots(data):
    
    colors = ['#99004f', '#007acc', '#009900', '#e67300',
              '#cc0000','#0000b3', '#7a00cc', '#e6e600',
              '#2eb8b8']
    
    data = data.rename(columns={'Amphet':'Amphetamine', 'Amyl':'Amyl Nitrite',
            'Benzos':'Benzodiazepine','Caff':'Caffeine','Coke':'Cocaine',
            'Meth':'Methamphetamine'})
    
    
    drug_data = data.loc[:,'Alcohol':].apply(lambda series: series.astype(bool).sum(),axis=0)
    drug_data = drug_data.sort_values(ascending=False)
    
    drugs = drug_data.index.values

    nrows = 5
    ncols= 4
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, sharex=False, sharey=True, figsize=(10,8))
    plt.subplots_adjust(wspace=0.02, hspace= 0.3, top = 0.95)
    color_idx = 0
    i = 0
    for row in range(nrows):
        for col in range(ncols):
            color_idx = (color_idx+1)%len(colors)
            if col != 0:
                axs[row, col].tick_params(axis='y', width=0)
            if i >= len(drugs):
                axs[row, col].set_visible(False)
            else:
                create_plot(axs[row, col], data[[drugs[i]]], colors[color_idx])
            i += 1
    
    plt.savefig('figures/drugs.png', dpi=300)
    plt.clf()


    
    
if __name__ == '__main__':
    
    data = pd.read_csv('drug_consumption.csv')
    mpl.rcParams['axes.linewidth'] = 0.5
    
    # consider numpy.inf as n/a value
    pd.options.mode.use_inf_as_na = True
    
    # label encode categorical variables
    for column in data.loc[:,'Alcohol':'VSA']:
        # get label encoding for column
        data[column] = data[column].astype('category').cat.codes
        # convert column to numeric type
        data[column] = data[column].astype('int32')
    
    # drop fake drug
    del data['Semer'] 
    # drop ID variable
    del data['ID']
    # drop chocolate
    del data['Choc']
    
    create_usage_subplots(data)
    
    min_score = data.loc[:,'Nscore':'Cscore'].min().values.min()
    data.loc[:,'Nscore':'Cscore'] += abs(min_score)
    
    data['Drug User'] = data.drop(['Alcohol', 'Nicotine','Caff'], axis=1).apply(is_drug_user, axis=1)
    data['Gender']= data['Gender'].apply(gender_map)
    data['Country']=data['Country'].apply(country_map)
    
    # Create dataset for drug users
    users = data.loc[data['Drug User'] == True]
    # Create dataset for non-drug users
    nonusers = data.loc[data['Drug User'] == False]
    
    # Get data for male drug users
    male_drug = users.loc[(users['Gender'] == 'Male')]
    male_drug = male_drug.reset_index(drop=True)
    # Get data for female drug users
    female_drug = users.loc[(users['Gender'] == 'Female')]
    female_drug = female_drug.reset_index(drop=True)
    
    
    traits = ['Neuroticism', 'Extraversion', 'Openness', 
              'Agreeablenes', 'Conscientiousnes']
    users_scores = users.loc[:,'Nscore':'Cscore'].mean().values
    nonusers_scores = nonusers.loc[:,'Nscore':'Cscore'].mean().values
    
    width = 0.3
    users_xpos = np.arange(len(users_scores))
    nonusers_xpos = [x + width for x in users_xpos]
    
    plt.bar(users_xpos, users_scores, width=width, color=['#cc7a00'], label='Users')
    plt.bar(nonusers_xpos, nonusers_scores, width=width, color=['#009900'], label='Non-users')
    ax = plt.gca()
    rects = ax.patches
    for rect, score in zip(rects[:5],users_scores):
        x = rect.get_x() + rect.get_width()/2
        y = rect.get_height() + 0.1
        ax.text(x, y, np.round(score, 2), ha='center',va='bottom', fontsize=10)
    for rect, score in zip(rects[5:],nonusers_scores):
        x = rect.get_x() + rect.get_width()/2
        y = rect.get_height() + 0.1
        ax.text(x, y, np.round(score, 2), ha='center',va='bottom', fontsize=10)
    plt.title('Personality Trait Scores for Illegal Drug Users')
    plt.xlabel('Personality Traits')
    plt.ylabel('Mean Scores')
    plt.xticks(users_xpos, traits)
    plt.legend(loc='upper center', bbox_to_anchor=(1.2, 0.9), shadow=True, ncol=1)
    fig = plt.gcf()
    fig.set_size_inches(10,8)
    plt.savefig('figures/traits.png', dpi=300)
    plt.clf()
    
    xpos = np.arange(2)
    bars = [len(male_drug.index), len(female_drug.index)]
    
    plt.bar(xpos, bars, width=0.3, color=['C1','C2'], align='center')
    plt.xticks(xpos, ['Male', 'Female'])
    plt.title('Illegal Drug Usage by Gender')
    plt.ylabel('Users')
    fig = plt.gcf()
    fig.set_size_inches(8,6)
    plt.savefig('figures/genders.png', dpi=300)
    plt.clf()
    







