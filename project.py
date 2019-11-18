    # -*- coding: utf-8 -*-

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
import math


def drug_user(row):
    row = row.drop(['Alcohol', 'Nicotine','Caff'])
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
    

def create_usage_plot(data, title, filename):

    drug_users = []
    
    for drug in data.loc[:,'Alcohol':]:
      # check for non-zero 0 values
      users = np.count_nonzero(data[drug].values > 0)
      drug_users.append((drug, users))
    
    # sort number of non zero values
    drug_users  = sorted(drug_users, key=lambda x: x[1], reverse=True)
    
    # separate users and drugs from sorted collection
    drugs = [drug for (drug, _) in drug_users]
    
    bar_colors = ['C0','C1','C2','C3','C4','C5','C6']
    y_pos = np.arange(len(data.loc[:,'Alcohol':].columns))
    
    pos = 0
    for drug in drugs:
        value_counts = data[drug].value_counts()
        if 0 in value_counts.index:
            value_counts.drop([0], inplace=True)
        # convert usage type to corresponding color
        colors = [bar_colors[i] for i in value_counts.index][::-1]
        # get number of usage for drug
        drug_users = value_counts.values[::-1]
        # shift drug usage for overlapping bar graphs
        for  i in range(1,len(drug_users)):
            drug_users[i] += drug_users[i -1]
        i = 0
        for (users, color) in zip(drug_users, colors):
            plt.barh(pos, users, color=color, zorder=-i, align='center')
            i += 1
        pos += 1
        
    usage_texts = ['Used over a Decade Ago', 
             'Used in Last Decade', 'Used in Last Year', 
             'Used in Last Month', 'Used in Last Week', 
             'Used in Last Day']
    
    legend_patches = []
    for (usage_text, color) in zip(usage_texts, bar_colors[1:]):
        patch = mpatches.Patch(color=color, label=usage_text)
        legend_patches.append(patch)
      
    plt.legend(handles=legend_patches)
    plt.title(title)
    plt.xlabel('Users')
    plt.yticks(y_pos, drugs)
    fig = plt.gcf()
    fig.set_size_inches(20,12)
    plt.savefig(filename)
    plt.clf()
    
    
if __name__ == '__main__':

    data = pd.read_csv('drug_consumption.csv')
    
    print(data.info())
    
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
    
    pd.set_option('precision', 5)
    
    min_score = data.loc[:,'Nscore':'Cscore'].min().values.min()
    data.loc[:,'Nscore':'Cscore'] += abs(min_score)
    
    data['Drug User'] = data.apply(drug_user, axis=1)
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
    print(nonusers_xpos)
    plt.bar(users_xpos, users_scores, width=width, color=['C1'], label='Users')
    plt.bar(nonusers_xpos, nonusers_scores, width=width, color=['C2'], label='Non-users')
    plt.title('Personality Trait Scores for Illegal Drug Users')
    plt.xlabel('Personality Traits')
    plt.ylabel('Mean Scores')
    plt.xticks(users_xpos, traits)
    plt.legend(loc='upper center', bbox_to_anchor=(1.2, 0.9), shadow=True, ncol=1)
    fig = plt.gcf()
    fig.set_size_inches(10,8)
    plt.show()
    plt.clf()
    
    xpos = np.arange(2)
    bars = [len(male_drug.index), len(female_drug.index)]
    
    plt.bar(xpos, bars, width=0.3, color=['C1','C2'], align='center')
    plt.xticks(xpos, ['Male', 'Female'])
    plt.title('Drug Usage by Gender')
    plt.ylabel('Users')
    fig = plt.gcf()
    fig.set_size_inches(8,6)
    plt.savefig('Figure1.png')
    plt.clf()
    
    
    create_usage_plot(male_drug.drop(['Drug User'], axis=1), 'Drug Usage for Males', 'Figure2.png')
    create_usage_plot(female_drug.drop(['Drug User'], axis=1), 'Drug Usage for Females', 'Figure3.png')
    create_usage_plot(data.drop(['Drug User'], axis=1), 'Drug Usage', 'Figure4.png')







