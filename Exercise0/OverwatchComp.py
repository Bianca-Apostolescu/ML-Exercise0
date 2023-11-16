#!pip install openml

# Basic libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Working with files
import os,re, glob, io


# Working with OpenML dataset
import openml
from openml.datasets import edit_dataset, fork_dataset, get_dataset

# Download dataset based on OpenML dataset ID

# Name = Overwatch-competitions-data-7-seasons
# ID = 43577
# URL = https://www.openml.org/search?type=data&sort=runs&status=active&id=43577

dataset = openml.datasets.get_dataset(43577)

# Print a summary
print(
    f"This is dataset '{dataset.name}', the target feature is "
    f"'{dataset.default_target_attribute}'"
)

print(dataset.description[:500])
print("In this report, we conduct an exploratory data analysis on a dataset containing details about Overwatch games.")

# Get the actual data
df, *_ = dataset.get_data()
print(f"The dataset of '{dataset.name}' has {len(df)} simples and {len(df.columns)} features")

#check for missing values
missing_values = df.isnull().sum()
missing_values[missing_values > 0]

# Basic statistics of numerical columns
df.describe()

# rename columns
df = df.rename(columns={"charcter_1": "character_1"})

# create DateTime variable
df['date'] = pd.to_datetime(df['date'],format='%m/%d/%Y')

# convert columns match_length and time_left to numeric
def convert_match_length(x, df):

    # Get the unique values
    unique_values = df.unique()
    # Use a regular expression to match only float and integer values
    pattern = re.compile(r'^\d+(\.\d+)?$')
    filtered_values = [value for value in unique_values if value is not None and (pattern.match(str(value)) or (':' in value and all(part.isdigit() for part in value.split(':'))))]

    if x not in filtered_values:
        return None
    elif ":" in x:
        num, sec = float(x.split(':')[0]), float(x.split(':')[1])
        if sec < 30:
            sec2 = 0
        else:
            sec2 = 0.5
        return num + sec2
    else:
        return float(x)


# Apply the function to the 'time_left' column
df['time_left'] = df['time_left'].apply(lambda x: convert_match_length(x, df['time_left']))
df['match_length'] = df['match_length'].apply(lambda x: convert_match_length(x, df['match_length']))

# fix typos in map
df['map'] = df['map'].str.replace('junkerTown', 'junkertown')
df['map'] = df['map'].str.replace('template of anubis', 'temple of anubis')

# fix typos in names
def fixName(x):
    if x == "luico":
        return "lucio"
    elif x == "torbjorn":
        return "torbojorn"
    elif x == "" or x == "error" or x == "After 4 minutes left after 1st point" or x == "After 1st round" or x == "After 1st point":
        return None
    else:
        return x

df['character_1'] = df['character_1'].apply(fixName)
df['character_2'] = df['character_2'].apply(fixName)
df['character_3'] = df['character_3'].apply(fixName)

df['communication'] = df['communication'].apply(lambda x: None if x == "-" else x)
df['team_role'] = df['team_role'].apply(lambda x: None if x == "w" else x)

# convert to numeric
list_to_numeric = ["round","score_distance",
                   "sr_start", "sr_finish", "sr_delta",
                   "streak_number","my_team_sr", "enemy_team_sr","team_sr_delta"]
for col in list_to_numeric:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Distribution of game results

# Set the style of seaborn
sns.set_style("whitegrid")

# Plot the distribution of game results
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='result', order=df['result'].value_counts().index, palette="crest")
plt.title('Distribution of Game Results')
plt.ylabel('Number of Games')
plt.show()

# the most frequently played maps
# Plot the most played maps
plt.figure(figsize=(14, 8))
sns.countplot(data=df, y='map', order=df['map'].value_counts().index, palette="crest")
plt.title('Most Played Maps')
plt.xlabel('Number of Games')
plt.show()

# Plot the distribution of team roles
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='team_role', order=df['team_role'].value_counts().index, palette="crest")
plt.title('Distribution of Team Roles')
plt.ylabel('Number of Games')
plt.show()

# combining the three character columns into one
characters_data = pd.concat([df['character_1'], df['character_2'], df['character_3']])

# plot the most popular characters
# plt.figure(figsize=(14, 8))
# sns.countplot(data=characters_data, y=characters_data, order=characters_data.value_counts().index, palette="crest")
# plt.title('Most Popular Characters Used')
# plt.xlabel('Number of Games')
# plt.ylabel('Character')
# plt.show()

# Create a cross-tabulation of psychological condition vs. game result
psych_condition_vs_result = pd.crosstab(df['psychological_condition'], df['result'])

# Plot the cross-tabulation
psych_condition_vs_result.plot(kind='bar', stacked=True, figsize=(12, 7), colormap="crest")
plt.title('Game Results by Psychological Condition')
plt.ylabel('Number of Games')
plt.xlabel('Psychological Condition')
plt.show()

# interactive plot
import plotly.express as px

# Convert the crosstab dataframe to a long format for plotly
long_df = psych_condition_vs_result.reset_index().melt(id_vars='psychological_condition',
                                                       value_vars=['win', 'loss', 'draw'],
                                                       var_name='result', value_name='count')

# Create an interactive bar chart using plotly
fig = px.bar(long_df, x='psychological_condition', y='count', color='result',
             title="Game Results by Psychological Condition",
             labels={'psychological_condition': 'Psychological Condition'},
             color_discrete_sequence=px.colors.sequential.Viridis)

fig.show()