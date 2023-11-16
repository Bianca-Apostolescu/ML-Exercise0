# !pip install openml

import matplotlib.pyplot as plt
# Basic libraries
import numpy as np
# Working with OpenML dataset
import openml
import pandas as pd

# Working with files

# Download dataset based on OpenML dataset ID

# Name = Crimes-in-Los-angeles-from-2010
# ID = 43579
# URL = https://www.openml.org/search?type=data&sort=runs&status=active&id=43579

dataset = openml.datasets.get_dataset(43579)

# Print a summary
print(
    f"This is dataset '{dataset.name}', the target feature is "
    f"'{dataset.default_target_attribute}'"
)

print(f"URL: {dataset.url}")
print(dataset.description[:500])

# Get the actual data (pandas dataframe and uninteresting metadata)

big_df, *_ = dataset.get_data()
print(f"The length of the '{dataset.name}' is = {len(big_df)}")
print(f"The number of features for the '{dataset.name}' is = {len(big_df.columns)}")
big_df.head()

# Checking for each column the type and the number of distinct and similar values (also see if there are missing values)

for col in big_df.columns:
    print(col)
    print(big_df[col].value_counts())
    print(f"Number of missing values = {big_df[col].isna().sum()}")
    print(f"Type of dataframe column = {big_df[col].dtypes}")
    print('\n')

# Convert 'date_reported' to a datetime object
big_df['Date_Reported_format'] = pd.to_datetime(big_df['Date_Reported'], format='%m/%d/%Y')

# Filter the DataFrame to include only years from 2010 to 2017
start_year = 2010
end_year = 2017

big_df = big_df[
    (big_df['Date_Reported_format'].dt.year >= start_year) & (big_df['Date_Reported_format'].dt.year <= end_year)]

# Distribution for datetime type columns

datetime_fields = ['Date_Reported', 'Date_Occurred']
datetime_df = big_df.copy()

for field in datetime_fields:
    datetime_df[field] = pd.to_datetime(datetime_df[field])
    datetime_df[field + '_year'] = datetime_df[field].dt.year

    result = datetime_df.groupby(field + '_year')[field].agg(['count', 'mean']).reset_index()

    # Plotting the data
    plt.figure(figsize=(6, 4))
    plt.bar(result[field + '_year'], result['count'], label='Count')
    plt.xlabel('Year')
    plt.ylabel('Count')
    plt.title(f'Bucket Distribution by Year for {field}')
    plt.legend()
    plt.show()


# Distribution for time type columns

def extract_hour(time_str):
    time_str = str(time_str)
    time_str = time_str.zfill(4)  # Ensure at least 4 digits
    hour_str = time_str[:2]
    return int(hour_str)


datetime_fields = ['Time_Occurred']
datetime_df = big_df.copy()

for field in datetime_fields:
    datetime_df[field + '_hour'] = datetime_df[field].apply(extract_hour)

    result = datetime_df.groupby(field + '_hour')[field].agg(['count', 'mean']).reset_index()

    # Plotting the data
    plt.figure(figsize=(6, 4))
    plt.bar(result[field + '_hour'], result['count'], label='Count')
    plt.xlabel('Hours')
    plt.ylabel('Count')
    plt.title(f'Bucket Distribution by Hours for {field}')
    plt.legend()
    plt.show()


datetime_fields = ['Date_Reported', 'Date_Occurred']
datetime_df = big_df.copy()

# Dictionary to map month numbers to month names
month_names = {
    1: 'January', 2: 'February', 3: 'March', 4: 'April',
    5: 'May', 6: 'June', 7: 'July', 8: 'August',
    9: 'September', 10: 'October', 11: 'November', 12: 'December'
}

for field in datetime_fields:
    datetime_df[field] = pd.to_datetime(datetime_df[field])
    datetime_df[field + '_month'] = datetime_df[field].dt.month

    result = datetime_df.groupby(field + '_month')[field].agg(['count', 'mean']).reset_index()

    # Map month numbers to month names
    result[field + '_month'] = result[field + '_month'].map(month_names)

    # Plotting the data
    plt.figure(figsize=(8, 6))
    plt.bar(result[field + '_month'], result['count'], label='Count')
    plt.xlabel('Month')
    plt.ylabel('Count')
    plt.title(f'Bucket Distribution by Month for {field}')
    plt.legend()
    plt.xticks(rotation=45)
    plt.show()

# Distribution for numerical values - reduced number
# Area_IDs varying form 1 to 21

datetime_fields = ['Area_Name', 'Victim_Sex', 'Victim_Descent', 'Status_Description']
datetime_df = big_df.copy()

for field in datetime_fields:
    datetime_df = big_df.copy()

    if field == 'Victim_Descent':

        # Transforming victim's ethnicity code into whole word

        mapped_descents = {'A': 'Other Asian', 'B': 'Black', 'C': 'Chinese', 'D': 'Cambodian', 'F': 'Filipino',
                           'G': 'Guamanian', 'H': 'Hispanic/Latin/Mexican', 'I': 'American Indian/Alaskan Native',
                           'J': 'Japanese', 'K': 'Korean', 'L': 'Laotian', 'O': 'Other', 'P': 'Pacific Islander',
                           'S': 'Samoan', 'U': 'Hawaiian', 'V': 'Vietnamese', 'W': 'White', 'X': 'Unknown',
                           'Z': 'Asian Indian'}

        datetime_df['Victim_Descent_New'] = datetime_df['Victim_Descent']
        datetime_df['Victim_Descent_New'] = datetime_df['Victim_Descent_New'].replace(mapped_descents)

        datetime_df['Victim_Descent_New' + '_bucket'] = datetime_df['Victim_Descent_New']

        result = datetime_df.groupby('Victim_Descent_New' + '_bucket')['Victim_Descent_New'].agg(
            ['count']).reset_index()

        # Plotting the data
        plt.figure(figsize=(6, 4))
        plt.bar(result['Victim_Descent_New' + '_bucket'], result['count'], label='Count')
        plt.xlabel('Bucket')
        plt.xticks(result['Victim_Descent_New' + '_bucket'], result['Victim_Descent_New' + '_bucket'], rotation=45,
                   ha='right')
        plt.ylabel('Count')
        plt.title(f'Bucket Distribution by Bucket for {field}')
        plt.legend()
        plt.show()

    elif field == 'Victim_Sex':

        # Transforming victim's gender code into whole word

        gender = ['F', 'M']
        datetime_df = datetime_df[datetime_df['Victim_Sex'].isin(gender)]
        mapped_gender = {'F': 'Female', 'M': 'Male'}

        datetime_df['Victim_Sex_New'] = datetime_df['Victim_Sex']
        datetime_df['Victim_Sex_New'] = datetime_df['Victim_Sex_New'].replace(mapped_gender)

        datetime_df['Victim_Sex_New' + '_bucket'] = datetime_df['Victim_Sex_New']

        result = datetime_df.groupby('Victim_Sex_New' + '_bucket')['Victim_Sex_New'].agg(['count']).reset_index()

        # Plotting the data
        plt.figure(figsize=(6, 4))
        plt.bar(result['Victim_Sex_New' + '_bucket'], result['count'], label='Count')
        plt.xlabel('Bucket')
        plt.xticks(result['Victim_Sex_New' + '_bucket'], result['Victim_Sex_New' + '_bucket'], rotation=45, ha='right')
        plt.ylabel('Count')
        plt.title(f'Bucket Distribution by Bucket for {field}')
        plt.legend()
        plt.show()

    else:
        datetime_df[field + '_bucket'] = datetime_df[field]

        result = datetime_df.groupby(field + '_bucket')[field].agg(['count']).reset_index()

        # Plotting the data
        plt.figure(figsize=(6, 4))
        plt.bar(result[field + '_bucket'], result['count'], label='Count')
        plt.xlabel('Bucket')
        plt.xticks(result[field + '_bucket'], result[field + '_bucket'], rotation=45, ha='right')
        plt.ylabel('Count')
        plt.title(f'Bucket Distribution by Bucket for {field}')
        plt.legend()
        plt.show()

datetime_df = big_df.copy()

# Define age buckets
age_bins = [0, 18, 25, 40, 60, 120]
age_labels = ['0-17', '18-24', '25-39', '40-59', '60+']

# Create a new column 'Age_Bucket' in the DataFrame
datetime_df['Age_Bucket'] = pd.cut(datetime_df['Victim_Age'], bins=age_bins, labels=age_labels, right=False)

# Adding year as a column
datetime_df['Date_Occurred_year'] = pd.to_datetime(datetime_df['Date_Occurred']).dt.year

# Group by 'Date_Occurred_year' and 'Age_Bucket' to get counts
age_by_year = datetime_df.groupby(['Date_Occurred_year', 'Age_Bucket']).size().unstack(fill_value=0)

# Print to get exact numbers
print(age_by_year)

# Plot age groups
age_by_year.plot(kind='bar', stacked=False, figsize=(12, 6))

plt.title("Age Distribution by Year")
plt.xlabel("Year")
plt.ylabel("Count")
plt.legend(title="Age Bucket")
plt.show()

datetime_df = big_df.copy()

# Adding year as a column
datetime_df['Date_Occurred_year'] = pd.to_datetime(datetime_df['Date_Occurred']).dt.year

# Group by 'Crime_Code_Description' and 'Date_Occurred_year' to get the count
crime_by_year = datetime_df.groupby(['Date_Occurred_year', 'Weapon_Description']).size().unstack(fill_value=0)

# Specify the crime types you want to include in the chart
crime_types = ['STRONG-ARM (HANDS, FIST, FEET OR BODILY FORCE)', 'RIFLE', 'STICK', 'SEMI-AUTOMATIC PISTOL',
               'KNIFE WITH BLADE 6INCHES OR LESS', 'OTHER KNIFE', 'UNKNOWN FIREARM', 'REVOLVER', 'TOY GUN', 'VEHICLE']

# Filter the data for the selected crime types
crime_by_year = crime_by_year[crime_types]

# Plot the data
plt.figure(figsize=(12, 8))
for crime_type in crime_by_year.columns:
    plt.plot(crime_by_year.index, crime_by_year[crime_type], label=crime_type)

plt.title("Crime Type by Year")
plt.xlabel("Year")
plt.ylabel("Count")
plt.legend(loc='center right', bbox_to_anchor=(1.75, 0.5), ncol=1, labelspacing=2)

plt.xlim([2010, 2017])  # 2018 is not a full year
plt.show()

datetime_df = big_df.copy()

# Adding year as a column
datetime_df['Date_Occurred_year'] = pd.to_datetime(datetime_df['Date_Occurred']).dt.year

# Group by 'Crime_Code_Description' and 'Date_Occurred_year' to get the count
crime_by_year = datetime_df.groupby(['Date_Occurred_year', 'Weapon_Description']).size().unstack(fill_value=0)

# Specify the crime types you want to include in the chart
crime_types = ['RIFLE', 'STICK', 'SEMI-AUTOMATIC PISTOL',
               'KNIFE WITH BLADE 6INCHES OR LESS', 'OTHER KNIFE', 'UNKNOWN FIREARM', 'REVOLVER', 'TOY GUN', 'VEHICLE']

# Filter the data for the selected crime types
crime_by_year = crime_by_year[crime_types]

# Plot the data
plt.figure(figsize=(12, 8))
for crime_type in crime_by_year.columns:
    plt.plot(crime_by_year.index, crime_by_year[crime_type], label=crime_type)

plt.title("Crime Type by Year")
plt.xlabel("Year")
plt.ylabel("Count")
plt.legend(loc='center right', bbox_to_anchor=(1.75, 0.5), ncol=1, labelspacing=2)

plt.xlim([2010, 2017])  # 2018 is not a full year
plt.show()

crime_by_year = datetime_df.groupby(['Weapon_Description', 'Date_Occurred_year']).count()['DR_Number'].sort_values(
    ascending=False)

# Access 'Crime_Code_Description' values and sort them by count in descending order
crime_types_by_count = crime_by_year.index.get_level_values('Weapon_Description').value_counts().sort_values(
    ascending=False)

print(crime_types_by_count.head(10))

datetime_df = big_df.copy()

# Adding year as a column
datetime_df['Date_Occurred_year'] = pd.to_datetime(datetime_df['Date_Occurred']).dt.year

# Group by 'Crime_Code_Description' and 'Date_Occurred_year' to get the count
crime_by_year = datetime_df.groupby(['Date_Occurred_year', 'Crime_Code_Description']).size().unstack(fill_value=0)

# Specify the crime types you want to include in the chart
crime_types = ['BATTERY - SIMPLE ASSAULT', 'THEFT PLAIN - PETTY (950  UNDER)', 'THEFT OF IDENTITY',
               'VANDALISM - FELONY (400  OVER, ALL CHURCH VANDALISMS) 0114', 'INTIMATE PARTNER - SIMPLE ASSAULT',
               'THEFT-GRAND (950.01  OVER)EXCPT,GUNS,FOWL,LIVESTK,PROD0036',
               'ASSAULT WITH DEADLY WEAPON, AGGRAVATED ASSAULT',
               'THEFT FROM MOTOR VEHICLE - PETTY (950  UNDER)', 'VEHICLE - STOLEN', 'SHOTS FIRED AT INHABITED DWELLING']

# Filter the data for the selected crime types
crime_by_year = crime_by_year[crime_types]

# Plot the data
plt.figure(figsize=(12, 8))
for crime_type in crime_by_year.columns:
    plt.plot(crime_by_year.index, crime_by_year[crime_type], label=crime_type)

plt.title("Crime Type by Year")
plt.xlabel("Year")
plt.ylabel("Count")
plt.legend(loc='center right', bbox_to_anchor=(1.75, 0.5), ncol=1, labelspacing=2)

plt.xlim([2010, 2017])  # 2018 is not a full year
plt.show()

crime_by_year = datetime_df.groupby(['Crime_Code_Description', 'Date_Occurred_year']).count()['DR_Number'].sort_values(
    ascending=False)

# Access 'Crime_Code_Description' values and sort them by count in descending order
crime_types_by_count = crime_by_year.index.get_level_values('Crime_Code_Description').value_counts().sort_values(
    ascending=False)

print(crime_types_by_count.head(10))

# Distribution for datetime type columns

datetime_fields = ['Date_Reported', 'Date_Occurred']
datetime_df = big_df.copy()

for field in datetime_fields:
    datetime_df[field] = pd.to_datetime(datetime_df[field])
    datetime_df[field + '_year'] = datetime_df[field].dt.year

    result = datetime_df.groupby(field + '_year')[field].agg(['count', 'mean']).reset_index()

    # Plotting the data
    plt.figure(figsize=(6, 4))
    plt.bar(result[field + '_year'], result['count'], label='Count')
    plt.xlabel('Year')
    plt.ylabel('Count')
    plt.title(f'Bucket Distribution by Year for {field}')
    plt.legend()
    plt.show()


# Distribution for Location type
# A lot of values with (0, 0) => unknown values theoretically - removed them

def extract_latitude(location_str):
    if location_str is not None:
        if float(location_str.split(',')[0]) != float(0):
            return float(location_str.split(',')[0])
        else:
            return np.nan
    else:
        return None


def extract_longitude(location_str):
    if location_str is not None:
        if float(location_str.split(',')[1]) != float(0):
            return float(location_str.split(',')[1])
        else:
            return np.nan
    else:
        return None


datetime_df = big_df.copy()

datetime_df['location'] = datetime_df['Location_'].str.strip('()')
datetime_df['latitude'] = datetime_df['location'].apply(extract_latitude)
datetime_df['longitude'] = datetime_df['location'].apply(extract_longitude)

plt.figure(figsize=(6, 4))
plt.scatter(datetime_df['longitude'], datetime_df['latitude'], marker='o', alpha=0.5)
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Scatter Plot of Location Data')
plt.grid(True)
plt.show()
