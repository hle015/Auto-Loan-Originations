# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 23:24:44 2023

@author: huyen
"""

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file"""

### Update with GDP

# Loading the provided Excel file to understand the data dictionary
import pandas as pd
import numpy as np
from datetime import datetime

# Reading the Excel file
data_dictionary_path = 'C:/Users/huyen/My Drive/UCR_Drive/Apply Job 2022/Navy Federal Credit Union/Data Dictionary.xlsx'
data_dictionary = pd.read_excel(data_dictionary_path)

# Displaying the first few rows to understand the contents
data_dictionary.head()

# Cleaning up the data dictionary by setting appropriate column names and removing unnecessary rows
data_dictionary.columns = data_dictionary.iloc[0]
data_dictionary = data_dictionary.drop(0)

# Displaying the cleaned data dictionary
data_dictionary.reset_index(drop=True).head()

# Listing the names of all the sheets in the Excel file
# Reading all the sheets from the Excel file
all_sheets = pd.read_excel(data_dictionary_path, sheet_name=None)


# Listing the names of all the sheets in the Excel file
sheet_names = list(all_sheets.keys())
sheet_names

# Doing analysis
#1. Membership

# Displaying the first few rows of the 'Membership' sheet to understand the contents
membership_data = all_sheets['Membership']
membership_data.head()

##Visualize


import matplotlib.pyplot as plt

# Combining the month and year to create a datetime column
membership_data['Date'] = pd.to_datetime(membership_data['Year'].astype(str) + '-' + membership_data['Month'].astype(str) + '-01')

# Plotting the total membership over time
plt.figure(figsize=[15,6])
plt.plot(membership_data['Date'], membership_data['Total'], marker='o', linestyle='-', color='blue')
plt.title('Total Membership Trend Over Time')
plt.xlabel('Date')
plt.ylabel('Total Membership')
plt.grid(True)
plt.show()

## more eye-catching visualization

# Applying the ggplot style
plt.style.use('ggplot')

# Plotting the total membership over time with the ggplot style
plt.figure(figsize=[18,8])
plt.plot(membership_data['Date'], membership_data['Total'], marker='o', markersize=5, linestyle='-', color='#1f77b4', linewidth=2)
plt.fill_between(membership_data['Date'], membership_data['Total'], color='#1f77b4', alpha=0.1)
plt.title('Total Membership Trend Over Time', fontsize=16, fontweight='bold')
plt.xlabel('Date', fontsize=14)
plt.ylabel('Total Membership', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(color='gray', linestyle='--', linewidth=0.5)
plt.show()

#2. Application Volume

# Retrieving the 'Application Volume' sheet and displaying the first few rows
application_volume_data = all_sheets['Application Volume']
application_volume_data.head()

# Listing the unique values in the 'PRODUCT_GROUP' column
unique_product_groups = application_volume_data['PRODUCT_GROUP'].unique()
unique_product_groups

# Defining the categories related to auto loans
auto_loan_categories = ['LATE MODEL USED VEHICLE', 'NEW VEHICLE', 'USED VEHICLE']

# Filtering the dataset to include only the observations related to auto loans
auto_loans_data = application_volume_data[application_volume_data['PRODUCT_GROUP'].isin(auto_loan_categories)]

# Displaying the first few rows of the filtered dataset
auto_loans_data.head()

# Checking for missing values in the filtered auto loans dataset
missing_values_summary_auto_loans = auto_loans_data.isnull().sum()

# Combining the month and year to create a datetime column
auto_loans_data['DATE'] = pd.to_datetime(auto_loans_data['YEARENTERED'].astype(str) + '-' + 
                                         auto_loans_data['MONTHENTERED'].astype(str) + '-01')

# Dropping the original month and year columns
auto_loans_data.drop(['MONTHENTERED', 'YEARENTERED'], axis=1, inplace=True)

# Displaying the summary of missing values and the first few rows after preprocessing
missing_values_summary_auto_loans, auto_loans_data.head()

# One-hot encoding the 'PRODUCT_GROUP' column
#auto_loans_data = pd.get_dummies(auto_loans_data, columns=['PRODUCT_GROUP'], prefix='PRODUCT')

# Encoding the 'PREAPPROVE' column as binary (YES=1, NO=0)
auto_loans_data['PREAPPROVE'] = auto_loans_data['PREAPPROVE'].map({'YES': 1, 'NO': 0})

# Displaying the first few rows after encoding
auto_loans_data.head()

# Defining the mapping for encoding the risk tiers
tier_mapping = {
    'A+': 12, 'AA': 11, 'A': 10, 'A-': 9,
    'B': 8, 'B-': 7,
    'C': 6, 'C-': 5,
    'D': 4, 'D-': 3,
    'E': 2,
    'FT': 1
}

# Encoding the 'TIER' column using the defined mapping
auto_loans_data['TIER'] = auto_loans_data['TIER'].map(tier_mapping)

# Displaying the first few rows after encoding
auto_loans_data.head()

############ Do EDA for Application volume

# Calculating summary statistics for the numerical variables
summary_statistics = auto_loans_data.describe().transpose()
summary_statistics

#Visualization

# Plotting application volume trend over time
plt.figure(figsize=[18,6])
plt.plot(auto_loans_data.groupby('DATE')['VOLUME'].sum(), marker='o', linestyle='-', color='blue')
plt.title('Application Volume Trend Over Time')
plt.xlabel('Date')
plt.ylabel('Total Application Volume')
plt.grid(True)
plt.show()

# Plotting application volume by product group
#auto_loans_data.groupby('DATE').agg({
#    'PRODUCT_LATE MODEL USED VEHICLE': 'sum',
 #   'PRODUCT_NEW VEHICLE': 'sum',
 #   'PRODUCT_USED VEHICLE': 'sum'
#}).plot(figsize=[18,6], marker='o')
#plt.title('Application Volume Trend by Product Group')
#plt.xlabel('Date')
#plt.ylabel('Application Volume')
#plt.legend(['Late Model Used Vehicle', 'New Vehicle', 'Used Vehicle'])
#plt.grid(True)
#plt.show()

### Do visualization again


#3. Competitor rates

# Retrieving the 'Competitor Rates' sheet
competitor_rates_data_new = all_sheets['Competitor Rates']

# Converting the 'Report Date' column to datetime and filtering data from 2013 onwards
competitor_rates_data_new['Report Date'] = pd.to_datetime(competitor_rates_data_new['Report Date'])
competitor_rates_data_2013_onwards_new = competitor_rates_data_new[competitor_rates_data_new['Report Date'].dt.year >= 2012]

# Converting the 'Product Group' column to uppercase to match the 'Application Volume' data
competitor_rates_data_2013_onwards_new['Product Group'] = competitor_rates_data_2013_onwards_new['Product Group'].str.upper()

# Filtering the data to include only the categories related to auto loans
auto_loan_categories_new = ['NEW VEHICLE', 'LATE MODEL USED VEHICLE', 'USED VEHICLE']
auto_loans_competitor_rates_new = competitor_rates_data_2013_onwards_new[
    competitor_rates_data_2013_onwards_new['Product Group'].isin(auto_loan_categories_new)]

# Transforming the 'Report Date' variable to represent only the month and year
auto_loans_competitor_rates_new['Report Date'] = auto_loans_competitor_rates_new['Report Date'].apply(lambda x: datetime(year=x.year, month=x.month, day=1))

# Displaying the first few rows of the filtered and aligned data
auto_loans_competitor_rates_new.head()

# Do EDA

# Calculating summary statistics for the numerical variables in the 'Competitor Rates' data
competitor_rates_summary_statistics_new = auto_loans_competitor_rates_new[['Term', 'Rate']].describe().transpose()
competitor_rates_summary_statistics_new

# Do visualization

import matplotlib.pyplot as plt
import seaborn as sns

# Setting the style for the plots
sns.set_style("whitegrid")

# Plotting the trends in rates over time for different auto loan categories
plt.figure(figsize=[18, 6])
for category in auto_loan_categories_new:
    subset_data = auto_loans_competitor_rates_new[auto_loans_competitor_rates_new['Product Group'] == category]
    plt.plot(subset_data['Report Date'], subset_data['Rate'], label=category)

plt.title('Trends in Competitor Rates Over Time (Auto Loans)')
plt.xlabel('Report Date')
plt.ylabel('Rate (Interest Rate)')
plt.legend()
plt.grid(True)
plt.show()

# Histogram

# Plotting the distribution of rates for different auto loan categories using histograms
fig, axes = plt.subplots(1, 3, figsize=[18, 6], sharey=True)
fig.suptitle('Distribution of Competitor Rates (Auto Loans)')

for idx, category in enumerate(auto_loan_categories_new):
    subset_data = auto_loans_competitor_rates_new[auto_loans_competitor_rates_new['Product Group'] == category]
    sns.histplot(subset_data['Rate'], bins=15, kde=True, ax=axes[idx], color='skyblue')
    axes[idx].set_title(category)
    axes[idx].set_xlabel('Rate (Interest Rate)')
    axes[idx].set_ylabel('Frequency')
    axes[idx].grid(True)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

# Plotting box plots to compare the rates across different auto loan categories
plt.figure(figsize=[12, 6])
sns.boxplot(x='Product Group', y='Rate', data=auto_loans_competitor_rates_new, palette='pastel')
plt.title('Comparison of Competitor Rates (Auto Loans)')
plt.xlabel('Product Group (Auto Loan Categories)')
plt.ylabel('Rate (Interest Rate)')
plt.grid(True)
plt.show()

# 4. Moodys data

# Retrieving the 'Moodys' sheet
moodys_data = all_sheets['Moodys']

# Displaying the first few rows of the 'Moodys' sheet to understand the structure
moodys_data.head()

# Extracting the description row for column names
moodys_descriptions = moodys_data.iloc[0]

# Loading the actual time-series data, excluding the first few rows with metadata
moodys_time_series_data = moodys_data.iloc[4:]

# Setting the extracted descriptions as column names
moodys_time_series_data.columns = moodys_descriptions

# Resetting the index
moodys_time_series_data.reset_index(drop=True, inplace=True)

# Displaying the first few rows of the transformed 'Moodys' time-series data
moodys_time_series_data.head()

# Filtering the data based on the relevant columns based on descriptions
relevant_columns_by_description_newest = [
    "Description:",
    "Baseline Scenario (August 2022): Interest Rates: 3-Year Treasury Constant Maturities, (% p.a., NSA)",
    "Baseline Scenario (August 2022): NIPA: Gross Domestic Product, (Bil. Ch. 2012 USD, SAAR)",
    "Baseline Scenario (August 2022): Household Survey: Unemployment Rate, (%, SA)"
]

moodys_relevant_data_newest = moodys_time_series_data[relevant_columns_by_description_newest]

# Renaming columns for clarity
moodys_relevant_data_newest.columns = [
    "Date",
    "3-Year Treasury Interest Rate (%)",
    "GDP (Bil)",
    "Unemployment Rate (%)"
]

# Replacing 'ND' with NaN and converting columns to numeric data type
numeric_columns_moodys_newest = [
    "3-Year Treasury Interest Rate (%)",
    "GDP (Bil)",
    "Unemployment Rate (%)"
]
moodys_relevant_data_newest[numeric_columns_moodys_newest] = moodys_relevant_data_newest[numeric_columns_moodys_newest].apply(pd.to_numeric, errors='coerce')

# Converting the Date column to datetime format and extracting year and month for consistency
moodys_relevant_data_newest['Date'] = pd.to_datetime(moodys_relevant_data_newest['Date']).dt.to_period('M')

# Displaying the first few rows of the processed 'Moodys' data
moodys_relevant_data_newest.head()


# Converting the Date column to the appropriate format
moodys_relevant_data_newest['Date'] = moodys_relevant_data_newest['Date'].astype('period[M]')

# Filtering the data to keep only entries from 2010 onwards
moodys_data_2010_onwards = moodys_relevant_data_newest[moodys_relevant_data_newest['Date'] >= '2012-01']

# Displaying the first few rows of the filtered 'Moodys' data
moodys_data_2010_onwards.head()

### Visualization

# Converting the 'Date' column from period to datetime format for plotting
moodys_data_2010_onwards['Date'] = moodys_data_2010_onwards['Date'].dt.to_timestamp()

# Re-attempting the time series plots for each variable
fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(15, 12))

sns.lineplot(data=moodys_data_2010_onwards, x="Date", y="3-Year Treasury Interest Rate (%)", ax=axes[0], color="blue")
axes[0].set_title("3-Year Treasury Interest Rate Over Time", fontsize=16)
axes[0].set_ylabel("Interest Rate (%)", fontsize=14)


sns.lineplot(data=moodys_data_2010_onwards, x="Date", y="GDP (Bil)", ax=axes[1], color="green")
axes[1].set_title("Growth of GDP Over Time", fontsize=16)
axes[1].set_ylabel("Value Index", fontsize=14)

sns.lineplot(data=moodys_data_2010_onwards, x="Date", y="Unemployment Rate (%)", ax=axes[2], color="red")
axes[2].set_title("Unemployment Rate Over Time", fontsize=16)
axes[2].set_ylabel("Unemployment Rate (%)", fontsize=14)

plt.tight_layout()
plt.show()


#################################MERGE DATA#############################


# Merging the datasets based on the "Date" column

# Creating a "Date" column in the 'Membership' dataset by combining "Month" and "Year"
membership_data["Date"] = membership_data["Year"].astype(str) + "-" + membership_data["Month"].astype(str)
membership_data["Date"] = pd.to_datetime(membership_data["Date"]).dt.to_period('M')
membership_data.drop(columns=["Month", "Year"], inplace=True)

# Creating a "Date" column in the 'Application Volume' dataset by combining "MONTHENTERED" and "YEARENTERED"

auto_loans_competitor_rates_new["Date"] = auto_loans_competitor_rates_new["Report Date"].dt.to_period('M')
# Reprocessing the 'Competitor Rates' sheet again
auto_loans_data["Date"] = auto_loans_data["DATE"].dt.to_period('M')

# Creating a 'Date' columns

moodys_data_2010_onwards["Date"] = moodys_data_2010_onwards["Date"].dt.to_period('M')



# Starting with 'Application Volume' and 'Membership'
merged_data = pd.merge(auto_loans_data, membership_data, on="Date", how="left")

# Merge with Competitors rate

merged_data = pd.merge(merged_data, auto_loans_competitor_rates_new, 
                       left_on=['Date', 'TERM_CODE', 'PRODUCT_GROUP'], 
                       right_on=['Date', 'Term', 'Product Group'], 
                       how='inner').drop(columns=['Report Date','Term', 'Product Group'])

# Merge with Moodys

merged_data = pd.merge(merged_data, moodys_data_2010_onwards, on="Date", how="left")

merged_data = merged_data[merged_data['TERM_CODE'] != 999]

# Export the merged dataset to an Excel file
output_path = "C:/Users/huyen/My Drive/UCR_Drive/Apply Job 2022/Navy Federal Credit Union/Merged_Data_update_5.xlsx"
merged_data.to_excel(output_path, index=False)

output_path

##### Visualize after merging data##############

# 1. Volume

# Group data by Date and sum the application volumes
merged_data['Date'] = merged_data['Date'].dt.to_timestamp()

merged_data1 = merged_data.groupby('Date').agg({'VOLUME': 'sum'}).reset_index()

# Plot the total application volume over time
plt.figure(figsize=(15, 7))
plt.plot(merged_data1['Date'], merged_data1['VOLUME'], label='Auto Loan Application Volume', color='green')
plt.title('Auto Loan Application Volume Over Time')
plt.xlabel('Date')
plt.ylabel('Total Application Volume')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Volume by group

# Group data by Date and PRODUCT_GROUP, then sum the application volumes
grouped_by_product = merged_data.groupby(['Date', 'PRODUCT_GROUP']).agg({'VOLUME': 'sum'}).reset_index()

# Plot the application volume for each product group over time
plt.figure(figsize=(15, 7))
for product in ['LATE MODEL USED VEHICLE', 'NEW VEHICLE', 'USED VEHICLE']:
    subset = grouped_by_product[grouped_by_product['PRODUCT_GROUP'] == product]
    plt.plot(subset['Date'], subset['VOLUME'], label=product)

plt.title('Auto Loan Application Volume by Product Group Over Time')
plt.xlabel('Date')
plt.ylabel('Application Volume')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#2. Loan terms

# Plot the distribution of loan terms
plt.figure(figsize=(12, 6))
merged_data['TERM_CODE'].value_counts().sort_index().plot(kind='bar', color='purple')
plt.title('Distribution of Loan Terms')
plt.xlabel('Loan Term Code')
plt.ylabel('Number of Applications')
plt.grid(axis='y')
plt.tight_layout()
plt.show()

# 3. Preapprove

# Plot the distribution of pre-approval status
plt.figure(figsize=(8, 6))
merged_data['PREAPPROVE'].value_counts().plot(kind='bar', color='orange')
plt.title('Distribution of Pre-Approval Status')
plt.xlabel('Pre-Approval Status')
plt.ylabel('Number of Applications')
plt.grid(axis='y')
plt.tight_layout()
plt.show()

# 4. Risk tier

# Plot the distribution of risk tiers
plt.figure(figsize=(12, 6))
merged_data['TIER'].value_counts().sort_index().plot(kind='bar', color='teal')
plt.title('Distribution of Risk Tiers')
plt.xlabel('Risk Tier')
plt.ylabel('Number of Applications')
plt.grid(axis='y')
plt.tight_layout()
plt.show()

#5. Group by risk tier and loan term

# Group the data by risk tier and loan term, then sum the application volumes
grouped_by_tier_term = merged_data.groupby(['TIER', 'TERM_CODE']).size().unstack()

# Plot the stacked bar chart
grouped_by_tier_term.plot(kind='bar', stacked=True, figsize=(15, 7), colormap='viridis')
plt.title('Distribution of Loan Terms by Risk Tier')
plt.xlabel('Risk Tier')
plt.ylabel('Number of Applications')
plt.legend(title='Loan Term Code')
plt.grid(axis='y')
plt.tight_layout()
plt.show()


#############Do FORECAST#########################

####1. Cleaning data

# Checking for missing values in the merged dataset
missing_values = merged_data.isnull().sum()

# Displaying columns with missing values and their counts
missing_values[missing_values > 0]

# Imputing missing values
merged_data['PREAPPROVE'].fillna(merged_data['PREAPPROVE'].mode()[0], inplace=True)
merged_data['Total'].fillna(merged_data['Total'].mean(), inplace=True)

# Checking again for missing values to ensure they've been addressed
missing_values_after = merged_data.isnull().sum()

# Displaying columns with missing values and their counts after imputation
missing_values_after[missing_values_after > 0]

# Checking the data types of each column
data_types = merged_data.dtypes

# Displaying the data types
data_types

# Adjusting data types
merged_data['PREAPPROVE'] = merged_data['PREAPPROVE'].astype(int)

# Checking the data type and first few values of the 'Date' column

date_column_dtype_2 = merged_data['Date'].dtype
date_column_sample_2 = merged_data['Date'].head()

date_column_dtype_2, date_column_sample_2

# Converting the 'Date' column to datetime format
# Step 1: Convert the 'Date' column to a Period type
merged_data['Date'] = merged_data['Date'].astype('period[M]')

# Step 2: Convert the Period type to a timestamp
merged_data['Date'] = merged_data['Date'].dt.to_timestamp()

## OUTLIERS

# Function to calculate the percentage of outliers using the IQR method
def compute_outliers_percentage_iqr(data, column_name):
    Q1 = data[column_name].quantile(0.25)
    Q3 = data[column_name].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Filter the data for outliers
    outliers = data[(data[column_name] < lower_bound) | (data[column_name] > upper_bound)]
    
    # Calculate the percentage of outliers
    percentage = (len(outliers) / len(data)) * 100
    
    return percentage

# Compute the percentage of outliers for each numerical column using the IQR method
outliers_percentage_iqr = {}

#List of numerical columns to check for outliers

numerical_columns = ['VOLUME', 'BPS25', 'BPS100', 'Rate', '3-Year Treasury Interest Rate (%)', 'GDP (Bil)', 'Unemployment Rate (%)']

for col in numerical_columns:
    outliers_percentage_iqr[col] = compute_outliers_percentage_iqr(merged_data, col)

# Display the percentage of outliers for each column
outliers_percentage_iqr

# Imputation outliers

# Function to impute outliers with the mean using the IQR method
def impute_outliers_with_mean(data, column_name):
    Q1 = data[column_name].quantile(0.25)
    Q3 = data[column_name].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Calculate the mean of the column
    mean_value = data[column_name].mean()
    
    # Replace outliers with the mean
    data[column_name] = data.apply(lambda row: mean_value if (row[column_name] < lower_bound or row[column_name] > upper_bound) else row[column_name], axis=1)
    
    return data

# Apply imputation for each numerical column with detected outliers using the IQR method
for col in ['VOLUME', 'BPS25', 'BPS100', 'Unemployment Rate (%)']:
    merged_data = impute_outliers_with_mean(merged_data, col)

# Check if the imputation was successful by recomputing the percentage of outliers
outliers_percentage_after_imputation = {}
for col in ['VOLUME', 'BPS25', 'BPS100', 'Unemployment Rate (%)']:
    outliers_percentage_after_imputation[col] = compute_outliers_percentage_iqr(merged_data, col)

outliers_percentage_after_imputation

# Take log of GDP

#merged_data['GDP (Bil)'] = np.log(merged_data['GDP (Bil)'])

#merged_data['VOLUME'] = np.log(merged_data['VOLUME'])
#2. DO EDA

# Obtaining summary statistics for the dataset
summary_stats = merged_data.describe()

# Displaying the summary statistics
summary_stats

# Export the statistical description table to an Excel file
output_path = "C:/Users/huyen/My Drive/UCR_Drive/Apply Job 2022/Navy Federal Credit Union/statistical_description.xlsx"
summary_stats.to_excel(output_path)

output_path

# Encodign

# One-hot encoding the 'PRODUCT_GROUP' column
merged_data_encoded = pd.get_dummies(merged_data, columns=['PRODUCT_GROUP'], drop_first=True)

# Displaying the first few rows of the encoded dataset
merged_data_encoded.head()

###3. Feature importance

import seaborn as sns

# Calculating the correlation matrix
correlation_matrix = merged_data.corr()

# Extracting correlations of all features with "VOLUME"
volume_correlations = correlation_matrix['VOLUME'].sort_values(ascending=False)


# Plotting a heatmap for correlations of all features, but prioritizing "VOLUME"
plt.figure(figsize=(15, 12))

# Reordering the columns based on their correlation with 'VOLUME' for prioritization
ordered_columns = volume_correlations.index

# Plotting the heatmap
sns.heatmap(merged_data[ordered_columns].corr(), annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()

#########Hypothesis-based feature

import statsmodels.api as sm
# Prepare data for linear regression
X = merged_data_encoded.drop(columns=['VOLUME', 'DATE', 'Date'])
X = sm.add_constant(X)  # Adding a constant for the intercept
y = merged_data_encoded['VOLUME']

# Fit the linear regression model
model = sm.OLS(y, X).fit()

# Display the summary
model_summary = model.summary()
model_summary

# Creating time-based features
merged_data_encoded['Month'] = merged_data_encoded['Date'].dt.month
merged_data_encoded['Year'] = merged_data_encoded['Date'].dt.year
merged_data_encoded['Quarter'] = merged_data_encoded['Date'].dt.quarter

# Displaying the first few rows with the new time-based features
merged_data_encoded.head()

# Creating lag features for the "VOLUME" column for the previous three months
for i in range(1, 4):
    merged_data_encoded[f'VOLUME_Lag_{i}'] = merged_data_encoded['VOLUME'].shift(i)

# Displaying the first few rows with the new lag features
merged_data_encoded.head()

# Creating a moving average feature for the "VOLUME" column over a 3-month window
merged_data_encoded['VOLUME_MA_3'] = merged_data_encoded['VOLUME'].rolling(window=3).mean()

# Displaying the first few rows with the new moving average feature
merged_data_encoded.head()


### MODELS
## SPLIT DATA

from sklearn.model_selection import train_test_split

# Dropping rows with NaN values introduced due to feature engineering (like lag features and moving average)
merged_data_encoded = merged_data_encoded.dropna()

# Features and Target Variable
X = merged_data_encoded.drop(columns=['VOLUME', 'Date', 'DATE'])
y = merged_data_encoded['VOLUME']

# Splitting data into training (80%) and testing (20%) sets
train_size = int(0.8 * len(X))
X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

X_train.shape, X_test.shape

### LINER REGRESSION MODEL

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Initializing and training the linear regression model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Predicting on the testing set
y_pred = lr_model.predict(X_test)

# Evaluating the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

mae, mse, r2
## PREDICT FROM TRAINING

# Predicting on the training set
y_train_pred = lr_model.predict(X_train)

# Evaluating the model on training data
mae_train = mean_absolute_error(y_train, y_train_pred)
mse_train = mean_squared_error(y_train, y_train_pred)
r2_train = r2_score(y_train, y_train_pred)

mae_train, mse_train, r2_train

## Extracting coefficients and intercept
coefficients = lr_model.coef_
intercept = lr_model.intercept_

# Creating a DataFrame for the coefficients
coeff_df = pd.DataFrame(coefficients, index=X_train.columns, columns=['Coefficient'])

coeff_df, intercept

# Removing the lag features
X_train_simplified = X_train.drop(columns=['VOLUME_Lag_1', 'VOLUME_Lag_2', 'VOLUME_Lag_3'])
X_test_simplified = X_test.drop(columns=['VOLUME_Lag_1', 'VOLUME_Lag_2', 'VOLUME_Lag_3'])

# Training the simplified linear regression model
lr_model_simplified = LinearRegression()
lr_model_simplified.fit(X_train_simplified, y_train)

# Predicting on the simplified testing set
y_pred_simplified = lr_model_simplified.predict(X_test_simplified)

# Evaluating the simplified model
mae_simplified = mean_absolute_error(y_test, y_pred_simplified)
mse_simplified = mean_squared_error(y_test, y_pred_simplified)
r2_simplified = r2_score(y_test, y_pred_simplified)

mae_simplified, mse_simplified, r2_simplified


### Coefficeit for simplified model

# Extracting coefficients for the simplified model
coefficients_simplified = lr_model_simplified.coef_

# Creating a DataFrame for the coefficients
coeff_simplified_df = pd.DataFrame(coefficients_simplified, index=X_train_simplified.columns, columns=['Coefficient'])

coeff_simplified_df.sort_values(by='Coefficient', ascending=False)

## Time-series models

from statsmodels.tsa.arima.model import ARIMA
from datetime import datetime

# Convert the 'Date' column to datetime format and sort the dataframe by date
merged_data_encoded['Date'] = pd.to_datetime(merged_data_encoded['Date'])
merged_data_encoded = merged_data_encoded.sort_values(by='Date')

# Split data: 80% for training and 20% for testing
train_size = int(len(merged_data_encoded) * 0.8)
train, test = merged_data_encoded.iloc[:train_size], merged_data_encoded.iloc[train_size:]

# Define the training and testing sets
train_series = train.set_index('Date')['VOLUME']
test_series = test.set_index('Date')['VOLUME']

import statsmodels.api as sm
import matplotlib.pyplot as plt

# Plot ACF and PACF
fig, ax = plt.subplots(1, 2, figsize=(15, 7))

# ACF plot
sm.graphics.tsa.plot_acf(train_series, lags=40, ax=ax[0])

# PACF plot
sm.graphics.tsa.plot_pacf(train_series, lags=40, ax=ax[1])

plt.tight_layout()
plt.show()

# Fit ARIMA model using the parameters p=2, d=1, q=2
model_arima_optimized = ARIMA(train_series, order=(2,1,2))
arima_result_optimized = model_arima_optimized.fit()

# Forecast on the test set
forecast_optimized = arima_result_optimized.forecast(steps=len(test_series))

# Calculate error metrics for the optimized model
mae_arima_optimized = mean_absolute_error(test_series, forecast_optimized)
mse_arima_optimized = mean_squared_error(test_series, forecast_optimized)
r2_arima_optimized = r2_score(test_series, forecast_optimized)

mae_arima_optimized, mse_arima_optimized, r2_arima_optimized

## Fit actual values and predicted values

arima_in_sample_preds = arima_result_optimized.predict(start=1, end=len(merged_data['VOLUME']))

# Plot the actual data and ARIMA model's in-sample predictions

plt.figure(figsize=(15, 7))
plt.plot(merged_data['DATE'], merged_data['VOLUME'], label='Actual Volume', color='blue')
plt.plot(merged_data['DATE'], arima_in_sample_preds, label='ARIMA Predicted Volume', color='red', linestyle='--')
plt.title('Actual vs. ARIMA Predicted Auto Loan Origination Volume')
plt.xlabel('Date')
plt.ylabel('Volume')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

### Random forest

from sklearn.ensemble import RandomForestRegressor

# Define features and target
X_train_rf = X_train.copy()
X_test_rf = X_test.copy()
y_train_rf = y_train.copy()
y_test_rf = y_test.copy()

# Initialize Random Forest Regressor
rf = RandomForestRegressor(n_estimators=100, random_state=42)

# Fit the model
rf.fit(X_train_rf, y_train_rf)

# Predict on the test set
y_pred_rf = rf.predict(X_test_rf)

# Calculate error metrics for the Random Forest model
mae_rf = mean_absolute_error(y_test_rf, y_pred_rf)
mse_rf = mean_squared_error(y_test_rf, y_pred_rf)
r2_rf = r2_score(y_test_rf, y_pred_rf)

mae_rf, mse_rf, r2_rf

### Decision tree

from sklearn.tree import DecisionTreeRegressor

# Initialize Decision Tree Regressor
dt = DecisionTreeRegressor(random_state=42)

# Fit the model
dt.fit(X_train, y_train)

# Predict on the test set
y_pred_dt = dt.predict(X_test)

# Calculate error metrics for the Decision Tree model
mae_dt = mean_absolute_error(y_test, y_pred_dt)
mse_dt = mean_squared_error(y_test, y_pred_dt)
r2_dt = r2_score(y_test, y_pred_dt)

mae_dt, mse_dt, r2_dt


## Cross-validation

from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, make_scorer

# Define the models to test
models_selected = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(random_state=42)
}

# Create a custom scorer for RMSE
rmse_scorer = make_scorer(mean_squared_error, greater_is_better=False, squared=False)

# Evaluate each model using cross-validation with TimeSeriesSplit (given the nature of our data)
tscv = TimeSeriesSplit(n_splits=3)
cv_scores_selected = {}

for model_name, model in models_selected.items():
    score = -cross_val_score(model, X_train, y_train, cv=tscv, scoring=rmse_scorer).mean()
    cv_scores_selected[model_name] = score

cv_scores_selected
