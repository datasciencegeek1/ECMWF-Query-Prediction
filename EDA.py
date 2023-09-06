import pandas as pd
import time
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import numpy as np
import re
import pandas as pd
import random
import glob
import ReadFiles as rf
import os

warnings.filterwarnings("ignore")


def draw_plots(df):

    columns = [f"Feature_{i}" for i in range(df.shape[1])]

    # Plot a histogram for one of the features using Matplotlib
    plt.figure(figsize=(10, 6))
    plt.hist(df['$startdate'], bins=100, color='blue', alpha=0.7)
    plt.title('Histogram of Feature 0')
    plt.xlabel('Feature Value')
    plt.ylabel('Frequency')
    plt.show()

    # Plot a histogram for one of the features using Matplotlib
    plt.figure(figsize=(10, 6))
    plt.hist(df['$retdate'], bins=100, color='blue', alpha=0.7)
    plt.title('Histogram of Feature 0')
    plt.xlabel('Feature Value')
    plt.ylabel('Frequency')
    plt.show()

    # Plot a scatter plot between two features using Seaborn
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x= df['$startdate'], y= df['$disk_files'], data=df, alpha=0.5)
    plt.title('Scatter Plot between Feature 1 and Feature 2')
    plt.xlabel('Feature 1 Value')
    plt.ylabel('Feature 2 Value')
    plt.show()


def feature_count(df):
    # Get the number of features
    n_features = len(df.columns)

    # Iterate over the features
    for i in range(0, n_features, 2):
        # Get the features in the current iteration
        features_i = df.columns[i:i + 2]

        # Create a plot
        fig, ax = plt.subplots()

        # Plot the frequency of each feature in the plot
        df[features_i].value_counts().plot(kind='bar', ax=ax)
        ax.set_title('Frequency of Features {}'.format(i))

        # Show the plot
        plt.tight_layout()
        plt.show()

def queries_by_date(df):

    # Group the data by 'date' and count the number of queries for each day
    daily_query_counts = df['$date'].value_counts().sort_index()

    # Create a plot
    plt.figure(figsize=(10, 6))
    plt.plot(daily_query_counts.index, daily_query_counts.values, marker='o', linestyle='-', color='b')
    plt.title('Number of Queries Generated Each Day')
    plt.xlabel('Date')
    plt.ylabel('Number of Queries')
    plt.xticks(rotation=45)
    plt.grid(True)

    # Show the plot
    plt.tight_layout()
    plt.show()

def queries_by_verb(df):

    # Group the data by 'date' and count the number of queries for each day
    daily_query_counts = df['$verb'].value_counts().sort_index()

    # Create a plot
    plt.figure(figsize=(10, 6))
    plt.plot(daily_query_counts.index, daily_query_counts.values, marker='o', color='b')
    plt.title('Number of Queries categorised by action to be taken on the query')
    plt.xlabel('Verb')
    plt.ylabel('Number of Queries')
    plt.xticks(rotation=45)
    plt.grid(True)

    # Show the plot
    plt.tight_layout()
    plt.show()

def convert_objects_to_numbers(df):
    
    # Step 1: Identify columns with object data type
    object_columns = df.select_dtypes(include=['object']).columns

    # Step 2: Convert each object column to numerical values
    for col in object_columns:
        unique_values = df[col].unique()
        mapping = {value: idx + 1 for idx, value in enumerate(unique_values)}
        df[col] = df[col].map(mapping).astype(np.float64)

    # Fill missing values with 0
    df.fillna(0, inplace=True)
  # df.to_csv("/Users/anas/Documents/UoR/MSc Project/Report/Logs/output2.csv", sep=',', encoding='utf-8', index=False)

    return df

def feature_correlation(df):

    df1 = df.copy()
    # Convert date columns to datetime objects
    
    df1['$startdate'] = pd.to_datetime(df['$startdate'])
    df1['$starttime'] = pd.to_datetime(df['$starttime'])
    
    correlation_matrix = df1.corr()

    # Print labels with correlations greater than 0.8
    high_correlation_labels = []
    for col in correlation_matrix.columns:
        correlated_cols = correlation_matrix.index[(correlation_matrix[col] > 0.8) | (correlation_matrix[col] < -0.8)].tolist()
        if len(correlated_cols) == 0:
            break
        else:
            correlated_cols.remove(col)  # Remove the column itself from the list
            for correlated_col in correlated_cols.copy():  # Use a copy of the list to iterate
                if (col, correlated_col) not in high_correlation_labels and (correlated_col, col) not in high_correlation_labels:
                    high_correlation_labels.append((col, correlated_col))
                    print(f"Correlation > 0.8: {col} - {correlated_col} - {correlation_matrix.loc[col, correlated_col]}")

    # Create masks for high and low correlations
    mask_high = np.abs(correlation_matrix) > 0.8
    mask_low = np.abs(correlation_matrix) < -0.7

    # Create correlation matrices with masked values
    correlation_matrix_high = np.where(mask_high, correlation_matrix, np.nan)

    # Plot the heatmap for high correlations
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix_high, annot=False, cmap='coolwarm', center=0)
    plt.title('Correlation Heatmap (High Correlations)')
    plt.xticks(range(len(correlation_matrix.columns)), correlation_matrix.columns, rotation=90)
    plt.yticks(range(len(correlation_matrix.columns)), correlation_matrix.columns, rotation = 0)
    plt.show()


def outliers(df):

    plt.figure(figsize=(12,8))
    sns.boxplot(data=df, orient='v')  # 'orient' specifies horizontal orientation
    plt.title('Box Plots of Features')
    plt.xlabel('Values')
    plt.show()

    # Calculate Z-scores for each feature
    z_scores = np.abs((df - df.mean()) / df.std())

    # Set a threshold for outlier detection (e.g., Z-score > 3)
    outlier_threshold = 3

    # Create a DataFrame of boolean values indicating outliers
    outliers = z_scores > outlier_threshold

    # Summarize which features have outliers
    features_with_outliers = outliers.any()

    print("Features with Outliers:", features_with_outliers[features_with_outliers == True].index)


# Read file in data frame
def read_into_df(filename):
    return pd.read_csv(filename)

# Check duplicate values
def check_duplicates(df):
  
    duplicated_df = df.duplicated(keep=False)

    # Count the number of unique and duplicate rows
    num_unique = (~duplicated_df).sum()
    num_duplicates = duplicated_df.sum()

    # Create a bar plot
    plt.bar(['Unique', 'Duplicate'], [num_unique, num_duplicates])
    plt.xlabel('Row Type')
    plt.ylabel('Count')
    plt.title('Unique vs. Duplicate Rows')

    # Display the plot
    plt.show()

# Check null values
def check_null_values(df):
    missing_values = df.isnull().sum()
    total_rows = len(df)
    percentage_null_values = (missing_values/total_rows)*100
    result = []
    for column, count in missing_values.items():
        percentage = percentage_null_values[column]
        result.append({'Column': column, 'Null Count': count, '% Null Values': percentage})

    result_df = pd.DataFrame(result)
    result_df.to_csv('/Users/anas/Documents/UoR/MSc Project/Report/Logs/Null Values.csv', index=False)
    return result_df

# Check unique values
def check_unique_values(df):
    result = []
    for column in df.columns:
        unique_values_count = df[column].nunique()
        unique_values = df[column].unique()
        result.append({'Column': column, 'Unique Values': unique_values_count,
                       'List of Unique Values': unique_values })


    result_df = pd.DataFrame(result)
    result_df.to_csv('/Users/anas/Documents/UoR/MSc Project/Report/Logs/Column Unique Values1.csv', index=False)


def check_missingvalues(df):

    #Finding missing values in the dataframe
    missing_values = df.isnull().sum()
    total_values = df.count()

    # Splitting columns into two groups
    num_columns = df.shape[1]
    half_num_columns = num_columns // 2
    first_half_columns = df.columns[:half_num_columns]
    second_half_columns = df.columns[half_num_columns:]

    # Creating subplots
    fig, axes = plt.subplots(2, 1, figsize=(15, 6))

    # Plotting for the first half of columns
    axes[0].bar(total_values[first_half_columns].index, total_values[first_half_columns].values, color='blue', label='Total Values')
    axes[0].bar(missing_values[first_half_columns].index, missing_values[first_half_columns].values, color='orange', label='Missing Values')
    axes[0].set_xlabel('Features')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Total Values vs Missing Values')
    axes[0].set_xticklabels(labels=first_half_columns, rotation=90)
    axes[0].legend()

    # Plotting for the second half of columns
    axes[1].bar(total_values[second_half_columns].index, total_values[second_half_columns].values, color='blue', label='Total Values')
    axes[1].bar(missing_values[second_half_columns].index, missing_values[second_half_columns].values, color='orange', label='Missing Values')
    axes[1].set_xlabel('Features')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Total Values vs Missing Values')
    axes[1].set_xticklabels(labels=second_half_columns, rotation=90)
    axes[1].legend()

    plt.tight_layout()
    plt.show()

    return

def list_unique_values(df, columns):
    result = []
    for column in df.columns:
        unique_values = df[column].unique()
        result.append({'Column': column,'List of Unique Values': unique_values})
    
    result_df = pd.DataFrame(result)
    result_df.to_csv('/Users/anas/Documents/UoR/MSc Project/Report/Logs/ListOfUniqueValues.csv', index=False)

if __name__ == '__main__':

    filename = "/Users/anas/Documents/UoR/MSc Project/Report/Logs/ConvertToLog.csv"
    with open(filename, 'r') as file:
        start = time.time()
        chunk = pd.read_csv(filename,chunksize=1000000)
        end = time.time()
        print("Read csv with chunks: ",(end-start),"sec")
        start = time.time()
        pd_df = pd.concat(chunk)
        end = time.time()
        print("Concatenation time: ",(end-start),"sec")
        

        start = time.time()
        check_unique_values(pd_df)
        end = time.time()
        print(f'List Unique values time: {end - start}, sec')


    
