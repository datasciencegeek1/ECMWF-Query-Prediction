import numpy as np
import re
import pandas as pd
import random
import glob
import os

def read_folder(folder_path):

    # Define the keys to remove
    keys_to_remove = ['$age', '$reqno','$postprocessing','$elapsed','$status',
                       '$reason','$system', '$online', '$Disk_files', 
                       '$Fields_online', '$transfertime', '$readfiletime',
                       '$queuetime','$bytes_offline','$fields_offline', '$tape_files',
                       '$tapes', '$duplicates','$reason','$password','$expect','bytes', 'written'
                       '$email']

    # Initialize an empty list to store modified lines from all files
    combined_modified_lines = []

    # Iterate through each file in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith('.log'):
            log_file_path = os.path.join(folder_path, filename)

            # Read the content of the .log file
            with open(log_file_path, 'r') as file:
                log_content = file.read()

            # Split the log content into individual logs based on '$startdate'
            logs = log_content.split('$startdate=')

            # Remove the empty string at the beginning
            logs = [log for log in logs if log.strip()]

            # Randomly select 10000 logs
            random_logs = random.sample(logs, 10000)
            
            # Initialize an empty list to store modified logs
            modified_logs = []

            # Iterate through each log and split it into lines
            for log in random_logs:
                lines = log.strip().split('\n')
                modified_log_lines = []

                # Iterate through each line and remove the specified keys and their values
                for line in lines:
                    for key in keys_to_remove:
                        line = re.sub(rf'{re.escape(key)}=\'[^\']+\'\s*;', '', line)
                    modified_log_lines.append(line)

                # Join the modified lines back into a single log
                modified_log = '\n'.join(modified_log_lines)

                # Append the modified log with the $startdate field
                modified_logs.append('$startdate=' + modified_log)

            # Combine the modified logs from this file into a single text
            combined_modified_log_text = '\n'.join(modified_logs)

            # Add the combined modified log from this file to the overall list
            combined_modified_lines.append(combined_modified_log_text)

    # Join the combined modified logs from all files into a single text
    combined_modified_log_text = '\n'.join(combined_modified_lines)

    # Define the output file path
    output_file_path = '/Users/anas/Documents/UoR/MSc Project/Data/combined_modified_logs.txt'

    # Write the combined modified log text to the output file
    with open(output_file_path, 'w') as output_file:
        output_file.write(combined_modified_log_text)

def count_logs_in_file(file_path):
    # Read the content of the .log file
    with open(file_path, 'r') as file:
        log_content = file.read()

    # Split the log content into individual logs based on '$startdate'
    logs = log_content.split('$startdate=')

    # Remove the empty string at the beginning
    logs = [log for log in logs if log.strip()]

    return len(logs)

def read_file(input_file_path):

    # Get the count of logs in the file
    log_count = count_logs_in_file(input_file_path)
    print(f'Total number of logs in the file: {log_count}')

     # Define the keys to remove
    keys_to_remove = ['$age', '$reqno', '$postprocessing', '$elapsed', '$status',
                      '$reason', '$system', '$online', '$Disk_files',
                      '$Fields_online', '$transfertime', '$readfiletime',
                      '$queuetime', '$bytes_offline', '$fields_offline', '$tape_files',
                      '$tapes', '$duplicates', '$reason', '$password', '$expect', 'bytes', 'written',
                      '$email']
   
    with open(input_file_path, 'r') as file:
        log_content = file.read()

    # Split the log content into individual logs based on '$startdate'
    logs = log_content.split('$startdate=')

    # Remove the empty string at the beginning
    logs = [log for log in logs if log.strip()]

    # Initialize an empty list to store modified logs
    modified_logs = []

    # Iterate through each log and split it into lines
    for log in logs:
        lines = log.strip().split('\n')
        modified_log_lines = []

        # Iterate through each line and remove the specified keys and their values
        for line in lines:
            for key in keys_to_remove:
                line = re.sub(rf'{re.escape(key)}=\'[^\']+\'\s*;', '', line)
            modified_log_lines.append(line)

        # Join the modified lines back into a single log
        modified_log = '\n'.join(modified_log_lines)

        # Append the modified log with the $startdate field
        modified_logs.append('$startdate=' + modified_log)

    # Combine the modified logs from this file into a single text
    combined_modified_log_text = '\n'.join(modified_logs)

    # Define the output file path
    output_file_path = '/Users/anas/Documents/UoR/MSc Project/Data/sequential_combined_logs.txt'

    # Write the combined modified log text to the output file
    with open(output_file_path, 'w') as output_file:
        output_file.write(combined_modified_log_text)


    return combined_modified_log_text

def read_data_tocsv(file_path):
    
    dataframe = pd.DataFrame(sample_for_eda(file_path, sample_size=1000000)) 
    dataframe = dataframe.drop(dataframe.columns[0], axis=1)
    dataframe.to_csv("/Users/anas/Documents/UoR/MSc Project/Report/Logs/SequentialEngineeringv3.csv")

    return dataframe

def read_samples_fromcsv():

    df1 = pd.read_csv("/Users/anas/Documents/UoR/MSc Project/Report/Logs/FeatureEngineering1.csv")
    df2 = pd.read_csv("/Users/anas/Documents/UoR/MSc Project/Report/Logs/FeatureEngineering2.csv")
    df3 = pd.read_csv("/Users/anas/Documents/UoR/MSc Project/Report/Logs/FeatureEngineering3.csv")
    
    combined_df_all = pd.concat([df1, df2, df3], axis=0)

    return combined_df_all


def sample_for_eda(file_path, sample_size):
    
        # Read the text file
    with open(file_path, 'r') as file:
        data = file.read()

    # Split the log content into individual logs based on '$startdate'
        logs = data.split('$startdate=')

        # Remove the empty string at the beginning
        logs = [log for log in logs if log.strip()]
       
       # Add '$startdate=' back to each log entry
        logs = ['$startdate=' + log for log in logs[:sample_size]]


        sequential_logs = logs[:sample_size]

        # Convert the list of logs to a single string
        logs_as_string = '\n'.join(sequential_logs)

    return logs_as_string, sequential_logs

def process_data_to_dfv2(file_path):

    df = pd.DataFrame()
    data = []
    text, lines = sample_for_eda(file_path, sample_size=100000)
    
    sampled_text_file = "/Users/anas/Documents/UoR/MSc Project/Data/sample_final_sequential_logs.txt"
    
    # Export extracted records raw data for future use
    with open(sampled_text_file, 'w') as file:
        file.write(text)


    count = 0

    for line in lines:
        line = line.strip()
        count += 1

        if line:
            pairs = line.split(';')
            values = {}

            for pair in pairs:
                if '=' in pair:
                    key, value = pair.split('=', 1)
                    values[key] = value

            data.append(values)

    df = pd.DataFrame(data)
        
    # Export the file for future use for EDA - raw data
    df.to_csv('/Users/anas/Documents/UoR/MSc Project/Report/Logs/SequentialEngineeringv3.csv')
    print(df.info)

    return df


def process_data_to_df(file_path):

    df = pd.DataFrame()
    data = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        count = 0

        for line in lines:
            line = line.strip()
            count += 1

            if line:
                pairs = line.split(';')
                values = {}

                for pair in pairs:
                    if '=' in pair:
                        key, value = pair.split('=', 1)
                        values[key] = value

                data.append(values)

        df = pd.DataFrame(data)
        
        # Export the file for future use
        df.to_csv('/Users/anas/Documents/UoR/MSc Project/Data/sequential_sampled_df.csv')
        print(df.info)

    return df

def read_sample_log(file_path, sample_size, drop_keys):

    # Read the text file
    with open(file_path, 'r') as file:
        data = file.read()

     # Define the keys to remove
        keys_to_remove = ['$age', '$reqno','$postprocessing','$elapsed','$status',
                       '$reason','$system', '$online', '$Disk_files', 
                       '$Fields_online', '$transfertime', '$readfiletime',
                       '$queuetime','$bytes_offline','$fields_offline', '$tape_files',
                       '$tapes', '$duplicates','$reason','$password','$expect','bytes', 'written'
                       '$email'] + drop_keys
    # Split the log content into individual logs based on '$startdate'
        logs = data.split('$startdate=')

        # Remove the empty string at the beginning
        logs = [log for log in logs if log.strip()]

        # Randomly select 500 logs

        random_logs = random.sample(logs, sample_size)
        
        # Initialize an empty list to store modified logs
        modified_logs = []

        # Iterate through each log and split it into lines
        for log in random_logs:
            lines = log.strip().split('\n')
            modified_log_lines = []

    # Iterate through each line and remove the specified keys and their values
            for line in lines:
                for key in keys_to_remove:
                    line = re.sub(rf'{re.escape(key)}=\'[^\']+\'\s*;', '', line)
                modified_log_lines.append(line)
            # Iterate through each line and remove the specified keys and their values
            for line in lines:
                modified_log_lines.append(line)

            # Join the modified lines back into a single log
            modified_log = '\n'.join(modified_log_lines)

            # Append the modified log with the $startdate field
            modified_logs.append('$startdate=' + modified_log)

        # Combine the modified logs from this file into a single text
        combined_modified_log_text = '\n'.join(modified_logs)

    return combined_modified_log_text


def process_csv_reverse(file_path, max_lines = 10000):

    df = pd.DataFrame()
    data = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        lines.reverse()
        count = 0

        for line in lines:
            line = line.strip()
            count += 1

            if count > max_lines:
                break

            if line:
                pairs = line.split(';')
                values = {}

                for pair in pairs:
                    if '=' in pair:
                        key, value = pair.split('=', 1)
                        values[key] = value

                data.append(values)

        df = pd.DataFrame(data)
    return df


def merge_data():

     # Define the paths to the three output files
    output_file1_path = '/Users/anas/Documents/UoR/MSc Project/Data/combined_modified_logs1.txt'
    output_file2_path = '/Users/anas/Documents/UoR/MSc Project/Data/combined_modified_logs2.txt'
    output_file3_path = '/Users/anas/Documents/UoR/MSc Project/Data/combined_modified_logs3.txt'

    # Define the path for the final combined output file
    final_output_file_path = '/Users/anas/Documents/UoR/MSc Project/Data/final_combined_logs.txt'

    # Initialize an empty string to store the combined content
    combined_content = ""

    # Read the content of the first file and append it to combined_content
    with open(output_file1_path, 'r') as file1:
        combined_content += file1.read()

    # Read the content of the second file and append it to combined_content
    with open(output_file2_path, 'r') as file2:
        combined_content += file2.read()

    # Read the content of the third file and append it to combined_content
    with open(output_file3_path, 'r') as file3:
        combined_content += file3.read()

    # Write the combined content to the final output file
    with open(final_output_file_path, 'w') as final_output_file:
        final_output_file.write(combined_content)

def read_sequentialdata(file_path, sample_size, drop_keys):

    # Read the text file
    with open(file_path, 'r') as file:
        data = file.read()

    # Split the log content into individual logs based on '$startdate'
        logs = data.split('$startdate=')

        # Remove the empty string at the beginning
        logs = [log for log in logs if log.strip()]

        sequential_logs = logs[:sample_size]
        
        # Initialize an empty list to store modified logs
        modified_logs = []

        # Iterate through each log and split it into lines
        for log in sequential_logs:
            lines = log.strip().split('\n')
            modified_log_lines = []

    # Iterate through each line and remove the specified keys and their values
            for line in lines:
                for key in drop_keys:
                    line = re.sub(rf'{re.escape(key)}=\'[^\']+\'\s*;', '', line)
                modified_log_lines.append(line)
            
            # for line in lines:
            #     modified_log_lines.append(line)

            # Join the modified lines back into a single log
            modified_log = '\n'.join(modified_log_lines)

            # Append the modified log with the $startdate field
            modified_logs.append('$startdate=' + modified_log)

        # Combine the modified logs from this file into a single text
        combined_modified_log_text = '\n'.join(modified_logs)

    return combined_modified_log_text

def read_total_logs(folder_path):

    # Iterate through each file in the folder
    log_count = 0
    for filename in os.listdir(folder_path):
        if filename.endswith('.log'):
            log_file_path = os.path.join(folder_path, filename)

            # Read the content of the .log file
            with open(log_file_path, 'r') as file:
                log_content = file.read()

            # Split the log content into individual logs based on '$startdate'
            logs = log_content.split('$startdate=')

            # Remove the empty string at the beginning

            logs = [log for log in logs if log.strip()]

            print(f'Total number of logs: {len(logs)}')
            log_count += len(logs)
        
            
    print(f'Total number of logs in all files: {log_count}')

    return

if __name__ == '__main__':
    pass