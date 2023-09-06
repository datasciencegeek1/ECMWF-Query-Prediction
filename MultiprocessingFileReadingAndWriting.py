# Working correctly

import csv
import re
import pandas as pd

# Open the input log file
input_file = '/Users/anas/Documents/UoR/MSc Project/Report/Logs/20230301_anon.log'
output_file = '/Users/anas/Documents/UoR/MSc Project/Report/Logs/logtxt.csv'

# Create a dataframe
data = pd.DataFrame()

def read_log_file(input_file, data):
    with open(input_file, 'r') as file, open(output_file, 'w', newline='') as csvfile:
        # Create a CSV writer
        writer = csv.writer(csvfile)
        # Read each line
        for line in file:
            # Use regular expressions to extract keys and values
            matches = re.findall(r'\$([\w-]+)=\'([\w\-:]+)\'', line)
            if matches:
                record = {}
                for match in matches:
                    key = match[0]
                    value = match[1]
                    record[key] = value
                    
                # Append record to the data frame
                data = pd.concat([data, pd.DataFrame([record])], ignore_index=True)

                # Write the data frame to csv file   
                """ if writer is not None:
                    if csvfile.tell() == 0:
                        writer.writerow(record.keys())
                    writer.writerow(record.values())`
    """
    return data

df = read_log_file(input_file, data)
print(df.info())
# # Introducing multi processing
# import re
# import csv
# import pandas as pd
# from concurrent.futures import ThreadPoolExecutor, as_completed
# import time
# import threading

# # Create an empty DataFrame
# data = pd.DataFrame()

# # Function to process each line and extract key-value pairs
# def process_line(line):
#     matches = re.findall(r"\$(\w+)='([^']*)'", line)
#     if matches:
#         record = {}
#         for match in matches:
#             key = match[0]
#             value = match[1]
#             record[key] = value
#         return record

# # Open the input file and create the output CSV file
# start_time = time.time()

# input_file = '/Users/anas/Documents/UoR/MSc Project/Report/Logs/20230301_anon.log'
# output_file = '/Users/anas/Documents/UoR/MSc Project/Report/Logs/logtxt.csv'

# with open(input_file,  'r') as file, open(output_file, 'w', newline='') as csvfile:
#     # Create a CSV writer
#     writer = csv.writer(csvfile)
    
#     # Create a thread pool executor with 1000 threads
#     with ThreadPoolExecutor(max_workers=50) as executor:
#         # Process each line using threads
#         futures = []
#         lock = threading.Lock()
#         counter = 0
        
#         for line in file:
#             future = executor.submit(process_line, line)
#             future.line = line  # Attach the line to the future for error reporting
            
#             with lock:
#                 counter += 1

#             futures.append(future)
        
#         # Write header to the CSV file
#         header_written = False
        
#         # Iterate over the futures and write to CSV file
#         for future in as_completed(futures):
#             try:
#                 result = future.result()
#                 if result:
#                     # Append the record to the DataFrame
#      #               data = data.append(result, ignore_index=True)
                    
#                     # Write header and values to the CSV file
#                     if writer is not None:
#                         if not header_written:
#                             writer.writerow(result.keys())  # Write header once
#                             header_written = True
#                         writer.writerow(result.values())  # Write values
            
#             except Exception as e:
#                 print(f"An error occurred while processing line: {future.line}")
#                 print(f"Error: {str(e)}")
            
#             with lock:
#                 counter -= 1
#                 if counter == 0:
#                     break

# # Display the DataFrame
# print(data)

# # Calculate execution time
# end_time = time.time()
# execution_time = end_time - start_time
# print(f"Execution time: {execution_time} seconds")