import pandas as pd
import numpy as np
import threading

def read_dataframe(file_path):
    # Read the data frame from the file
    df = pd.read_csv(file_path)  # Modify this according to your file format
    return df


def combine_dataframes(df_list):
    # Combine the data frames into a single data frame
    combined_df = pd.concat(df_list, ignore_index=True)  # Modify this according to your combination logic
    
    return combined_df


def func1(df):
    # Perform operations on the combined data frame
    # ...
    pass


def func2(df):
    # Perform operations on the combined data frame
    # ...
    pass

def return_df(df):
    return df

df3 = pd.DataFrame(np.random.randint(0,100, size = (1000,108)))
# Create the first dataframe
df1 = pd.DataFrame(np.random.randint(0, 100, size=(1000, 108)))

# Create the second dataframe
df2 = pd.DataFrame(np.random.randint(0, 100, size=(1000, 108)))

df4 = pd.DataFrame(np.random.randint(0,100,size = (1000,108)))



# List to store the individual data frames
df_list = []

# List to store combined data frames
combined_df = []

# List to store the individual data frames
data_frames = [df1,df3,df4,df4]

# Create threads for reading the data frames
threads = []
for data_frame in data_frames:
    thread = threading.Thread(target=lambda path: df_list.append(return_df(data_frame)), args=(data_frame,))
    threads.append(thread)
    thread.start()

# Wait for all threads to finish
for thread in threads:
    thread.join()

# Create a thread for combining the data frames
combine_thread = threading.Thread(target=lambda: combined_df.append(combine_dataframes(df_list)))
combine_thread.start()
combine_thread.join()

print(df_list)
print(combined_df)
# Create threads for calling different functions on the combined data frame
func1_thread = threading.Thread(target=lambda: func1(combined_df))
func2_thread = threading.Thread(target=lambda: func2(combined_df))

# Start the function threads
func1_thread.start()
func2_thread.start()

# Wait for the function threads to finish
func1_thread.join()
func2_thread.join()

# The functions func1 and func2 have been called on the combined data frame
