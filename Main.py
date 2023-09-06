import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
import glob
import EDA as eda
import ReadFiles as rf
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense


def drop_columns(df):

    # Dropping columns that are not essenstial to the request log
    columns_to_drop = ['$obstype', 
                       '$bytes_online', '$disk_files', '$fields_online', '$obsgroup', '$reportype', '$disp', 
                       '$fieldset', '$origin', '$hdate', '$method', '$odbs', '$area', '$reports', '$intgrid',
                       '$schedule', '$process', '$padding', '$wrid', '$multitarget', '$dataset', '$filter',
                       '$truncation', '$accuracy', '$frequency', '$direction', '$model', '$gaussian', '$use',
                       '$fcmonth', '$channel', '$iteration', '$range', '$rotation', '$quantile', '$interpolation',
                       '$format', '$packing', '$lsm', '$ident', '$instrument', '$frame', '$wait', '$decache',
                       '$diagnostic', '$decade', '$product', '$section', '$style', '$refdate', '$fcperiod',
                       '$branch', '$ppengine', '$latitude', '$longitude', '$location', '$country', '$bcmodel',
                       '$icmodel', '$grib', '$token' ]
    
    df = df.drop(columns = columns_to_drop)

    return df

def EDA(df):

    # Show grid for missing values
    #eda.check_missingvalues(df)
    
    # Show table for missing values
    #eda.check_null_values(df)

    # Dropping column with missing values
    df = drop_columns(df)
    
#    eda.check_duplicates(df)
 #   eda.queries_by_date(df)
 #   eda.queries_by_verb(df)
    
#    eda.check_null_values(df)
#    eda.check_unique_values(df)

    #outliers(df)
    return

def read_data_tocsv():
    
    csv_files1 = glob.glob('/Users/anas/Documents/UoR/MSc Project/Data/202301/*.log')  # Replace 'path_to_folder' with the actual folder path
    csv_files2 = glob.glob('/Users/anas/Documents/UoR/MSc Project/Data/202302/*.log')  # Replace 'path_to_folder' with the actual folder path
    csv_files3 = glob.glob('/Users/anas/Documents/UoR/MSc Project/Data/202303/*.log')  # Replace 'path_to_folder' with the actual folder path

    dataframes1 = [] 
    for file in csv_files3:
        dataframe1 = rf.process_data_to_df(file)
        dataframes1.append(dataframe1)

    combined_df1 = pd.concat(dataframes1)
    combined_df1.to_csv("/Users/anas/Documents/UoR/MSc Project/Report/Logs/FeatureEngineeringv3.csv")

def read_samples_fromcsv():

    df1 = pd.read_csv("/Users/anas/Documents/UoR/MSc Project/Report/Logs/FeatureEngineering1.csv")
    df2 = pd.read_csv("/Users/anas/Documents/UoR/MSc Project/Report/Logs/FeatureEngineering2.csv")
    df3 = pd.read_csv("/Users/anas/Documents/UoR/MSc Project/Report/Logs/FeatureEngineering3.csv")
    
    combined_df_all = pd.concat([df1, df2, df3], axis=0)

    return combined_df_all

# Define a function to plot training history
def plot_training_history_v2(epochs, training_loss, training_accuracy, validation_loss, validation_accuracy):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs + 1), training_loss, label='Training Loss')
    plt.plot(range(1, epochs + 1), validation_loss, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')

    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs + 1), training_accuracy, label='Training Accuracy')
    plt.plot(range(1, epochs + 1), validation_accuracy, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')

    plt.show()


def plot_training_history(epochs, training_loss, training_accuracy):
    plt.figure(figsize=(12, 6))

    # Plot training loss
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs + 1), training_loss, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss')

    # Plot training accuracy
    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs + 1), training_accuracy, label='Training Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy')

    plt.tight_layout()
    plt.show()

def ml_algorithm_tf_v3(file_path):

    sample_size = int(input("Enter logs sample size: "))
    token_size = int(input("Enter number of tokens for tokenization: "))
    sequence_length = int(input("Enter the length of sequence: "))
    batch_size = int(input("Enter batch size: "))
    dropout_rate = float(input("Enter dropout rate: "))
    activation_mode = str(input("Choose from sigmoid, relu, tanh: "))
    Number_of_epochs = int(input("Number of epochs: "))

    drop_keys = ['$obstype', '$bytes_online', '$disk_files', '$fields_online', '$obsgroup', '$reportype', '$disp', 
                 '$fieldset', '$origin', '$hdate', '$method', '$odbs', '$area', '$reports', '$intgrid',
                 '$schedule', '$process', '$padding', '$wrid', '$multitarget', '$dataset', '$filter',
                 '$truncation', '$accuracy', '$frequency', '$direction', '$model', '$gaussian', '$use',
                 '$fcmonth', '$channel', '$iteration', '$range', '$rotation', '$quantile', '$interpolation',
                 '$format', '$packing', '$lsm', '$ident', '$instrument', '$frame', '$wait', '$decache',
                 '$diagnostic', '$decade', '$product', '$section', '$style', '$refdate', '$fcperiod',
                 '$branch', '$ppengine', '$latitude', '$longitude', '$location', '$country', '$bcmodel',
                 '$icmodel', '$grib', '$token' ]

    data = rf.read_sequentialdata(file_path, sample_size, drop_keys)
    
    # Preprocess the data
    tokenizer = Tokenizer(token_size)
    tokenizer.fit_on_texts([data])
    sequences = tokenizer.texts_to_sequences([data])
    vocab_size = len(tokenizer.word_index) + 1

    # Generate input and output sequences
    input_sequences = []
    output_sequences = []

    for i in range(0, len(sequences[0]) - sequence_length):
        input_sequences.append(sequences[0][i:i+sequence_length])
        output_sequences.append(sequences[0][i+sequence_length])

    # Pad sequences to ensure they have the same length
    input_sequences = pad_sequences(input_sequences, maxlen=sequence_length)
    output_sequences = np.array(output_sequences)

    # Split the data into training, validation, and testing sets
    train_size = int(0.6 * len(input_sequences))
    val_size = int(0.2 * len(input_sequences))
    test_size = len(input_sequences) - train_size - val_size

    x_train = input_sequences[:train_size]
    y_train = output_sequences[:train_size]

    x_val = input_sequences[train_size:train_size+val_size]
    y_val = output_sequences[train_size:train_size+val_size]

    x_test = input_sequences[train_size+val_size:]
    y_test = output_sequences[train_size+val_size:]

    # Build the LSTM model
    model = Sequential()
    model.add(Embedding(vocab_size, 128))
    model.add(LSTM(128, dropout=dropout_rate, recurrent_dropout=0.2))
    model.add(Dense(vocab_size, activation=activation_mode))

    # Compile the model
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Initialize lists to store training history
    training_loss = []
    training_accuracy = []
    validation_loss = []
    validation_accuracy = []

    # Measure ML Performance time
    start = time.time()

    # Train the model and collect training history
    for epoch in range(Number_of_epochs):
        history = model.fit(x_train, y_train, epochs=1, batch_size=batch_size, validation_data=(x_val, y_val), verbose=1)
        training_loss.append(history.history['loss'][0])
        training_accuracy.append(history.history['accuracy'][0])
        validation_loss.append(history.history['val_loss'][0])
        validation_accuracy.append(history.history['val_accuracy'][0])

    end = time.time()
    plot_training_history_v2(Number_of_epochs, training_loss, training_accuracy, validation_loss, validation_accuracy)

    # Generate text
    seed_text = "$startdate='20230501'; $starttime='08:08:08'"
    num_words = 100  # Adjust the number of words to generate

    for _ in range(num_words):
        # Tokenize the seed text
        seed_sequence = tokenizer.texts_to_sequences([seed_text])
        padded_sequence = pad_sequences(seed_sequence, maxlen=sequence_length)

        # Predict the next word probabilities
        predicted_probs = model.predict(padded_sequence, verbose=0)[0]

        # Select the word with the highest probability
        predicted_index = np.argmax(predicted_probs)

        # Convert the predicted word index to the actual word
        predicted_word = tokenizer.index_word[predicted_index]

        # Append the predicted word to the seed text
        seed_text += " " + predicted_word

    # Print the generated text
    print(seed_text)
    print("Time for model: ", (end - start), "sec")

    # Evaluate the model on the test data
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")


    return


def ml_algorithm_tf_v2(file_path):

    sample_size = int(input("Enter logs sample size: "))
    token_size = int(input("Enter number of tokens for tokenization: "))
    sequence_length = int(input("Enter the length of sequence: "))
    batch_size = int(input("Enter batch size: "))
    dropout_rate = float(input("Enter dropout rate: "))
    activation_mode = str(input("Choose from sigmoid, relu, tanh: "))
    Number_of_epochs = int(input("Number of epochs: "))


    drop_keys = ['$obstype', '$bytes_online', '$disk_files', '$fields_online', '$obsgroup', '$reportype', '$disp', 
                 '$fieldset', '$origin', '$hdate', '$method', '$odbs', '$area', '$reports', '$intgrid',
                 '$schedule', '$process', '$padding', '$wrid', '$multitarget', '$dataset', '$filter',
                 '$truncation', '$accuracy', '$frequency', '$direction', '$model', '$gaussian', '$use',
                 '$fcmonth', '$channel', '$iteration', '$range', '$rotation', '$quantile', '$interpolation',
                 '$format', '$packing', '$lsm', '$ident', '$instrument', '$frame', '$wait', '$decache',
                 '$diagnostic', '$decade', '$product', '$section', '$style', '$refdate', '$fcperiod',
                 '$branch', '$ppengine', '$latitude', '$longitude', '$location', '$country', '$bcmodel',
                 '$icmodel', '$grib', '$token' ]

    data = rf.read_sample_log(file_path, sample_size, drop_keys)
    

    
    # Preprocess the data
    tokenizer = Tokenizer(token_size)
    tokenizer.fit_on_texts([data])
    sequences = tokenizer.texts_to_sequences([data])
    vocab_size = len(tokenizer.word_index) + 1

    # Generate input and output sequences
    input_sequences = []
    output_sequences = []
    sequence_length = 100  

    for i in range(0, len(sequences[0]) - sequence_length):
        input_sequences.append(sequences[0][i:i+sequence_length])
        output_sequences.append(sequences[0][i+sequence_length])

    # Pad sequences to ensure they have the same length
    input_sequences = pad_sequences(input_sequences, maxlen=sequence_length)
    output_sequences = np.array(output_sequences)

    # Split the data into training and testing sets
    train_size = int(0.8 * len(input_sequences))
    x_train = input_sequences[:train_size]
    y_train = output_sequences[:train_size]
    x_test = input_sequences[train_size:]
    y_test = output_sequences[train_size:]

#    # Build the LSTM model
#     model = Sequential()
#     model.add(Embedding(vocab_size, 100, input_length=sequence_length))
#     model.add(LSTM(100))
#     model.add(Dense(vocab_size, activation='softmax'))

    # Build the LSTM model
    model = Sequential()
    model.add(Embedding(vocab_size, 128))
    model.add(LSTM(128, dropout=dropout_rate, recurrent_dropout=0.2))
    model.add(Dense(vocab_size, activation=activation_mode))

    # Compile the model
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Initialize lists to store training history
    training_loss = []
    training_accuracy = []

    # Measure ML processing time
    start = time.time()
    
    # Train the model and collect training history
    for epoch in range(Number_of_epochs):
        history = model.fit(x_train, y_train, epochs=1, batch_size=batch_size, verbose=1)
        training_loss.append(history.history['loss'][0])
        training_accuracy.append(history.history['accuracy'][0])

    end = time.time()

    plot_training_history(Number_of_epochs, training_loss, training_accuracy)

    # Evaluate the model on the test data
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")

    # Generate text
    #seed_text = result = f"$startdate='{start_date}'; $starttime='{start_time}'"
    seed_text = "$startdate='20230501'; $starttime='08:08:08'"
    num_words = 100  # Adjust the number of words to generate

    for _ in range(num_words):
        # Tokenize the seed text
        seed_sequence = tokenizer.texts_to_sequences([seed_text])
        padded_sequence = pad_sequences(seed_sequence, maxlen=sequence_length)
        
        # Predict the next word probabilities
        predicted_probs = model.predict(padded_sequence, verbose=0)[0]
        
        # Select the word with the highest probability
        predicted_index = np.argmax(predicted_probs)
        
        # Convert the predicted word index to the actual word
        predicted_word = tokenizer.index_word[predicted_index]
        
        # Append the predicted word to the seed text
        seed_text += " " + predicted_word

    
    # Print the generated text
    print(seed_text)
    
    print("Time for model: ",(end-start),"sec")


def ml_algorithm_tf(file_path):

    sample_size = int(input("Enter logs sample size: "))
    token_size = int(input("Enter number of tokens for tokenization: "))
    sequence_length = int(input("Enter the length of sequence: "))
    batch_size = int(input("Enter batch size: "))
    dropout_rate = float(input("Enter dropout rate: "))
    activation_mode = str(input("Choose from sigmoid, relu, tanh: "))
    Number_of_epochs = int(input("Number of epochs: "))


    drop_keys = ['$obstype', '$bytes_online', '$disk_files', '$fields_online', '$obsgroup', '$reportype', '$disp', 
                 '$fieldset', '$origin', '$hdate', '$method', '$odbs', '$area', '$reports', '$intgrid',
                 '$schedule', '$process', '$padding', '$wrid', '$multitarget', '$dataset', '$filter',
                 '$truncation', '$accuracy', '$frequency', '$direction', '$model', '$gaussian', '$use',
                 '$fcmonth', '$channel', '$iteration', '$range', '$rotation', '$quantile', '$interpolation',
                 '$format', '$packing', '$lsm', '$ident', '$instrument', '$frame', '$wait', '$decache',
                 '$diagnostic', '$decade', '$product', '$section', '$style', '$refdate', '$fcperiod',
                 '$branch', '$ppengine', '$latitude', '$longitude', '$location', '$country', '$bcmodel',
                 '$icmodel', '$grib', '$token' ]

    data = rf.read_sample_log(file_path, sample_size, drop_keys)

    
    # Preprocess the data
    tokenizer = Tokenizer(token_size)
    tokenizer.fit_on_texts([data])
    sequences = tokenizer.texts_to_sequences([data])
    vocab_size = len(tokenizer.word_index) + 1

    # Generate input and output sequences
    input_sequences = []
    output_sequences = []
    sequence_length = 100  

    for i in range(0, len(sequences[0]) - sequence_length):
        input_sequences.append(sequences[0][i:i+sequence_length])
        output_sequences.append(sequences[0][i+sequence_length])

    # Pad sequences to ensure they have the same length
    input_sequences = pad_sequences(input_sequences, maxlen=sequence_length)
    output_sequences = np.array(output_sequences)

    # Split the data into training and testing sets
    train_size = int(0.8 * len(input_sequences))
    x_train = input_sequences[:train_size]
    y_train = output_sequences[:train_size]
    x_test = input_sequences[train_size:]
    y_test = output_sequences[train_size:]

#    # Build the LSTM model
#     model = Sequential()
#     model.add(Embedding(vocab_size, 100, input_length=sequence_length))
#     model.add(LSTM(100))
#     model.add(Dense(vocab_size, activation='softmax'))

    # My code
    model = Sequential()
    model.add(Embedding(vocab_size,128))
    model.add(LSTM(128,dropout=dropout_rate,recurrent_dropout=0.2))
    model.add(Dense(vocab_size,activation=activation_mode))

    # Compile and train the model
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Initialize lists to store training history
    training_loss = []
    training_accuracy = []

    # Measure ML Performance time
    start = time.time()

    # Train the model and collect training history
    for epoch in range(Number_of_epochs):
        history = model.fit(x_train, y_train, epochs=1, batch_size=batch_size, verbose=1)
        training_loss.append(history.history['loss'][0])
        training_accuracy.append(history.history['accuracy'][0])

    end = time.time()
    plot_training_history(Number_of_epochs, training_loss, training_accuracy)

    
 #   model.fit(x_train, y_train, epochs=Number_of_epochs, batch_size=batch_size)


    # Generate text
    #seed_text = result = f"$startdate='{start_date}'; $starttime='{start_time}'"
    seed_text = "$startdate='20230501'; $starttime='08:08:08'"
    num_words = 100  # Adjust the number of words to generate

    for _ in range(num_words):
        # Tokenize the seed text
        seed_sequence = tokenizer.texts_to_sequences([seed_text])
        padded_sequence = pad_sequences(seed_sequence, maxlen=sequence_length)
        
        # Predict the next word probabilities
        predicted_probs = model.predict(padded_sequence, verbose=0)[0]
        
        # Select the word with the highest probability
        predicted_index = np.argmax(predicted_probs)
        
        # Convert the predicted word index to the actual word
        predicted_word = tokenizer.index_word[predicted_index]
        
        # Append the predicted word to the seed text
        seed_text += " " + predicted_word

    end = time.time()
    # Print the generated text
    print(seed_text)
    print("Time for model: ",(end-start),"sec")
# Evaluate the model on the test data
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
if __name__ == '__main__':
    
    folder_path1 = '/Users/anas/Documents/UoR/MSc Project/Data/202301/'
    folder_path2 = '/Users/anas/Documents/UoR/MSc Project/Data/202302/'
    folder_path3 = '/Users/anas/Documents/UoR/MSc Project/Data/202303/'

    # Read 10000 records from each file after removing the unncessary keys
    # This is done for 3 folder paths separately because of system limitations
    # Data is stored in 3 separate text files
    # Step 1
    #rf.read_folder(folder_path3)
    
    # Read file instead of folder
    # file_path = '/Users/anas/Documents/UoR/MSc Project/Data/202303/20230301_anon.log'
    # rf.read_file(file_path)

    # The three files produced above are combined as one file
    #rf.merge_data()

    # Convert text_file to csv for data analysis
    #file_path = '/Users/anas/Documents/UoR/MSc Project/Data/final_combined_logs.txt'
    #df = rf.process_data_to_df(file_path)

    # Perform Exploratory Data Analysis
    df = pd.read_csv('/Users/anas/Documents/UoR/MSc Project/Data/sampled_df.csv')
    df = df.drop(df.columns[0], axis=1)
    #EDA(df)

    # Read 500 sample logs for machine learning model
    # and return as list
    file_path_random_combined_logs = '/Users/anas/Documents/UoR/MSc Project/Data/final_combined_logs.txt'
    file_path_sequential_logs = '/Users/anas/Documents/UoR/MSc Project/Data/202303/20230301_anon.log'
    #df = EDA(combined_alldf)
    
    rf.read_total_logs(folder_path3)
    
    #ml_algorithm_tf_v3(file_path_sequential_logs)
    
    