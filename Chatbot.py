import pandas as pd
import os
import re

# Set the directory where your CSV files are located
directory = '/Users/lewis/PersonalProjects/ChatBot'

# Initialize an empty list to hold conversational pairs
conversational_pairs = []

# Regular expression pattern for messages starting with $, _, &, <, or "http"
pattern = re.compile(r'^(\$|_|&|<|http)')

# Function to process each file and extract conversational pairs
def process_file(file_path):
    try:
        # Read the 'Username' and 'Content' columns of the CSV file into a pandas DataFrame
        df = pd.read_csv(file_path, usecols=['Username', 'Content'])

        # Drop rows with NaN values in 'Content'
        df = df.dropna(subset=['Content'])

        # Remove messages that match the pattern or are empty
        df = df[~df['Content'].str.match(pattern) & df['Content'].str.strip().astype(bool)]

        # Track the last user to identify conversation shifts
        last_user = None
        for index, row in df.iterrows():
            if last_user and last_user != row['Username']:
                # Ensure this is not the first message and that the user has changed
                conversational_pairs.append([last_message, row['Content']])
            last_user = row['Username']
            last_message = row['Content']

    except pd.errors.EmptyDataError:
        print(f"No content in {file_path}")
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except Exception as e:
        print(f"An error occurred with {file_path}: {e}")

# Loop through all the files in the directory
for filename in os.listdir(directory):
    if filename.endswith('.csv'):
        # Construct the full file path
        file_path = os.path.join(directory, filename)
        process_file(file_path)

# Convert conversational pairs to a DataFrame
conversations_df = pd.DataFrame(conversational_pairs, columns=['Input', 'Reply'])

# Optional: Save the conversational pairs to a new CSV file
conversations_df.to_csv('/Users/lewis/PersonalProjects/ChatBot/conversational_pairs.csv', index=False)

print("Conversational pairs extraction is completed.")