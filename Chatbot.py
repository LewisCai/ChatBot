import pandas as pd
import os
import re
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, TrainingArguments, Trainer
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

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

        last_user = None
        last_message = ""
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

# Preparing the data for the model
conversations_df['formatted'] = conversations_df['Input'] + " <SEP> " + conversations_df['Reply']

# Split the dataset
train_texts, val_texts = train_test_split(conversations_df['formatted'], test_size=0.2)

# Load the tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token  # Set padding token

# Tokenize the datasets
train_encodings = tokenizer(train_texts.tolist(), truncation=True, padding=True, max_length=512)
val_encodings = tokenizer(val_texts.tolist(), truncation=True, padding=True, max_length=512)

def add_labels(encodings):
    encodings['labels'] = encodings['input_ids'].copy()
    return encodings

train_encodings = add_labels(train_encodings)
val_encodings = add_labels(val_encodings)

# Update your dataset class to handle labels correctly
class ChatDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        # This method needs to return a dictionary of tensors
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        return item

    def __len__(self):
        return len(self.encodings['input_ids'])

# Create dataset objects
train_dataset = ChatDataset(train_encodings)
val_dataset = ChatDataset(val_encodings)

# Load pre-trained model
model = GPT2LMHeadModel.from_pretrained('gpt2')
model.config.pad_token_id = model.config.eos_token_id  # Set pad_token_id to eos_token_id

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',          # Output directory
    num_train_epochs=3,              # Total number of training epochs
    per_device_train_batch_size=4,   # Batch size per device during training
    per_device_eval_batch_size=4,    # Batch size for evaluation
    warmup_steps=500,                # Number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # Strength of weight decay
    logging_dir='./logs',            # Directory for storing logs
    logging_steps=10,
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

# Train
trainer.train()