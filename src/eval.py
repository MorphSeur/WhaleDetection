import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from models.models import ConvNN
from tqdm import tqdm
import argparse
from loadData import load_data

# Parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained CNN model on the validation set.")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the model checkpoint.")
    parser.add_argument('--data_dir', type=str, default='/path/to/data', help='Path to the data directory')
    parser.add_argument('--csv_file', type=str, default='/path/to/data', help='Path to the labels file')
    return parser.parse_args()

args = parse_args()

def evaluate_model(model, dataloader, criterion):
    model.eval()  # Set the model to evaluation mode
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc='Validation'):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # Calculate accuracy and return the result
    accuracy = correct / total
    return val_loss / len(dataloader), accuracy

# Load data
_, val_dataloader, _, _, _, _ = load_data(data_dir=args.data_dir, csv_file=args.csv_file)

# Initialize the model
model = ConvNN()

# Load the best trained model checkpoint
checkpoint_path = args.checkpoint_path
checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
model.load_state_dict(checkpoint)

# Define the loss function
criterion = nn.CrossEntropyLoss()

# Evaluate the model on the validation set
val_loss, accuracy = evaluate_model(model, val_dataloader, criterion)

# Print the result
print(f"Validation Loss: {val_loss}, Accuracy: {accuracy}")