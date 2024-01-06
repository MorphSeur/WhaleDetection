import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from models.models import ConvNN
import argparse
from loadData import load_data

def parse_args():
    parser = argparse.ArgumentParser(description='Train CNN Model')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--data_dir', type=str, default='/path/to/data', help='Path to the data directory')
    parser.add_argument('--csv_file', type=str, default='/path/to/data', help='Path to the labels file')
    parser.add_argument('--model_path', type=str, default='best_model.pt', help='Path to save the best model')
    parser.add_argument('--weights_path', type=str, default='best_model_weights.pt', help='Path to save the best model weights')
    return parser.parse_args()

args = parse_args()

# Load data
train_dataloader, val_dataloader, _, train_dataset, val_dataset, _ = load_data(data_dir=args.data_dir, csv_file=args.csv_file)

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize your model and move it to GPU
model = ConvNN().to(device)

# Define your loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr)

# Move the dataloaders to GPUj
train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

# Function to train the model
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs):
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        for inputs, labels in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}'):
            inputs, labels = inputs.to(device), labels.to(device)
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Validate the model after each epoch
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            correct = 0
            total = 0
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            avg_val_loss = val_loss / len(val_loader)
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_val_loss}, Accuracy: {correct / total}")

            # Save the model if the current validation loss is the best so far
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model, args.model_path)
                torch.save(model.state_dict(), args.weights_path)

# Train the model
train_model(model, train_dataloader, val_dataloader, criterion, optimizer, num_epochs=args.num_epochs)