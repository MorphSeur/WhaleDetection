import os
import pandas as pd
from sklearn.model_selection import train_test_split
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
import argparse

def replace_file_extension(file_path, new_extension):
    # Split the file path into the base name and the current extension
    base_name, current_extension = os.path.splitext(file_path)
    new_file_path = base_name + new_extension
    return new_file_path

# Define a custom PyTorch dataset
class WhaleDataset(Dataset):
    def __init__(self, dataframe, root_dir, transform=None):
        self.dataframe = dataframe
        self.root_dir = root_dir
        self.transform = transform

    def decode_image(self, image_path):
        try:
            image = Image.open(image_path).convert("RGB")
            image = transforms.ToTensor()(image)
            image = transforms.Resize((256, 256))(image)  # Resize to 256x256
            return image
        except FileNotFoundError:
            # Handle the case where the image file is not found
            print(f"Image file not found: {image_path}")
            return None

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.dataframe.iloc[idx, 0])

        img_name = replace_file_extension(img_name, new_extension=".png")

        image = self.decode_image(img_name)
        if image is None:
            return None, None

        label = int(self.dataframe.iloc[idx, 1])

        if self.transform:
            image = self.transform(image)

        return image, label

def load_data(data_dir, csv_file):
    # Load CSV file into a pandas dataframe
    csv_path = os.path.join(data_dir, csv_file)
    df = pd.read_csv(csv_path, header=None, names=["image", "label"])

    # Split dataset into train, validation, and test sets
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=42)

    # Create custom datasets and dataloaders
    train_dataset = WhaleDataset(train_df, root_dir=os.path.join(data_dir, "./"))
    val_dataset = WhaleDataset(val_df, root_dir=os.path.join(data_dir, "./"))
    test_dataset = WhaleDataset(test_df, root_dir=os.path.join(data_dir, "./"))

    # Filter out None values (images not found) from the datasets
    train_dataset = [entry for entry in tqdm(train_dataset, desc='Loading Train Dataset') if entry[0] is not None]
    val_dataset = [entry for entry in tqdm(val_dataset, desc='Loading Validation Dataset') if entry[0] is not None]
    test_dataset = [entry for entry in tqdm(test_dataset, desc='Loading Test Dataset') if entry[0] is not None]

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    return train_dataloader, val_dataloader, test_dataloader, train_dataset, val_dataset, test_dataset
