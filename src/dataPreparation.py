import glob
import librosa
import matplotlib.pyplot as plt
import numpy as np
import os
import soundfile as sf
import zipfile
from tqdm import tqdm
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Convert audio files to spectrogram images')
    parser.add_argument('--output_path', type=str, default='/home/kmoulouel/environments/ai-tech-test', help='Path to the output directory')
    parser.add_argument('--download_path', type=str, default='/home/kmoulouel/environments/ai-tech-test/zipFiles', help='Path to the download directory')
    return parser.parse_args()

args = parse_args()

# Define paths
output_path = args.output_path
download_path = args.download_path

# Download data (if needed)
if not os.path.exists(os.path.join(output_path, 'train')):
    os.system(f"kaggle competitions download -c whale-detection-challenge -p {download_path}")
    os.system(f"unzip {os.path.join(download_path, 'whale-detection-challenge.zip')} -d {download_path}")

whale = glob.glob(os.path.join(download_path, 'whale_data.zip'))

def get_images(samples, sr, output_path):
    n_fft = 200
    hop_length = 40
    n_mels = 50
    S = librosa.feature.melspectrogram(y=samples,
                                       sr=sr,
                                       n_fft=n_fft,
                                       hop_length=hop_length,
                                       n_mels=n_mels,
                                       fmax=500)
    S_db = librosa.power_to_db(S, ref=np.max)
    librosa.display.specshow(S_db,
                             x_axis='time',
                             y_axis='linear',
                             sr=sr,
                             hop_length=hop_length, 
                             )
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, format='png' , bbox_inches='tight', pad_inches=0)
    plt.close()

for zippath in whale:
    with zipfile.ZipFile(zippath,'r') as zip:
        zip.extractall(output_path)
        train_files = glob.glob(os.path.join(output_path, 'data', 'train', 'train*.aiff'))

for input_file in tqdm(train_files, desc='Processing files'):
    # Load the audio file
    audio_data, sr = sf.read(input_file)

    # Generate the spectrogram and save it with a unique name in the output directory
    output_folder = os.path.join(output_path, 'data', 'spectrogram')
    os.makedirs(output_folder, exist_ok=True)  # Create the folder if it doesn't exist
    output_file = os.path.join(output_folder, os.path.splitext(os.path.basename(input_file))[0] + '.png')

    get_images(audio_data, sr, output_file)