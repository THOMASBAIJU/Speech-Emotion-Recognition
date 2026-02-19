import os
import requests
import zipfile

# URL for RAVDESS (Example URL - Users often need to download manually from Zenodo or Kaggle due to size/auth)
# For this script, we'll provide instructions and a folder check.

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')

def create_data_dir():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"Created data directory at {DATA_DIR}")
    else:
        print(f"Data directory exists at {DATA_DIR}")

def main():
    create_data_dir()
    print("Please download the RAVDESS dataset and extract it into the 'data' folder.")
    print("Structure should look like: data/Actor_01/03-01-01-01-01-01-01.wav, etc.")
    print("You can find the dataset at: https://zenodo.org/record/1188976")

if __name__ == "__main__":
    main()
