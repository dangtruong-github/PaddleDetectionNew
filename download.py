

import gdown
import os

def download_google_drive_folder(folder_url, save_path):
    """
    Downloads a Google Drive folder and saves it to a specified directory.

    Args:
        folder_url (str): The public URL of the Google Drive folder.
        save_path (str): The local directory where the folder will be saved.
    """
    # Ensure the save_path directory exists
    os.makedirs(save_path, exist_ok=True)
    
    # Change to the specified directory
    os.chdir(save_path)
    
    # Download the folder
    gdown.download_folder(folder_url, quiet=False)
    print(f"Folder downloaded and saved to {save_path}")

# Replace with your folder URL and desired save path
folder_url = "https://drive.google.com/drive/folders/1ll2i1MBrXRTWcepD5mBq-HOv6gB4ZtRP"
save_path = "/N/slate/tnn3/TruongChu/PaddleDetection/dataset/naver/augmented"  # Replace with your desired path

download_google_drive_folder(folder_url, save_path)
