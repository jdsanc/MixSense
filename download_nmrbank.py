#!/usr/bin/env python3
"""
Script to download NMRBank dataset file from Hugging Face Hub
Downloads: NMRBank_data_with_SMILES_156621_in_225809.zip
"""

import os
from huggingface_hub import hf_hub_download, HfApi
from pathlib import Path


def download_nmrbank_file():
    """
    Download the NMRBank dataset file from Hugging Face Hub
    """
    # Dataset repository information
    repo_id = "sweetssweets/NMRBank"
    filename = "NMRBank/NMRBank_data_with_SMILES_156621_in_225809.zip"

    # Create output directory if it doesn't exist
    output_dir = Path("NMRBank")
    output_dir.mkdir(exist_ok=True)

    print(f"Downloading {filename} from {repo_id}...")
    print("This may take a while depending on your internet connection...")

    try:
        # Download the file
        file_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="dataset",
            local_dir=output_dir,
            local_dir_use_symlinks=False,
        )

        print(f"✅ Successfully downloaded!")
        print(f"📁 File saved to: {file_path}")

        # Get file size
        file_size = os.path.getsize(file_path) / (1024 * 1024)  # Convert to MB
        print(f"📊 File size: {file_size:.2f} MB")

        return file_path

    except Exception as e:
        print(f"❌ Error downloading file: {e}")
        return None


def list_dataset_files():
    """
    List all available files in the NMRBank dataset
    """
    repo_id = "sweetssweets/NMRBank"

    print(f"📋 Listing files in {repo_id} dataset...")

    try:
        api = HfApi()
        files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")

        print("Available files:")
        for file in files:
            print(f"  - {file}")

    except Exception as e:
        print(f"❌ Error listing files: {e}")


if __name__ == "__main__":
    print("🧪 NMRBank Dataset Downloader")
    print("=" * 40)

    # List available files first
    list_dataset_files()
    print()

    # Download the specific file
    downloaded_file = download_nmrbank_file()

    if downloaded_file:
        print(f"\n🎉 Download completed successfully!")
        print(f"You can now use the file at: {downloaded_file}")
    else:
        print(f"\n💥 Download failed. Please check the error messages above.")
