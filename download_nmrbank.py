#!/usr/bin/env python3
"""
Script to download NMRBank dataset file from Hugging Face Hub
Downloads, unzips, and creates a DataFrame from NMRBank_data_with_SMILES_156621_in_225809.zip
"""

import os
import zipfile
import pandas as pd
from huggingface_hub import hf_hub_download, HfApi
from pathlib import Path


def download_and_process_nmrbank():
    """
    Download the NMRBank dataset file from Hugging Face Hub, unzip it, and create a DataFrame
    """
    # Dataset repository information
    repo_id = "sweetssweets/NMRBank"
    filename = "NMRBank/NMRBank_data_with_SMILES_156621_in_225809.zip"

    # Target directory
    target_dir = Path("/Users/dsjes/Desktop/hack/LLMHackathon/NMRBank")
    target_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {filename} from {repo_id}...")
    print("This may take a while depending on your internet connection...")

    try:
        # Download the file
        file_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="dataset",
            local_dir=target_dir,
            local_dir_use_symlinks=False,
        )

        print("✅ Successfully downloaded!")
        print(f"📁 File saved to: {file_path}")

        # Get file size
        file_size = os.path.getsize(file_path) / (1024 * 1024)  # Convert to MB
        print(f"📊 File size: {file_size:.2f} MB")

        # Extract the zip file
        print("📦 Extracting zip file...")
        extract_dir = target_dir / "extracted"
        extract_dir.mkdir(exist_ok=True)

        with zipfile.ZipFile(file_path, "r") as zip_ref:
            zip_ref.extractall(extract_dir)

        print(f"✅ Successfully extracted to: {extract_dir}")

        # Find and load the data file
        print("🔍 Looking for data files...")
        data_files = (
            list(extract_dir.rglob("*.json"))
            + list(extract_dir.rglob("*.csv"))
            + list(extract_dir.rglob("*.tsv"))
        )

        if not data_files:
            print("❌ No data files found in the extracted archive")
            return None

        # Load the first data file found
        data_file = data_files[0]
        print(f"📊 Loading data from: {data_file}")

        # Create DataFrame based on file extension
        if data_file.suffix.lower() == ".json":
            df = pd.read_json(data_file)
        elif data_file.suffix.lower() == ".csv":
            df = pd.read_csv(data_file)
        elif data_file.suffix.lower() == ".tsv":
            df = pd.read_csv(data_file, sep="\t")
        else:
            print(f"❌ Unsupported file format: {data_file.suffix}")
            return None

        print(f"✅ DataFrame created with shape: {df.shape}")
        print(f"📋 Columns: {list(df.columns)}")

        # Save DataFrame to the target directory
        df_path = target_dir / "nmrbank_dataframe.pkl"
        df.to_pickle(df_path)
        print(f"💾 DataFrame saved to: {df_path}")

        # Also save as CSV for easy inspection
        csv_path = target_dir / "nmrbank_dataframe.csv"
        df.to_csv(csv_path, index=False)
        print(f"💾 DataFrame also saved as CSV: {csv_path}")

        # Clean up the downloaded zip file
        os.remove(file_path)
        print("🗑️ Cleaned up downloaded zip file")

        return df_path

    except Exception as e:
        print(f"❌ Error downloading or processing file: {e}")
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
    print("🧪 NMRBank Dataset Downloader & Processor")
    print("=" * 50)

    # List available files first
    list_dataset_files()
    print()

    # Download, unzip, and create DataFrame
    result = download_and_process_nmrbank()

    if result:
        print("\n🎉 Download and processing completed successfully!")
        print(f"You can now use the DataFrame at: {result}")
    else:
        print(
            "\n💥 Download and processing failed. Please check the error messages above."
        )
