import os
from app.data.natality_loader import download_natality_file

def main():
    # Configuration
    YEAR = 2022  # Most stable recent year often available
    TARGET_DIR = "./data/nchs/natality"
    
    print(f"--- Maternal Health v2: NCHS Data Downloader ---")
    print(f"Target Year: {YEAR}")
    print(f"Destination: {os.path.abspath(TARGET_DIR)}")
    print("-" * 50)
    
    try:
        download_natality_file(YEAR, TARGET_DIR)
        
        print("-" * 50)
        print("Download request sent successfully.")
        print("\nNEXT STEPS:")
        print(f"1. Go to {TARGET_DIR}")
        print(f"2. Unzip the file: 'unzip Nat{YEAR}us.zip'")
        print("3. Ensure the large .txt file is in that directory.")
        print("4. Run the calibration pipeline.")
        
    except Exception as e:
        print(f"Error during download: {e}")
        print("\nFALLBACK (Manual Download):")
        print(f"URL: ftp://ftp.cdc.gov/pub/Health_Statistics/NCHS/Datasets/DVS/natality/Nat{YEAR}us.zip")
        print("Once downloaded, place and unzip in the folder mentioned above.")

if __name__ == "__main__":
    main()
