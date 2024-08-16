import os
import argparse

def remove_zero_length_files(directory):
    try:
        # List all files in the directory
        files = os.listdir(directory)
        
        # Initialize a counter for removed files
        removed_files_count = 0
        
        for file_name in files:
            file_path = os.path.join(directory, file_name)
            
            # Check if it's a file and if its size is zero
            if os.path.isfile(file_path) and os.path.getsize(file_path) == 0:
                os.remove(file_path)
                removed_files_count += 1
                print(f"Removed zero-length file: {file_path}")
        
        if removed_files_count == 0:
            print("No zero-length files found.")
        else:
            print(f"Total zero-length files removed: {removed_files_count}")
    
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description="Remove all zero-length files in the specified directory.")
    parser.add_argument('directory', help="Path to the directory to scan for zero-length files")

    # Parse the command-line arguments
    args = parser.parse_args()

    # Call the function with the provided directory path
    remove_zero_length_files(args.directory)

