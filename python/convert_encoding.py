import argparse

def read_and_write_file(input_file_path, output_file_path):
    try:
        # Open and read the file using 'latin-1' encoding
        with open(input_file_path, 'r', encoding='latin-1') as file:
            content = file.read()
        
        # Open and write to the new file using 'utf-8' encoding
        with open(output_file_path, 'w', encoding='utf-8') as file:
            file.write(content)
        
        print("File read with 'latin-1' and written with 'utf-8' successfully.")
    except UnicodeDecodeError as e:
        print(f"UnicodeDecodeError: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description="Read a file with 'latin-1' encoding and write it with 'utf-8' encoding.")
    parser.add_argument('input_file', help="Path to the input file")
    parser.add_argument('output_file', help="Path to the output file")

    # Parse the command-line arguments
    args = parser.parse_args()

    # Call the function with the provided file paths
    read_and_write_file(args.input_file, args.output_file)

