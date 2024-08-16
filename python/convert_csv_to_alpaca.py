import argparse
import pandas as pd
import json

def convert_to_alpaca_format(csv_file, output_file):
    """
    Converts a CSV file to a JSON file in AlpacaInstructTemplate format.

    Parameters:
    csv_file (str): Path to the input CSV file.
    output_file (str): Path to the output JSON file.
    """
    # Read CSV file
    df = pd.read_csv(csv_file, delimiter=';')

    # Create a list of dictionaries in the AlpacaInstructTemplate format
    data = []
    for _, row in df.iterrows():
        entry = {
            "instruction": row['Question'],
            "output": row['Response']
        }
        data.append(entry)
    
    # Save the list as a JSON file
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"Data saved to {output_file}")

def main():
    """
    Main function to handle command-line arguments and call the conversion function.
    """
    # Create an argument parser
    parser = argparse.ArgumentParser(
        description="Convert a CSV file with questions and answers to a JSON file in AlpacaInstructTemplate format."
    )
    
    # Add command-line arguments for input and output files
    parser.add_argument('csv_file', type=str, help='Path to the input CSV file.')
    parser.add_argument('--output_file', type=str, default="nobase_alpaca.json", help='Path to the output JSON file.')
    
    # Parse the arguments
    args = parser.parse_args()
    if args.output_file == "nobase_alpaca.json":
        args.output_file = f"{args.csv_file.replace('.csv', '')}_alpaca.json"
    
    # Call the conversion function with the provided arguments
    convert_to_alpaca_format(args.csv_file, args.output_file)

if __name__ == "__main__":
    main()

