"""
Script Name: question_row_compare.py 

Description:
    Reads a row from the specified csv file and compares the contents of two fields. If they are different,
    the entirety of each field is printed, separated by dashed lines. This enables an easier comparison
    of LLM responses of two models (such as before and after finetuning) that were given the same prompt. 

Usage: python question_row_compare.py <csv file> --row <row number> --field_name1 <field1> --field_name2 <field2>

"""

import sys
import argparse
import pandas as pd

def compare_csv_fields(file1, row_number, field_name1, field_name2):

    # Read the CSV file into a DataFrame
    df = pd.read_csv(file1, delimiter=';')
    
    # Check if the row number is within the bounds
    if row_number >= len(df):
        print("Row number exceeds the number of rows in the file.")
        sys.exit(1)
    
    # Check if the field name exists in both DataFrames
    if field_name1 not in df.columns or field_name2 not in df.columns:
        raise ValueError("A field name does not exist.")
    
    # Get the value of the specified field
    Question = df.at[row_number, 'Question']
    print(f"\nQuestion: {Question}")
    field_value1 = df.at[row_number, field_name1]
    field_value2 = df.at[row_number, field_name2]
    
    # Compare the two values
    if field_value1 == field_value2:
        print(f"The values at row {row_number} and field {field_name1} is the same as {field_value2}")
    else:
        print("---------------------------------------------------------------------------------------------------------------")
        print(f"{field_name1}: {field_value1}")
        print("---------------------------------------------------------------------------------------------------------------")
        print(f"{field_name2}: {field_value2}")

def main():
    parser = argparse.ArgumentParser(description="Compare two fields in a specified row of a CSV file.")
    parser.add_argument("file", help="The CSV file to compare fields.")
    parser.add_argument("--row", type=int, default=1, help="The row number to compare (0-indexed).")
    parser.add_argument("--field_name1", default="Response_Llama3_1-8B-Instruct", help="First field/column name to compare.")
    parser.add_argument("--field_name2", default="Response_Llama3_1-8B-Instruct-AMD-python", help="Second field/column name to compare.")
    
    args = parser.parse_args()
    
    compare_csv_fields(args.file, args.row-1, args.field_name1, args.field_name2)

if __name__ == "__main__":
    main()

