'''
This is a helper script to collect all file paths from a source directory.
The list of files is later used by the data pipeline to extract features 
from the CSV files.

USAGE:
python script.py --source_dir ../path/to/your/data
'''

import os
import argparse

parser = argparse.ArgumentParser(description='Collect file paths from a source directory.')
parser.add_argument('source_dir', type=str, help='Path to the source directory')
args = parser.parse_args()

destination_file = 'data/data_source_files.txt'

source_files=[]
# Process all CSV files in the directory
for dirpath, dirnames, filenames in os.walk(args.source_dir):
    for filename in filenames:
        source_files.append(os.path.join(dirpath, filename))

with open(destination_file, "a") as file:
    # Join the list elements into a single string with newline separators
    file.write('\n'.join(source_files))
