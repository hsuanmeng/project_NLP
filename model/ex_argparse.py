# Include standard modules
import argparse
import pandas as pd

def argsparse():
    # Initiate the parser
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="input csvfile")

    # Read arguments from the command line
    args = parser.parse_args()
    return args

args = argsparse()
# Check for -i or --input
if args.input:
    df = pd.read_csv(args.input)
print(df.head())
