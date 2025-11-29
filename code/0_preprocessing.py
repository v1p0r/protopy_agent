import os
import argparse
import sys

def main(args):
    input_latex_path = args.input_latex_path
    output_json_path = args.output_json_path

    # read the latex files
    latex_files = os.listdir(input_latex_path)
    


if __name__ == "__main__":

    # 0 argument message and exit
    if  "--input_latex_path" not in sys.argv and "--output_json_path" not in sys.argv:
        print("Set input_latex_path and output_json_path or pass CLI flags.")
        sys.exit(1)

    # parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_latex_path", type=str)
    parser.add_argument("--output_json_path", type=str)

    
    args = parser.parse_args()
    main(args)