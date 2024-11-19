"""Convert all .qmd to .py files in a given directory.

Written with the help of Claude 3.5 Sonnet.
"""

import argparse
import subprocess
import os
import glob

def convert_qmd_to_py(directory):
    # Find all .qmd files in the directory
    qmd_files = glob.glob(os.path.join(directory, "*.qmd"))
    
    if not qmd_files:
        print(f"No .qmd files found in {directory}")
        return
    
    for qmd_file in qmd_files:
        base_name = os.path.splitext(qmd_file)[0]
        ipynb_file = f"{base_name}.ipynb"
        
        try:
            # Step 1: Convert .qmd to .ipynb using quarto
            print(f"Converting {qmd_file} to {ipynb_file}...")
            subprocess.run(["quarto", "convert", qmd_file], check=True)
            
            # Step 2: Convert .ipynb to .py using nbconvert
            print(f"Converting {ipynb_file} to .py...")
            subprocess.run(["jupyter", "nbconvert", "--to", "python", ipynb_file], check=True)
            
            # Step 3: Remove the intermediate .ipynb file
            os.remove(ipynb_file)
            print(f"Successfully converted {qmd_file} to {base_name}.py")
            
        except subprocess.CalledProcessError as e:
            print(f"Error converting {qmd_file}: {e}")
        except Exception as e:
            print(f"Unexpected error with {qmd_file}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Convert .qmd files to .py files")
    parser.add_argument("directory", help="Directory containing .qmd files")
    args = parser.parse_args()
    
    if not os.path.isdir(args.directory):
        print(f"Error: {args.directory} is not a valid directory")
        return
    
    convert_qmd_to_py(args.directory)

if __name__ == "__main__":
    main()