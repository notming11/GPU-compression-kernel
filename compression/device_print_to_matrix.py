import re
import sys

def parse_to_matrix(raw_data):
    # Regular expression to extract row, col, and the metadata value
    pattern = re.compile(r"idx\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*metadata:\s*(-?\d+)")
    
    parsed_entries = []
    max_row = 0
    max_col = 0
    
    # Process the data line by line
    i = 0
    for line in raw_data.strip().split('\n'):
        match = pattern.search(line)
        if match:
            row = int(match.group(1))
            col = int(match.group(2))
            val = int(match.group(3))
            
            parsed_entries.append((row, col, val))
            max_row = max(max_row, row)
            max_col = max(max_col, col)
            i += 1

            if i == 64 * 4:
                break
            
    # If no valid data was found, return an empty list
    if not parsed_entries:
        return []

    # Initialize the matrix with zeros based on the largest indices found
    matrix = [[0 for _ in range(max_col + 1)] for _ in range(max_row + 1)]
    
    # Populate the matrix with the parsed metadata values
    for row, col, val in parsed_entries:
        matrix[row][col] = val
        
    return matrix

def print_matrix(matrix):
    if not matrix:
        print("No data found to create a matrix.")
        return
        
    print("[")
    for i, row in enumerate(matrix):
        # Format each number to be right-aligned with a width of 6
        formatted_row = ", ".join(f"{val:6}" for val in row)
        
        # Add a trailing comma for all but the last row
        ending = "," if i < len(matrix) - 1 else ""
        print(f"  [ {formatted_row} ]{ending}")
    print("]")

if __name__ == "__main__":
    # Use the first command line argument as the filename, or default to "data.txt"
    filename = "metadata_64.txt"
    if len(sys.argv) > 1:
        filename = sys.argv[1]
        
    try:
        # Open and read the file
        with open(filename, 'r') as file:
            raw_text = file.read()
            
        # Parse the text and print the matrix
        matrix = parse_to_matrix(raw_text)
        print_matrix(matrix)
        
    except FileNotFoundError:
        print(f"Error: The file '{filename}' was not found. Please ensure it exists in the same directory.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")