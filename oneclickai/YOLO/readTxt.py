def load_data_from_txt(file_path):
    data = []

    try:
        with open(file_path, 'r') as f:
            for line in f:
                # Strip whitespace and split by spaces
                row_strs = line.strip().split()
                # Convert each element to float
                row_floats = [float(value) for value in row_strs]
                data.append(row_floats)
        return data
    
    # if no file found, return None
    except:
        return None




# Example usage:
if __name__ == '__main__':
    file_path = './imgs/10000_64_4420756.txt'
    data_array = load_data_from_txt(file_path)

    # Print each row
    for row in data_array:
        print(row)