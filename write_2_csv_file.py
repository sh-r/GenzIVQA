import csv

# Open the CSV file in write mode
def write_csv_file(dict_data, csv_file_path):
    csv_file = csv_file_path

    column_names = ['filename', 'prediction', 'target']

    with open(csv_file, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=column_names)

        # Write the header to the CSV file
        writer.writeheader()

        # Write the data to the CSV file
        for filename, scores in dict_data.items():
            row = {'filename': filename, 'prediction': scores['prediction'], 'target': scores['target']}
            writer.writerow(row)
