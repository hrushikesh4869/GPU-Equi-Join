import csv
import random

def sample_words(input_file, output_file, num_words_to_sample):
    # Read words from the text file
    with open(input_file, 'r') as file:
        words = file.read().splitlines()

    # Sample the specified number of words
    sampled_words = random.sample(words, min(num_words_to_sample, len(words)))

    # Write sampled words to CSV file
    with open(output_file, 'w', newline='') as csvfile:
        fieldnames = ['Index', 'Word']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # Write header
        writer.writeheader()

        # Write sampled words with index to CSV
        for index, word in enumerate(sampled_words):
            writer.writerow({'Index': index + 1, 'Word': word})

# Example usage
input_file_path = 'words_alpha.txt'
output_file_path = 'table2.csv'
num_words_to_sample = 100  # specify the number of words to sample

sample_words(input_file_path, output_file_path, num_words_to_sample)
