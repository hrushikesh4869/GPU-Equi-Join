import csv  
import random

number_of_rows = int(5000000)
# excess_factor = 0.20
number_of_words = int(2*1e5)
factor = int(number_of_rows/number_of_words)

def generate_tables():
    # Read words from the text file
    with open('words_alpha.txt', 'r') as file:
        words = file.read().splitlines()
        
        # Sample the specified number of words
        sampled_words = random.sample(words, min(number_of_words, len(words)))
        
        # Generate a permutation of the sampled words
        sampled_words_table_r = random.sample(factor*sampled_words, number_of_rows)
        sampled_words_table_s = random.sample(factor*sampled_words, number_of_rows)
        
        # Write sampled words to CSV file
        with open('table_r7.csv', 'w', newline='') as csvfile:
            fieldnames = ['Index', 'Word']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            # Write header
            writer.writeheader()
            # Write sampled words with index to CSV
            for index, word in enumerate(sampled_words_table_r):
                writer.writerow({'Index': index + 1, 'Word': word})
        
        with open('table_s7.csv', 'w', newline='') as csvfile:
            fieldnames = ['Index', 'Word']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)     
            # Write header
            writer.writeheader()
            # Write sampled words with index to CSV
            for index, word in enumerate(sampled_words_table_s):
                writer.writerow({'Index': index + 1, 'Word': word})


def generate_worst():
    dummyString = "hello"
    with open('worst_r.csv', 'w', newline='') as csvfile:
        fieldnames = ['Index', 'Word']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        # Write header
        writer.writeheader()
        # Write sampled words with index to CSV
        for index in range(90000):
            writer.writerow({'Index': index + 1, 'Word': dummyString})
    with open('worst_s.csv', 'w', newline='') as csvfile:
        fieldnames = ['Index', 'Word']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        # Write header
        writer.writeheader()
        # Write sampled words with index to CSV
        for index in range(90000):
            writer.writerow({'Index': index + 1, 'Word': dummyString})


if __name__ == "__main__":
    # take input from the user
    # Make this input optional
    # number_of_rows = int(input("Enter number of rows: "))
    # excess_factor = float(input("Enter excess factor: "))
    # number_of_words = number_of_rows*excess_factor
    # generate_tables()
    generate_tables()
