#!/usr/bin/env python
import os
import re
import csv
import json
import argparse
from nltk.tokenize import sent_tokenize, word_tokenize
import pandas as pd
from sklearn.model_selection import train_test_split


##############################
# Preprocessing and CSV Generation Functions
##############################

def preprocess_text(text):
    text = text.lower()
    text = text.replace('â', "'")
    text = text.replace('“', '"').replace('”', '"')
    text = text.replace('’', "'").replace('‘', "'")
    text = text.replace('—', '-')
    text = ' '.join(text.split())  # Remove unnecessary whitespace characters
    return text


def remove_header(text):
    header_pattern = r'\*\*\* START OF THE PROJECT GUTENBERG EBOOK .* \*\*\*'
    header_match = re.search(header_pattern, text)
    if header_match:
        start = header_match.end()
        return text[start:]
    else:
        raise ValueError("Header not found in the text file.")


def remove_footer(text):
    footer_pattern = r'\*\*\* END OF THE PROJECT GUTENBERG EBOOK .* \*\*\*'
    footer_match = re.search(footer_pattern, text)
    if footer_match:
        end = footer_match.start()
        return text[:end]
    else:
        raise ValueError("Footer not found in the text file.")


def preprocess_gutenberg_text(text):
    try:
        text = remove_header(text)
    except ValueError as e:
        print(f"Warning: {e}")
    try:
        text = remove_footer(text)
    except ValueError as e:
        print(f"Warning: {e}")
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text


def generate_dataset(book_path, output_file):
    with open(book_path, 'r', encoding='utf-8') as file:
        book_text = file.read()
    book_text = preprocess_gutenberg_text(book_text)
    sentences = sent_tokenize(book_text)
    tokenized_sentences = [word_tokenize(sentence) for sentence in sentences]
    all_words = [word for sentence in tokenized_sentences for word in sentence]
    dataset = []
    index = 0
    # Process in chunks of 200 words: first 100 for question, next 100 for answer.
    while index + 200 <= len(all_words):
        question_words = all_words[index:index + 100]
        answer_words = all_words[index + 100:index + 200]
        question = preprocess_text(' '.join(question_words))
        answer = preprocess_text(' '.join(answer_words))
        dataset.append({
            'question': question,
            'answer': answer
        })
        index += 200
    # Remove the first element as per the original code
    if dataset:
        dataset.pop(0)
    with open(output_file, 'w', encoding='utf-8', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['question', 'answer'])
        writer.writeheader()
        for data in dataset:
            writer.writerow(data)
    print(f"Dataset generated and saved to {output_file}")


def combine_csv_files(csv_files, output_file):
    """Combine CSV files with headers: question, answer (and optionally document)."""
    combined_dataset = []
    for csv_file in csv_files:
        with open(csv_file, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                combined_dataset.append(row)
    with open(output_file, 'w', encoding='utf-8', newline='') as file:
        # Check if the first row has a 'document' field
        fieldnames = ['question', 'answer']
        if combined_dataset and 'document' in combined_dataset[0]:
            fieldnames.append('document')
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for data in combined_dataset:
            writer.writerow(data)
    print(f"Combined dataset saved to {output_file}")


def combine_csv_files_bf(csv_files, output_file):
    """
    For BF mode, combine CSV files and add a new 'document' field that concatenates question and answer.
    """
    combined_dataset = []
    for csv_file in csv_files:
        with open(csv_file, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                row['document'] = row['question'] + ' ' + row['answer']
                combined_dataset.append(row)
    with open(output_file, 'w', encoding='utf-8', newline='') as file:
        fieldnames = ['question', 'answer', 'document']
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for data in combined_dataset:
            writer.writerow(data)
    print(f"BF combined dataset saved to {output_file}")


##############################
# JSON Generation Functions
##############################

def convert_csv_to_json_train_test(csv_file_path, output_dir, time_step_num, test_size=0.2):
    """
    Read a CSV file and split it into train/test sets. Save them as JSON files.
    """
    df = pd.read_csv(csv_file_path)
    # If df is empty or too small, handle gracefully.
    if df.empty:
        print(f"Warning: {csv_file_path} is empty. Skipping JSON train/test conversion.")
        return
    # train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)
    # instead of using train test split. we use all data for training and test. and create a dummy test set with no data

    train_df = df
    test_df = pd.DataFrame(columns=['question', 'answer'])

    train_qa_pairs = train_df.to_dict(orient="records")
    test_qa_pairs = test_df.to_dict(orient="records")
    train_json_file_path = os.path.join(output_dir, f'time_step_{time_step_num}_train_dataset_unlearn.json')
    test_json_file_path = os.path.join(output_dir, f'time_step_{time_step_num}_test_dataset_unlearn.json')
    with open(train_json_file_path, 'w', encoding='utf-8') as train_json_file:
        json.dump(train_qa_pairs, train_json_file, ensure_ascii=False, indent=4)
    with open(test_json_file_path, 'w', encoding='utf-8') as test_json_file:
        json.dump(test_qa_pairs, test_json_file, ensure_ascii=False, indent=4)
    print(f"Train JSON saved to {train_json_file_path}")
    print(f"Test JSON saved to {test_json_file_path}")


def convert_csv_to_json(csv_file_path, output_dir, time_step_num):
    """
    Read a CSV file and convert all its rows into a JSON file.
    """
    df = pd.read_csv(csv_file_path)
    if df.empty:
        print(f"Warning: {csv_file_path} is empty. Skipping JSON conversion.")
        return
    qa_pairs = df.to_dict(orient="records")
    json_file_path = os.path.join(output_dir, f'time_step_{time_step_num}_combined_dataset_unlearn.json')
    with open(json_file_path, 'w', encoding='utf-8') as json_file:
        json.dump(qa_pairs, json_file, ensure_ascii=False, indent=4)
    print(f"Combined JSON saved to {json_file_path}")


def combine_previous_test_jsons(output_dir, current_time_step_num):
    """
    Combine the test JSON files from all previous time steps into a single JSON file.
    """
    combined_test_qa_pairs = []
    for time_step_num in range(1, current_time_step_num):
        previous_json_file_path = os.path.join(output_dir, f'time_step_{time_step_num}',
                                               f'time_step_{time_step_num}_test_dataset_unlearn.json')
        if os.path.exists(previous_json_file_path):
            with open(previous_json_file_path, 'r', encoding='utf-8') as json_file:
                test_qa_pairs = json.load(json_file)
                combined_test_qa_pairs.extend(test_qa_pairs)
        else:
            print(f"Warning: {previous_json_file_path} does not exist and will be skipped.")
    combined_json_file_path = os.path.join(output_dir, f'time_step_{current_time_step_num}',
                                           f'time_step_{current_time_step_num}_combined_previous_tests.json')
    with open(combined_json_file_path, 'w', encoding='utf-8') as combined_json_file:
        json.dump(combined_test_qa_pairs, combined_json_file, ensure_ascii=False, indent=4)
    print(f"Combined previous tests JSON saved to {combined_json_file_path}")


##############################
# Main Processing Function
##############################

def main(input_dir, num_timesteps):
    # Step 1: Process each time_step directory to generate CSVs from text files
    print("Step 1: Generating CSVs from text files...")
    # We assume directories are named like time_step_X inside input_dir.
    for subdir in os.listdir(input_dir):
        subdir_path = os.path.join(input_dir, subdir)
        if os.path.isdir(subdir_path) and re.match(r'time_step_\d+', subdir):
            # Extract time step number (if needed)
            match = re.search(r'time_step_(\d+)', subdir)
            time_step = match.group(1)
            csv_files = []
            for file_name in os.listdir(subdir_path):
                if file_name.endswith('.txt'):
                    book_path = os.path.join(subdir_path, file_name)
                    book_name = os.path.splitext(file_name)[0]
                    output_csv = os.path.join(subdir_path, f'{book_name}_train.csv')
                    try:
                        generate_dataset(book_path, output_csv)
                        csv_files.append(output_csv)
                    except Exception as e:
                        print(f"Error processing {book_path}: {e}")
            if csv_files:
                combined_csv = os.path.join(subdir_path, f'time_step_{time_step}_combined_dataset_unlearn.csv')
                combine_csv_files(csv_files, combined_csv)

    # Step 2: Generate BF CSV files by accumulating previous combined CSVs
    print("\nStep 2: Generating BF CSV files...")
    # Sort time_step directories numerically
    time_step_dirs = sorted([d for d in os.listdir(input_dir) if re.match(r'time_step_\d+', d)],
                            key=lambda x: int(re.search(r'\d+', x).group()))
    previous_combined_files = []
    for idx, time_step_dir in enumerate(time_step_dirs, start=1):
        current_dir = os.path.join(input_dir, time_step_dir)
        combined_unlearn_csv = os.path.join(current_dir, f'time_step_{idx}_combined_dataset_unlearn.csv')
        combined_bf_csv = os.path.join(current_dir, f'time_step_{idx}_combined_dataset_bf.csv')
        if os.path.exists(combined_unlearn_csv):
            previous_combined_files.append(combined_unlearn_csv)
        if previous_combined_files:
            combine_csv_files_bf(previous_combined_files, combined_bf_csv)

    # Step 3: Convert combined CSV to JSON train/test splits (using 20% for test)
    print("\nStep 3: Converting CSV to JSON train/test splits...")
    for idx, time_step_dir in enumerate(time_step_dirs, start=1):
        current_dir = os.path.join(input_dir, time_step_dir)
        combined_unlearn_csv = os.path.join(current_dir, f'time_step_{idx}_combined_dataset_unlearn.csv')
        if os.path.exists(combined_unlearn_csv):
            convert_csv_to_json_train_test(combined_unlearn_csv, current_dir, idx, test_size=0.2)
        else:
            print(f"Warning: {combined_unlearn_csv} does not exist.")

    # Step 4: Convert combined CSV to JSON (complete conversion without splitting)
    print("\nStep 4: Converting CSV to combined JSON...")
    for idx, time_step_dir in enumerate(time_step_dirs, start=1):
        current_dir = os.path.join(input_dir, time_step_dir)
        combined_unlearn_csv = os.path.join(current_dir, f'time_step_{idx}_combined_dataset_unlearn.csv')
        if os.path.exists(combined_unlearn_csv):
            convert_csv_to_json(combined_unlearn_csv, current_dir, idx)
        else:
            print(f"Warning: {combined_unlearn_csv} does not exist.")

    # Step 5: Combine previous test JSON files for each time step (starting from time step 2)
    print("\nStep 5: Combining previous test JSON files...")
    # We assume all time_step directories are directly under input_dir.
    for idx, time_step_dir in enumerate(time_step_dirs, start=1):
        if idx >= 2:  # Only combine if there are previous time steps
            current_dir = os.path.join(input_dir, time_step_dir)
            combine_previous_test_jsons(input_dir, idx)

    print("\nProcessing completed.")


##############################
# Argument Parsing and Entry Point
##############################

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combined preprocessing script for dataset creation.")
    parser.add_argument("--input_dir", type=str, help="Path to the input directory containing time_step subdirectories."
                        , default='data_1000book/10_parts/')
    parser.add_argument("--num_timesteps", type=int, default=10, help="Number of time steps to process (default: 10).")
    args = parser.parse_args()

    main(args.input_dir, args.num_timesteps)
