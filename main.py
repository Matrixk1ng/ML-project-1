import chardet
from pathlib import Path
from feature import get_vocab_list, process_folder
import csv
import os

datasets = ["enron1", "enron2", "enron4"]
base_path = "/Users/obinna/Desktop/ML project 1/dataset"
output_path = "/Users/obinna/Desktop/ML project 1/dataset-train"

for dataset in datasets:
    # Set up paths
    spam_train = Path(base_path + "/" + dataset + " - train/train/spam")
    ham_train = Path(base_path + "/" + dataset + " - train/train/ham")
    spam_test = Path(base_path + "/" + dataset + " - test/test/spam")
    ham_test = Path(base_path + "/" + dataset + " - test/test/ham")
    
    # Create output folder for this dataset
    dataset_output = output_path + "/" + dataset
    if not os.path.exists(dataset_output):
        os.makedirs(dataset_output)
    
    # Build vocabulary from training data (spam + ham)
    text = ""
    for file in spam_train.glob("*.txt"):
        raw = file.read_bytes()
        encoding = chardet.detect(raw)["encoding"]
        text += raw.decode(encoding or "utf-8", errors="ignore")
    
    for file in ham_train.glob("*.txt"):
        raw = file.read_bytes()
        encoding = chardet.detect(raw)["encoding"]
        text += raw.decode(encoding or "utf-8", errors="ignore")
    
    vocabulary = get_vocab_list(text)
    vocab_list = sorted(list(vocabulary))
    
    vocab_index = {}
    for i in range(len(vocab_list)):
        vocab_index[vocab_list[i]] = i
    
    # Process training data
    spam_bow, spam_ber = process_folder(spam_train, vocab_list, vocab_index, 1)
    ham_bow, ham_ber = process_folder(ham_train, vocab_list, vocab_index, 0)
    matrix_bow_train = spam_bow + ham_bow
    matrix_ber_train = spam_ber + ham_ber
    
    # Process test data (same vocabulary)
    spam_bow, spam_ber = process_folder(spam_test, vocab_list, vocab_index, 1)
    ham_bow, ham_ber = process_folder(ham_test, vocab_list, vocab_index, 0)
    matrix_bow_test = spam_bow + ham_bow
    matrix_ber_test = spam_ber + ham_ber
    
    # Write CSVs to dataset folder
    with open(dataset_output + "/" + dataset + "_bow_train.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(vocab_list + ["label"])
        writer.writerows(matrix_bow_train)
    
    with open(dataset_output + "/" + dataset + "_bow_test.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(vocab_list + ["label"])
        writer.writerows(matrix_bow_test)
    
    with open(dataset_output + "/" + dataset + "_bernoulli_train.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(vocab_list + ["label"])
        writer.writerows(matrix_ber_train)
    
    with open(dataset_output + "/" + dataset + "_bernoulli_test.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(vocab_list + ["label"])
        writer.writerows(matrix_ber_test)
    
    print(dataset + " done - vocab size: " + str(len(vocab_list)))