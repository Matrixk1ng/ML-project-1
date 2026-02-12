import re



def get_vocab_list(email):
    words = re.findall(r'[a-z]+', email.lower())
    common_words = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "must", "shall", "can", "need",
    "it", "its", "this", "that", "these", "those", "i", "you", "he", "she",
    "we", "they", "what", "which", "who", "whom", "where", "when", "why", "how",
    "all", "each", "every", "both", "few", "more", "most", "other", "some",
    "such", "no", "not", "only", "same", "so", "than", "too", "very", "just",
    "as", "if", "then", "because", "while", "although", "though", "after", "before"
    }
    vocabulary = set()
    """
    matrix -> columns are the words, rows are each email - each entry is how many times those words appear in the email
    """
    for text in words:
        if text not in vocabulary and text not in common_words:
            vocabulary.add(text)
    return vocabulary

def process_folder(folder, vocab_list, vocab_index, label):
    import chardet
    
    matrix_bow = []
    matrix_ber = []
    
    for file in folder.glob("*.txt"):
        raw = file.read_bytes()
        encoding = chardet.detect(raw)["encoding"]
        email = raw.decode(encoding or "utf-8", errors="ignore")
        words = re.findall(r'[a-z]+', email.lower())
        
        row_bow = [0] * len(vocab_list)
        row_ber = [0] * len(vocab_list)
        
        for word in words:
            if word in vocab_index:
                index = vocab_index[word]
                row_bow[index] = row_bow[index] + 1
                row_ber[index] = 1
        
        row_bow.append(label)
        row_ber.append(label)
        matrix_bow.append(row_bow)
        matrix_ber.append(row_ber)
    
    return matrix_bow, matrix_ber
    


