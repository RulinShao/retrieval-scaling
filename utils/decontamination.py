import pdb


def check_below_lexical_overlap_threshold(doc, gold_text, threshold=0.25, mode='longest'):
    """
    Check if the *doc* has no more than *threshold* overlapping with the *gold_text*.
    If yes, return True; else return False.

    *threshold* is set between [0,1] which defines the ratio of tokens that are ok to overlap.
    *mode*: choose from ['longest', 'jaccard']
    """
    if threshold == 1:
        return True
    
    if mode == 'longest':
        doc_words = doc.split(' ')
        gold_text_words = gold_text.split(' ')

        max_overlap = max_contiguous_overlap(doc_words, gold_text_words)
        
        if threshold < 1:
            print(max_overlap, len(gold_text_words) * threshold, max_overlap < int(len(gold_text_words) * threshold))
            return max_overlap < int(len(gold_text_words) * threshold)
        else:
            # when threshold is the word count
            print(max_overlap, threshold, max_overlap < threshold)
            return max_overlap < threshold
    
    elif mode == 'jaccard':
        assert threshold < 1, f"Jaccard similarity decontamination doesn't support word limit. Set threshold within [0, 1]"
        return check_13word_jaccard_similarity(doc, gold_text, threshold)


def max_contiguous_overlap(list1, list2):
    # Function to find the length of the maximum contiguous overlap
    def find_overlap(start1, start2, list1, list2):
        length = 0
        while start1 + length < len(list1) and start2 + length < len(list2) and list1[start1 + length] == list2[start2 + length]:
            length += 1
        return length

    max_overlap = 0
    for i in range(len(list1)):
        for j in range(len(list2)):
            if list1[i] == list2[j]:
                overlap = find_overlap(i, j, list1, list2)
                max_overlap = max(max_overlap, overlap)

    return max_overlap



# N-gram Jaccard similarity overlap
def generate_13word_grams(text):
    # Split text into words
    words = text.split()
    # Generate all possible sequences of 13 words
    return {' '.join(words[i:i+13]) for i in range(len(words) - 12)}

def jaccard_similarity(set1, set2):
    # Calculate Jaccard similarity between two sets
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    return len(intersection) / len(union) if union else 0

def check_13word_jaccard_similarity(text1, text2, threshold=0.8):
    # Generate 13-word grams for each text
    grams1 = generate_13word_grams(text1)
    grams2 = generate_13word_grams(text2)
    
    # Calculate Jaccard similarity
    similarity = jaccard_similarity(grams1, grams2)
    print(similarity)
    
    # Return the similarity if it's 0.8 or less, otherwise indicate high overlap
    if similarity > threshold:
        return False
    else:
        return True


if __name__ == '__main__':
    doc = "checking for matches. When a match is found, it calls find_overlap to determine the length of the contiguous overlap starting from that point in both lists. It keeps track of and returns the list and then each max_contiguous_overlap iterates through each element of checking for matches."
    text = "In this function, max_contiguous_overlap iterates through each element of the first list and then each element of the second list, checking for matches. When a match is found, it calls find_overlap to determine the length of the contiguous overlap starting from that point in both lists. It keeps track of and returns the maximum overlap length found."
    check_below_lexical_overlap_threshold(doc, text, threshold=0.8, mode='jaccard')  # weak decon
    check_below_lexical_overlap_threshold(doc, text, 8, 'longest')  # strong decon
