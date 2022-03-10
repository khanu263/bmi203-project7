import numpy as np
from typing import List, Tuple
from numpy.typing import ArrayLike

def one_hot_encode_seqs(seq_arr: List[str]) -> ArrayLike:
    """
    This function generates a flattened one hot encoding of a list of nucleic acid sequences
    for use as input into a fully connected neural net.

    Args:
        seq_arr: List[str]
            List of sequences to encode.

    Returns:
        encodings: ArrayLike
            Array of encoded sequences, with each encoding 4x as long as the input sequence
            length due to the one hot encoding scheme for nucleic acids.

            For example, if we encode 
                A -> [1, 0, 0, 0]
                T -> [0, 1, 0, 0]
                C -> [0, 0, 1, 0]
                G -> [0, 0, 0, 1]
            Then, AGA -> [1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0]
    """
    
    # Define the encoding dictionary
    encoding = {"A": [1, 0, 0, 0], "T": [0, 1, 0, 0], "C": [0, 0, 1, 0], "G": [0, 0, 0, 1]}

    # Do the encoding for each string
    encoded = []
    for s in seq_arr:
        encoded.append(np.array([encoding[i] for i in s]).flatten())

    # Return an array version of the final list
    return np.array(encoded)

def sample_seqs(seqs: List[str], labels: List[bool]) -> Tuple[List[str], List[bool]]:
    """
    This function should sample your sequences to account for class imbalance. 
    Consider this as a sampling scheme with replacement.
    
    Args:
        seqs: List[str]
            List of all sequences.
        labels: List[bool]
            List of positive/negative labels

    Returns:
        sampled_seqs: List[str]
            List of sampled sequences which reflect a balanced class size
        sampled_labels: List[bool]
            List of labels for the sampled sequences
    """

    # Convert to NumPy arrays for easy manipulation
    seqs = np.array(seqs)
    labels = np.array(labels)
    
    # Split into positive and negative sequences
    positive_seqs = seqs[labels == True]
    negative_seqs = seqs[labels == False]

    # Since they're going to be used a lot, explicitly get the lengths of each array
    len_ps = len(positive_seqs)
    len_ns = len(negative_seqs)

    # Return the originals or oversample from the smaller array based on lengths
    if len_ps == len_ns:
        return list(seqs), list(labels)
    elif len_ps < len_ns:
        oversample_positives = positive_seqs[np.random.choice(len_ps, len_ns, replace = True)]
        sampled_seqs = list(np.concatenate((oversample_positives, negative_seqs), axis = None))
        sampled_labels = [True] * len_ns + [False] * len_ns
    elif len_ps > len_ns:
        oversample_negatives = negative_seqs[np.random.choice(len_ns, len_ps, replace = True)]
        sampled_seqs = list(np.concatenate((positive_seqs, oversample_negatives), axis = None))
        sampled_labels = [True] * len_ps + [False] * len_ps

    # Return final lists
    return sampled_seqs, sampled_labels
