# Imports
import numpy as np
from nn import nn, preprocess

def test_forward():
    pass

def test_single_forward():
    pass

def test_single_backprop():
    pass

def test_predict():
    pass

def test_binary_cross_entropy():
    pass

def test_binary_cross_entropy_backprop():
    pass

def test_mean_squared_error():
    pass

def test_mean_squared_error_backprop():
    pass

def test_one_hot_encode():
    
    # Define known test case
    seq_arr = ["ATCG", "GCTA"]
    encoded = np.array([[1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
                        [0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0]])

    # Make sure the encodings match
    assert np.all(np.isclose(preprocess.one_hot_encode_seqs(seq_arr), encoded)), "One-hot encodings do not match!"

def test_sample_seqs():
    
    # Try more positive than negative labels
    seqs = ["1", "2", "3", "4"]
    labels = [True, True, True, False]
    expected_seqs = ["1", "2", "3", "4", "4", "4"]
    expected_labels = [True, True, True, False, False, False]
    sampled_seqs, sampled_labels = preprocess.sample_seqs(seqs, labels)
    assert expected_seqs == sampled_seqs, "Sampled sequences are incorrect (more positive labels)."
    assert expected_labels == sampled_labels, "Sampled labels are incorrect (more positive labels)."
    
    # Try more negative than positive labels
    labels = [True, False, False, False]
    expected_seqs = ["1", "1", "1", "2", "3", "4"]
    sampled_seqs, sampled_labels = preprocess.sample_seqs(seqs, labels)
    assert expected_seqs == sampled_seqs, "Sampled sequences are incorrect (more negative labels)."
    assert expected_labels == sampled_labels, "Sampled labels are incorrect (more negative labels)."