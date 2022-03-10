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
    pass