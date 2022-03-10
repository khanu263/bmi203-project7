# Imports
import numpy as np
from nn import nn, preprocess

def test_forward():

    # Define simple network
    net = nn.NeuralNetwork(nn_arch = [{"input_dim": 3, "output_dim": 2, "activation": "relu"},
                                      {"input_dim": 2, "output_dim": 1, "activation": "relu"}],
                           lr = 0.001,
                           seed = 42,
                           batch_size = 1,
                           epochs = 1,
                           loss_function = "mse")

    # Manually define weights and biases
    net._param_dict = {"W1": np.array([[1, 1, 1], [1, 1, 1]]),
                       "b1": np.array([[1], [1]]),
                       "W2": np.array([[1, 1]]),
                       "b2": np.array([[1]])}
    
    # Put a manual input through the network
    output, cache = net.forward(np.array([1, 1, 1]))

    # Compare everything in the cache to hand calculations
    assert np.array_equal(cache["A0"], np.array([1, 1, 1]))
    assert np.array_equal(cache["A1"], np.array([[4, 4]]))
    assert np.array_equal(cache["Z1"], np.array([[4, 4]]))
    assert np.array_equal(cache["A2"], np.array([[9]]))
    assert np.array_equal(cache["Z2"], np.array([[9]]))
    assert output == 9

def test_single_forward():

    # Define simple network
    net = nn.NeuralNetwork(nn_arch = [{"input_dim": 5, "output_dim": 1, "activation": "relu"}],
                           lr = 0.001,
                           seed = 42,
                           batch_size = 1,
                           epochs = 1,
                           loss_function = "mse")

    # Manually define inputs
    W_curr = np.array([[3, 2, 1, 2, 3]])
    b_curr = np.array([[1]])
    A_prev = np.array([[1, 2, 3, 2, 1]])

    # Push through function
    A_curr, Z_curr = net._single_forward(W_curr, b_curr, A_prev, "relu")

    # Compare output to hand calculations
    assert np.array_equal(A_curr, np.array([[18]]))
    assert np.array_equal(Z_curr, np.array([[18]]))

def test_single_backprop():

    # Define simple network
    net = nn.NeuralNetwork(nn_arch = [{"input_dim": 5, "output_dim": 1, "activation": "relu"}],
                           lr = 0.001,
                           seed = 42,
                           batch_size = 1,
                           epochs = 1,
                           loss_function = "mse")

    # Manually define inputs
    W_curr = np.array([[3, 2, 1, 2, 3]])
    b_curr = np.array([[1]])
    Z_curr = np.array([[20]])
    A_prev = np.array([[1, 2, 3, 2, 1]])
    dA_curr = np.array([[5]])

    # Push through function
    dA_prev, dW_curr, db_curr = net._single_backprop(W_curr, b_curr, Z_curr, A_prev, dA_curr, "relu")

    # Compare output to hand calculations
    assert np.array_equal(dA_prev, np.array([[15, 10, 5, 10, 15]]))
    assert np.array_equal(dW_curr, np.array([[5, 10, 15, 10, 5]]))
    assert np.array_equal(db_curr, np.array([[5]]))

def test_predict():

    # Define simple network
    net = nn.NeuralNetwork(nn_arch = [{"input_dim": 3, "output_dim": 2, "activation": "relu"},
                                      {"input_dim": 2, "output_dim": 1, "activation": "relu"}],
                           lr = 0.001,
                           seed = 42,
                           batch_size = 1,
                           epochs = 1,
                           loss_function = "mse")

    # Manually define weights and biases
    net._param_dict = {"W1": np.array([[1, 1, 1], [1, 1, 1]]),
                       "b1": np.array([[1], [1]]),
                       "W2": np.array([[1, 1]]),
                       "b2": np.array([[1]])}
    
    # Put manual input through the network and make sure it's correct
    y = net.predict(np.array([2, 2, 2]))
    assert y == 15

def test_binary_cross_entropy():

    # Define simple network
    net = nn.NeuralNetwork(nn_arch = [{"input_dim": 5, "output_dim": 1, "activation": "relu"}],
                           lr = 0.001,
                           seed = 42,
                           batch_size = 1,
                           epochs = 1,
                           loss_function = "mse")

    # Manually define inputs
    y = np.array([0, 1, 1, 0])
    y_hat = np.array([0.4, 0.8, 0.9, 0.2])

    # Compute loss and compare to hand calculation
    bce = net._binary_cross_entropy(y, y_hat)
    assert np.isclose(bce, 0.265618)

def test_binary_cross_entropy_backprop():

    # Define simple network
    net = nn.NeuralNetwork(nn_arch = [{"input_dim": 5, "output_dim": 1, "activation": "relu"}],
                           lr = 0.001,
                           seed = 42,
                           batch_size = 1,
                           epochs = 1,
                           loss_function = "mse")

    # Manually define inputs
    y = np.array([0, 1, 1, 0])
    y_hat = np.array([0.4, 0.8, 0.9, 0.2])

    # Compute backprop and compare to hand calculation
    bce_backprop = net._binary_cross_entropy_backprop(y, y_hat)
    assert np.allclose(bce_backprop, np.array([0.416667, -0.3125, -0.277778, 0.3125]))

def test_mean_squared_error():

    # Define simple network
    net = nn.NeuralNetwork(nn_arch = [{"input_dim": 5, "output_dim": 1, "activation": "relu"}],
                           lr = 0.001,
                           seed = 42,
                           batch_size = 1,
                           epochs = 1,
                           loss_function = "mse")

    # Manually define inputs
    y = np.array([0, 1, 1, 0])
    y_hat = np.array([0.4, 0.8, 0.9, 0.2])

    # Compute loss and compare to hand calculation
    mse = net._mean_squared_error(y, y_hat)
    assert np.isclose(mse, 0.0625)

def test_mean_squared_error_backprop():

    # Define simple network
    net = nn.NeuralNetwork(nn_arch = [{"input_dim": 5, "output_dim": 1, "activation": "relu"}],
                           lr = 0.001,
                           seed = 42,
                           batch_size = 1,
                           epochs = 1,
                           loss_function = "mse")

    # Manually define inputs
    y = np.array([0, 1, 1, 0])
    y_hat = np.array([0.4, 0.8, 0.9, 0.2])

    # Compute backprop and compare to hand calculation
    mse_backprop = net._mean_squared_error_backprop(y, y_hat)
    assert np.allclose(mse_backprop, np.array([0.2, -0.1, -0.05, 0.1]))

def test_one_hot_encode():
    
    # Define known test case
    seq_arr = ["ATCG", "GCTA"]
    encoded = np.array([[1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
                        [0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0]])

    # Make sure the encodings match
    assert np.array_equal(preprocess.one_hot_encode_seqs(seq_arr), encoded), "One-hot encodings do not match!"

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