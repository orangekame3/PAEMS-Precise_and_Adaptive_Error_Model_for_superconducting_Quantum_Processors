import numpy as np


def compute_correlation_matrix(data, num_qubits, rounds_, shots):
    """
    Computes the correlation matrix from quantum measurement data.
    Vectorized implementation preserving the original PAEMS formula:
        corr[i,j] = (p_xi_xj - xi * xj) / ((1 - 2*xi) * (1 - 2*xj))

    The only change from the original is replacing Python loops with
    BLAS matrix multiplication for p_xi_xj computation.

    Parameters:
    - data : np.array
        The results array from quantum measurements.
    - num_qubits : int
        Total number of qubits.
    - rounds_ : int
        Number of measurement rounds.
    - shots : int
        Total number of measurement shots.

    Returns:
    - np.array
        The computed correlation matrix.
    """
    num_mea = (num_qubits - 1) // 2
    # Reshape the data to (shots, rounds, qubits)
    data1 = data.reshape(shots, rounds_, num_mea)

    # Transpose data to switch 'rounds' and 'qubits' to make qubits sequential per shot
    data2 = data1.transpose(0, 2, 1)

    # Flatten the data back to (shots, qubits * rounds)
    flattened_data = data2.reshape(shots, num_mea * rounds_).astype(np.float64)

    # Compute probabilities
    p_x = np.mean(flattened_data, axis=0)

    # BLAS matrix multiplication replaces:
    #   for k in range(shots):
    #       p_xi_xj += np.outer(flattened_data[k], flattened_data[k])
    #   p_xi_xj /= shots
    p_xi_xj = (flattened_data.T @ flattened_data) / shots

    # Vectorized correlation computation replaces:
    #   for i, j in zip(*valid_indices):
    #       correlation_matrix[i, j] = (xi_xj[i,j] - xi[i]*xj[j]) / ((1-2*xi[i])*(1-2*xj[j]))
    xi = p_x.reshape(-1, 1)
    xj = p_x.reshape(1, -1)

    term1 = (1 - 2 * xi)
    term2 = (1 - 2 * xj)
    denominator = term1 * term2

    with np.errstate(divide="ignore", invalid="ignore"):
        correlation_matrix = (p_xi_xj - xi * xj) / denominator

    # Zero out diagonal and invalid entries
    np.fill_diagonal(correlation_matrix, 0)
    correlation_matrix = np.nan_to_num(correlation_matrix, nan=0.0, posinf=0.0, neginf=0.0)

    # Symmetrize
    correlation_matrix_sim = np.triu(correlation_matrix) + np.triu(correlation_matrix, 1).T

    return correlation_matrix_sim
