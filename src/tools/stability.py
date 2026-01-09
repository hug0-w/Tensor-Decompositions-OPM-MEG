import numpy as np
import matplotlib.pyplot as plt
from tensorly.decomposition import non_negative_parafac
from scipy.optimize import linear_sum_assignment
from itertools import combinations
import tensorly as tl

tl.set_backend('pytorch')


def similarity_matrix(factors1, factors2):
    '''
    Calculates the cosine similarity matrix between components of two runs.

    Parameters:
    factors1 : list of np.ndarray
        List of factor matrices from the first run.
    factors2 : list of np.ndarray
        List of factor matrices from the second run.

    Returns:
    sim_matrix : np.ndarray
        Cosine similarity matrix.

    '''
    rank = factors1[0].shape[1]
    num_modes = len(factors1)

    # Initialize similarity with ones
    sim_matrix = np.ones((rank, rank))

    for mode in range(num_modes):
        mat1 = factors1[mode]
        mat2 = factors2[mode]

        # Normalize columns to unit length for Cosine Similarity
        mat1_norm = mat1 / (np.linalg.norm(mat1, axis=0) + 1e-16)
        mat2_norm = mat2 / (np.linalg.norm(mat2, axis=0) + 1e-16)

        # Calculate cross-correlation for this mode
        mode_sim = np.dot(mat1_norm.T, mat2_norm)

        # Accumulate (Hadamard product)
        sim_matrix *= mode_sim

    return sim_matrix


def optimal_score(similiarity_mat):
    '''
    Computes the optimal matching score using the Hungarian algorithm.

    Parameters:
    similiarity_mat : np.ndarray
        Cosine similarity matrix.

    Returns:
    opt_score : float
        Optimal matching score.

    '''

    # Find cost
    cost = -np.abs(similiarity_mat)

    # Hungarian Algo
    row_ind, col_ind = linear_sum_assignment(cost)

    # Get maximised similarity scores and find average
    matched_scores = similiarity_mat[row_ind, col_ind]
    opt_score = np.abs(matched_scores).mean()

    return opt_score


def rank_stability(tensor_data, rank, mask=None, n_repeats=10, verbose=0):
    '''
    Evaluates the stability of CP decomposition at a given rank.
    
    Parameters:
    tensor_data : np.ndarray
        Input tensor data.
    rank : int
        Rank of the CP decomposition.
    n_repeats: int
        Number of repeats per rank

    Returns:
    mean_stability : float
        Mean stability score.
    std_stability : float
        Standard deviation of the stability score.

    '''
    
    if verbose:
        print(f"--- Testing Rank {rank} with {n_repeats} repeats ---")

    # Identify the device of the input data
    device = tensor_data.device if hasattr(tensor_data, 'device') else 'cpu'

    # Ensure mask is on the same device if provided
    if mask is not None and hasattr(mask, 'to'):
        mask = mask.to(device)


    # Empty list for factors
    run_factors = []

    # Loop over n_repeats
    for i in range(n_repeats):

        cp_tensor = non_negative_parafac(
            tensor_data,
            rank=rank,
            init="random",
            mask=mask,
            n_iter_max=5000,
            tol=1e-9,
            random_state=i  # varies per run
        )

        # Store results
        _, factors = cp_tensor
        factors_np = [tl.to_numpy(f) for f in factors]
        run_factors.append(factors_np)

    # Empty list for scores
    pairwise_scores = []

    # Compute pairwise similarities
    for run_A, run_B in combinations(run_factors, 2):

        sim_mat = similarity_matrix(run_A, run_B)

        score = optimal_score(sim_mat)

        pairwise_scores.append(score)

    # Compute mean and std
    mean_stability = np.mean(pairwise_scores)
    std_stability = np.std(pairwise_scores)

    if verbose:
        print(
            f"Rank {rank}: Mean Stability = {mean_stability:.4f} (Std: {std_stability:.4f})")

    return mean_stability, std_stability

def stability_plot(ranks,stabilities,stds):
    '''
    Plots the stability scores against ranks.
    
    Parameters:
    ranks : list
        List of ranks.
    stabilities : list
        List of stability scores.
    stds : list
        List of standard deviations.
    
    '''
    
    plt.figure(figsize=(10,6))


    plt.errorbar(ranks,stabilities,stds,ls='',marker='o',capsize=4)

    plt.xlabel("Rank",fontsize=14)
    plt.ylabel("Stability (a.u.)",fontsize=14)
    plt.ylim(0,1.2)
    plt.axhline(0.9,0,100,ls='--',color='r')
    plt.text(min(ranks),0.91,'0.9',color='r')
    plt.title("CP Rank Stability (ALS)",fontsize=14)
    plt.grid(alpha=0.5,ls='--')
    plt.minorticks_on()
    plt.xticks(ranks)