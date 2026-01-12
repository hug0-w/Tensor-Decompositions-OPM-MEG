import numpy as np
import matplotlib.pyplot as plt
from tensorly.decomposition import non_negative_parafac
from scipy.optimize import linear_sum_assignment
from itertools import combinations
import tensorly as tl
import torch

tl.set_backend('pytorch')


def rel_error(full, cp_tensor):
    '''
    Calculates the relative error between two tensors.

    Parameters:
    full : torch.Tensor
        Original tensor.
    reconstructed : torch.
    '''
    reconstructed = tl.cp_to_tensor(cp_tensor)

    numerator = torch.norm(full - reconstructed)
    denominator = torch.norm(full)

    return numerator / denominator


def normalize_cp_tensor_torch(cp_tensor):
    """

    """
    weights, factors = cp_tensor
    rank = weights.shape[0]

    # Clone to avoid modifying original
    weights = weights.clone()
    new_factors = [f.clone() for f in factors]

    for r in range(rank):
        for mode in range(len(new_factors)):
            # PyTorch norm
            norm = torch.norm(new_factors[mode][:, r])

            # Absorb norm into weight
            weights[r] *= norm

            # Normalize the factor vector
            if norm > 0:
                new_factors[mode][:, r] /= norm

    return weights, new_factors


def similarity_sore(cp_tensor_A, cp_tensor_B):
    '''
    Calculate similarity score
       Parameters:
    factorsA : list of np.ndarray
        List of factor matrices from the first run.
    factorsA : list of np.ndarray
        List of factor matrices from the second run.
    '''

    lambdaA, factorsA = cp_tensor_A
    lambdaB, factorsB = cp_tensor_B

    rank = len(lambdaA)
    device = lambdaA.device

    sim_matrix = torch.zeros((rank, rank), device=device)

    mode_dots = []
    for mode in range(len(factorsA)):
        # factors[mode] is (Dim, Rank). Transpose to get (Rank, Dim) @ (Dim, Rank) -> (Rank, Rank)
        dot = torch.matmul(factorsA[mode].T, factorsB[mode])
        mode_dots.append(dot)

    for i in range(rank):
        for j in range(rank):
            # Weight Similarity: (1 - |lam1 - lam2| / max(lam1, lam2))
            max_lam = torch.max(lambdaA[i], lambdaB[j])
            max_lam = torch.max(max_lam, torch.tensor(1e-16, device=device))
            weight_sim = 1 - (torch.abs(lambdaA[i] - lambdaB[j]) / max_lam)

            # Factor Similarity: Product of dot products across all modes
            factor_sim = torch.tensor(1.0, device=device)
            for mode in range(len(factorsA)):
                factor_sim *= mode_dots[mode][i, j]

            sim_matrix[i, j] = weight_sim * factor_sim

    sim_matrix_np = sim_matrix.detach().cpu().numpy()

    # Hungarian Algo
    row_ind, col_ind = linear_sum_assignment(sim_matrix_np, maximize=True)

    score = sim_matrix_np[row_ind, col_ind].mean()

    return score


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
        print(
            f"--- Testing Rank {rank} with {n_repeats} repeats (Optimization Stability) ---")

    device = tensor_data.device if hasattr(tensor_data, 'device') else 'cpu'
    if mask is not None and hasattr(mask, 'to'):
        mask = mask.to(device)

    models = []
    errors = []

    for i in range(n_repeats):

        try:
            cp_tensor = non_negative_parafac(
                tensor_data,
                rank=rank,
                init="random",
                mask=mask,
                n_iter_max=5000,
                tol=1e-7,
                random_state=i  # Ensure different random init
            )

            models.append(cp_tensor)
            rec_tensor = tl.cp_to_tensor(cp_tensor)

            # Reconstruction Error (GPU)
            rec_tensor = tl.cp_to_tensor(cp_tensor)

            if mask is not None:
                diff = (tensor_data - rec_tensor) * mask
            else:
                diff = tensor_data - rec_tensor

            # .item() moves single scalar to CPU
            error = torch.norm(diff).item()
            errors.append(error)

        except Exception as e:
            print(f"Run {i} failed: {e}")

    if not models:
        return 0.0, 0.0

    # Identify Best Model (global minimum candidate)
    best_idx = np.argmin(errors)
    best_model = models[best_idx]

    # Compare all models to the Best Model
    similarities = []
    for i, model in enumerate(models):
        if i == best_idx:
            # The similarity of the best model to itself is 1.0
            similarities.append(1.0)
        else:
            score = similarity_sore(best_model, model)
            similarities.append(score)

    mean_stability = np.mean(similarities)
    std_stability = np.std(similarities)

    if verbose:
        print(
            f"Rank {rank}: Best Error={errors[best_idx]:.4f} | Mean Similarity={mean_stability:.4f}")

    return mean_stability, std_stability


def stability_plot(ranks, stabilities, stds):
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

    plt.figure(figsize=(10, 6))

    plt.errorbar(ranks, stabilities, stds, marker='o',
                 capsize=4, label='ALS Stabilty Score')

    plt.xlabel("Rank", fontsize=14)
    plt.ylabel("Stability (a.u.)", fontsize=14)
    plt.ylim(0, 1.2)
    plt.axhline(0.9, 0, 100, ls='--', color='r')
    plt.text(min(ranks), 0.91, '0.9', color='r')
    plt.title("CP Rank Stability (ALS)", fontsize=14)
    plt.grid(alpha=0.5, ls='--')
    plt.minorticks_on()
    plt.xticks(ranks)
    plt.legend(framealpha=0, loc='upper right')
