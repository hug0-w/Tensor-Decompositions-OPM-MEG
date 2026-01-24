import numpy as np
import matplotlib.pyplot as plt
from tensorly.decomposition import parafac  
from scipy.optimize import linear_sum_assignment
from itertools import combinations
from tensorly.cp_tensor import cp_normalize
import tensorly as tl
from tensorly.tenalg import mode_dot
import torch

tl.set_backend('pytorch')
torch.set_default_dtype(torch.float32)


def rel_error(full, cp_tensor):
    '''
    Calculates the relative error between two tensors.

    Parameters:
    full : torch.Tensor
        Original tensor.
    cp_tensor : tuple
        CP tensor (weights, factors).
    '''
    reconstructed = tl.cp_to_tensor(cp_tensor)

    numerator = torch.norm(full - reconstructed)
    denominator = torch.norm(full)

    return numerator / denominator


def similarity_score(cp_tensor_A, cp_tensor_B):
    '''
    Calculate similarity score between two CP decompositions.
    Uses absolute value of factor correlations to handle sign ambiguity
    in standard CP decomposition.
    
    Parameters:
    cp_tensor_A : tuple
        First CP tensor (weights, factors).
    cp_tensor_B : tuple
        Second CP tensor (weights, factors).
    '''

    lambdaA, factorsA = cp_normalize(cp_tensor_A)
    lambdaB, factorsB = cp_normalize(cp_tensor_B)

    if torch.is_tensor(lambdaA):
        lambdaA = lambdaA.detach().cpu().numpy().astype(np.float32)
    else:
        lambdaA = np.array(lambdaA, dtype=np.float32)

    if torch.is_tensor(lambdaB):
        lambdaB = lambdaB.detach().cpu().numpy().astype(np.float32)
    else:
        lambdaB = np.array(lambdaB, dtype=np.float32)

    factorsA = [f.detach().cpu().numpy().astype(np.float32) if torch.is_tensor(f) 
                else np.array(f, dtype=np.float32) for f in factorsA]
    factorsB = [f.detach().cpu().numpy().astype(np.float32) if torch.is_tensor(f) 
                else np.array(f, dtype=np.float32) for f in factorsB]
    
    rank = factorsA[0].shape[1]      

    sim_matrix = np.zeros((rank, rank), dtype=np.float32)

    for i in range(rank):
        for j in range(rank):
            # Weight similarity (use absolute values since signs can flip)
            max_lam = max(np.abs(lambdaA[i]), np.abs(lambdaB[j]))
            if max_lam == 0:
                weight_score = np.float32(1.0)
            else:
                weight_score = np.float32(1.0) - (np.abs(np.abs(lambdaA[i]) - np.abs(lambdaB[j])) / max_lam)

            # Factor similarity using absolute cosine similarity
            # This handles sign ambiguity in standard CP
            factor_score = np.float32(1.0)
            for mode in range(len(factorsA)):
                # Normalize vectors
                vec_a = factorsA[mode][:, i]
                vec_b = factorsB[mode][:, j]
                
                norm_a = np.linalg.norm(vec_a)
                norm_b = np.linalg.norm(vec_b)
                
                if norm_a > 0 and norm_b > 0:
                    # Absolute cosine similarity (handles sign flips)
                    cosine_sim = np.abs(np.dot(vec_a, vec_b) / (norm_a * norm_b))
                else:
                    cosine_sim = np.float32(0.0)
                
                factor_score *= cosine_sim

            sim_matrix[i, j] = weight_score * factor_score

    # Hungarian Algorithm for optimal matching
    row_ind, col_ind = linear_sum_assignment(sim_matrix, maximize=True)

    score = sim_matrix[row_ind, col_ind].mean(dtype=np.float32)

    return score


def rank_stability(tensor_data, rank, mask=None, n_repeats=10, verbose=0):
    '''
    Evaluates the stability of CP decomposition at a given rank.

    Parameters:
    tensor_data : torch.Tensor
        Input tensor data.
    rank : int
        Rank of the CP decomposition.
    mask : torch.Tensor, optional
        Mask for missing data.
    n_repeats : int
        Number of repeats per rank.
    verbose : int
        Verbosity level.

    Returns:
    mean_stability : float
        Mean stability score.
    std_stability : float
        Standard deviation of the stability score.
    '''

    if tensor_data.dtype != torch.float32:
        tensor_data = tensor_data.to(torch.float32)

    if verbose:
        print(f"--- Testing Rank {rank} with {n_repeats} repeats (Optimization Stability) ---")

    device = tensor_data.device if hasattr(tensor_data, 'device') else 'cpu'
    if mask is not None and hasattr(mask, 'to'):
        mask = mask.to(device)

    models = []
    errors = []

    for i in range(n_repeats):
        try:
            # Standard CP decomposition (allows negative values)
            cp_tensor = parafac(
                tensor_data,
                rank=rank,
                init="random",
                n_iter_max=2000,
                tol=1e-6,
                random_state=i,
                normalize_factors=True,
                linesearch=True  # Improves convergence
            )

            weights, factors = cp_tensor
            
            norm_weights, norm_factors = cp_normalize((weights, factors))
            cpu_weights = norm_weights.detach().cpu()
            cpu_factors = [f.detach().cpu() for f in norm_factors]
            models.append((cpu_weights, cpu_factors))
     
            # Compute Reconstruction Error
            rec_tensor = tl.cp_to_tensor((weights, factors))

            if mask is not None:
                diff = (tensor_data - rec_tensor) * mask
            else:
                diff = tensor_data - rec_tensor

            error = torch.norm(diff)
            error = error.detach().cpu().numpy().astype(np.float32)
            errors.append(error)

        except Exception as e:
            print(f"Run {i} failed: {e}")

    if not models:
        return 0.0, 0.0

    # Identify Best Model
    best_idx = np.argmin(errors)
    best_model = models[best_idx]

    # Compare all models to the Best Model
    similarities = []
    for i, model in enumerate(models):
        if i == best_idx:
            similarities.append(1.0)
        else:
            score = similarity_score(best_model, model)
            similarities.append(score)

    mean_stability = np.mean(similarities, dtype=np.float32)
    std_stability = np.std(similarities, dtype=np.float32)

    if verbose:
        print(f"Rank {rank}: Best Error={errors[best_idx]:.4f} | Mean Similarity={mean_stability:.4f}")

    return mean_stability, std_stability


def rank_fit(tensor_data, rank, mask=None, n_repeats=5, verbose=0):
    '''
    Evaluates the fit (R² score) of CP decomposition at a given rank.

    Parameters:
    tensor_data : torch.Tensor
        Input tensor data.
    rank : int
        Rank of the CP decomposition.
    mask : torch.Tensor, optional
        Mask for missing data.
    n_repeats : int
        Number of repeats.
    verbose : int
        Verbosity level.

    Returns:
    best_fit : float
        Best R² score across repeats.
    std_fit : float
        Standard deviation of fits.
    '''
    
    if tensor_data.dtype != torch.float32:
        tensor_data = tensor_data.to(torch.float32)
    
    device = tensor_data.device if hasattr(tensor_data, 'device') else 'cpu'
    if mask is not None and hasattr(mask, 'to'):
        mask = mask.to(device)

    # Total sum of squares (variance to explain)
    global_mean = torch.mean(tensor_data)
    sst = torch.sum((tensor_data - global_mean) ** 2)

    fits = []

    for i in range(n_repeats):
        try:
            # Standard CP decomposition
            cp_tensor = parafac(
                tensor_data,
                rank=rank,
                init="random",
                n_iter_max=1000,
                tol=1e-5,
                random_state=i,
                normalize_factors=True,
                linesearch=True
            )

            rec_tensor = tl.cp_to_tensor(cp_tensor)

            # Sum of squared residuals
            if mask is not None:
                ssr = torch.sum(((tensor_data - rec_tensor) * mask) ** 2)
            else:
                ssr = torch.sum((tensor_data - rec_tensor) ** 2)

            r2_score = 1 - (ssr / sst)
            
            fits.append(r2_score.detach().cpu().item())
            
            if verbose:
                print(f"  Run {i}: R² = {fits[-1]:.4f}")
             
        except Exception as e:
            print(f"Run {i} failed: {e}") 
    
    if not fits:
        return 0.0, 0.0
    
    best_fit = np.max(fits).astype(np.float32)
    std_fit = np.std(fits).astype(np.float32)
    
    return best_fit, std_fit


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
                 capsize=4, label='CP Stability Score')

    plt.xlabel("Rank", fontsize=14)
    plt.ylabel("Stability (a.u.)", fontsize=14)
    plt.ylim(0, 1.2)
    plt.axhline(0.9, 0, 100, ls='--', color='r')
    plt.text(min(ranks), 0.91, '0.9', color='r')
    plt.title("CP Rank Stability (Standard ALS)", fontsize=14)
    plt.grid(alpha=0.5, ls='--')
    plt.minorticks_on()
    plt.xticks(ranks)
    plt.legend(framealpha=0, loc='upper right')
    plt.tight_layout()


def fit_plot(ranks, fits, stds):
    '''
    Plots the fit (R²) scores against ranks.

    Parameters:
    ranks : list
        List of ranks.
    fits : list
        List of R² scores.
    stds : list
        List of standard deviations.
    '''

    plt.figure(figsize=(10, 6))

    plt.errorbar(ranks, fits, stds, marker='s', color='green',
                 capsize=4, label='R² Score')

    plt.xlabel("Rank", fontsize=14)
    plt.ylabel("R² (Variance Explained)", fontsize=14)
    plt.ylim(0, 1.05)
    plt.title("CP Decomposition Fit vs Rank", fontsize=14)
    plt.grid(alpha=0.5, ls='--')
    plt.minorticks_on()
    plt.xticks(ranks)
    plt.legend(framealpha=0, loc='lower right')
    plt.tight_layout()


def kron_mat_ten(matrices, X):
    '''
    Applies Kronecker-like transformation to tensor.
    '''
    Y = X
    for mode, M in enumerate(matrices):
        Y = mode_dot(Y, M, mode)
    return Y


def corcondia(tensor_data, rank=1, init='random'):
    '''
    Computes CORCONDIA (Core Consistency Diagnostic) for CP decomposition.
    
    Parameters:
    tensor_data : torch.Tensor
        Input tensor.
    rank : int
        Rank for decomposition.
    init : str
        Initialization method.
    
    Returns:
    cc : float
        CORCONDIA score (100 = perfect trilinear structure).
    '''
    
    # Standard CP decomposition
    cp_tensor = parafac(
        tensor_data,
        rank=rank,
        init=init,
        n_iter_max=5000,
        tol=1e-8,
        normalize_factors=True
    )
    
    _, factors = cp_tensor
    
    A, B, C, D = factors
    
    # SVD of each factor
    UA, SA, VAh = torch.linalg.svd(A, full_matrices=False)
    UB, SB, VBh = torch.linalg.svd(B, full_matrices=False)
    UC, SC, VCh = torch.linalg.svd(C, full_matrices=False)
    UD, SD, VDh = torch.linalg.svd(D, full_matrices=False)
    
    # Pseudo-inverse of singular value matrices
    SaI = torch.linalg.pinv(torch.diag(SA[:rank]))
    SbI = torch.linalg.pinv(torch.diag(SB[:rank]))
    ScI = torch.linalg.pinv(torch.diag(SC[:rank]))
    SdI = torch.linalg.pinv(torch.diag(SD[:rank]))

    # Compute core tensor G
    y = kron_mat_ten([UA[:, :rank].T, UB[:, :rank].T, UC[:, :rank].T, UD[:, :rank].T], tensor_data)
    z = kron_mat_ten([SaI, SbI, ScI, SdI], y)
    G = kron_mat_ten([VAh[:rank, :], VBh[:rank, :], VCh[:rank, :], VDh[:rank, :]], z)
    
    # Ideal superdiagonal core tensor
    C_ideal = torch.zeros((rank, rank, rank, rank), device=tensor_data.device)
    for i in range(rank):
        C_ideal[i, i, i, i] = 1
    
    # Compute CORCONDIA
    diff_sq = torch.sum((G - C_ideal) ** 2)
    cc = 100 * (1 - (diff_sq / rank))
    
    return cc.detach().cpu().item()


def rank_selection(tensor_data, ranks=range(1, 11), n_repeats=10, verbose=1):
    '''
    Run stability and fit analysis across multiple ranks.
    
    Parameters:
    tensor_data : torch.Tensor
        Input tensor.
    ranks : range or list
        Ranks to test.
    n_repeats : int
        Number of repeats per rank.
    verbose : int
        Verbosity level.
    
    Returns:
    results : dict
        Dictionary with stability and fit results.
    '''
    
    stabilities = []
    stab_stds = []
    fits = []
    fit_stds = []
    
    for rank in ranks:
        if verbose:
            print(f"\n=== Rank {rank} ===")
        
        stab, stab_std = rank_stability(tensor_data, rank, n_repeats=n_repeats, verbose=verbose)
        stabilities.append(stab)
        stab_stds.append(stab_std)
        
        fit, fit_std = rank_fit(tensor_data, rank, n_repeats=n_repeats, verbose=verbose)
        fits.append(fit)
        fit_stds.append(fit_std)
        
        if verbose:
            print(f"Rank {rank}: Stability={stab:.3f}±{stab_std:.3f}, R²={fit:.3f}±{fit_std:.3f}")
    
    results = {
        'ranks': list(ranks),
        'stabilities': stabilities,
        'stab_stds': stab_stds,
        'fits': fits,
        'fit_stds': fit_stds
    }
    
    return results