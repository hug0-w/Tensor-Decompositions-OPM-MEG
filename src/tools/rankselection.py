import numpy as np
import matplotlib.pyplot as plt
from tensorly.decomposition import parafac
from scipy.optimize import linear_sum_assignment
from tensorly.cp_tensor import cp_normalize
import tensorly as tl
from tensorly.tenalg import mode_dot
import torch

tl.set_backend('pytorch')
torch.set_default_dtype(torch.float32)


def rel_error(full, cp_tensor):
    """Calculates the relative error between original and reconstructed tensor."""
    reconstructed = tl.cp_to_tensor(cp_tensor)
    return torch.norm(full - reconstructed) / torch.norm(full)


def r_squared(tensor_data, cp_tensor):
    """Calculates R² (variance explained) for CP decomposition."""
    reconstructed = tl.cp_to_tensor(cp_tensor)
    ss_res = torch.sum((tensor_data - reconstructed) ** 2)
    ss_tot = torch.sum((tensor_data - torch.mean(tensor_data)) ** 2)
    return (1 - (ss_res / ss_tot)).item()


def similarity_score(cp_tensor_A, cp_tensor_B):
    """
    Calculate similarity score between two CP decompositions.
    Uses absolute value of factor correlations to handle sign ambiguity.
    """
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
            # Weight similarity
            max_lam = max(np.abs(lambdaA[i]), np.abs(lambdaB[j]))
            if max_lam == 0:
                weight_score = np.float32(1.0)
            else:
                weight_score = np.float32(1.0) - (np.abs(np.abs(lambdaA[i]) - np.abs(lambdaB[j])) / max_lam)

            # Factor similarity using absolute cosine similarity
            factor_score = np.float32(1.0)
            for mode in range(len(factorsA)):
                vec_a = factorsA[mode][:, i]
                vec_b = factorsB[mode][:, j]
                
                norm_a = np.linalg.norm(vec_a)
                norm_b = np.linalg.norm(vec_b)
                
                if norm_a > 0 and norm_b > 0:
                    cosine_sim = np.abs(np.dot(vec_a, vec_b) / (norm_a * norm_b))
                else:
                    cosine_sim = np.float32(0.0)
                
                factor_score *= cosine_sim

            sim_matrix[i, j] = weight_score * factor_score

    # Hungarian Algorithm for optimal matching
    row_ind, col_ind = linear_sum_assignment(sim_matrix, maximize=True)
    score = sim_matrix[row_ind, col_ind].mean(dtype=np.float32)

    return score


def run_parafac(tensor_data, rank, random_state=None, n_iter_max=2000, tol=1e-8, init='svd'):
    """
    Run CP decomposition.
    
    Parameters:
    tensor_data : torch.Tensor
        Input tensor.
    rank : int
        Rank of decomposition.
    random_state : int or None
        Random seed.
    n_iter_max : int
        Maximum iterations.
    tol : float
        Convergence tolerance.
    init : str
        Initialization method ('random' or 'svd').
    
    Returns:
    cp_tensor : tuple
        (weights, factors)
    """
    cp_tensor = parafac(
        tensor_data,
        rank=rank,
        init=init,
        n_iter_max=n_iter_max,
        tol=tol,
        random_state=random_state,
        normalize_factors=True,
        linesearch=True
    )
    return cp_tensor


def rank_stability(tensor_data, rank, n_repeats=10, verbose=0):
    """
    Evaluates the stability of CP decomposition at a given rank.

    Parameters:
    tensor_data : torch.Tensor
        Input tensor data.
    rank : int
        Rank of the CP decomposition.
    n_repeats : int
        Number of repeats per rank.
    verbose : int
        Verbosity level.

    Returns:
    mean_stability : float
        Mean stability score.
    std_stability : float
        Standard deviation of the stability score.
    """
    if tensor_data.dtype != torch.float32:
        tensor_data = tensor_data.to(torch.float32)

    if verbose:
        print(f"--- Testing Rank {rank} with {n_repeats} repeats ---")

    device = tensor_data.device if hasattr(tensor_data, 'device') else 'cpu'

    models = []
    errors = []

    for i in range(n_repeats):
        try:
            cp_tensor = run_parafac(tensor_data, rank=rank, random_state=i)

            weights, factors = cp_tensor
            norm_weights, norm_factors = cp_normalize((weights, factors))
            cpu_weights = norm_weights.detach().cpu() if hasattr(norm_weights, 'cpu') else norm_weights
            cpu_factors = [f.detach().cpu() if hasattr(f, 'cpu') else f for f in norm_factors]
            models.append((cpu_weights, cpu_factors))
     
            error = rel_error(tensor_data, cp_tensor)
            error = error.detach().cpu().numpy().astype(np.float32)
            errors.append(error)

        except Exception as e:
            if verbose:
                print(f"Run {i} failed: {e}")

    if not models:
        return 0.0, 0.0

    # Identify best model
    best_idx = np.argmin(errors)
    best_model = models[best_idx]

    # Compare all models to the best model
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
        print(f"Rank {rank}: Best Error={errors[best_idx]:.4f} | Mean Stability={mean_stability:.4f}")

    return mean_stability, std_stability


def rank_r2(tensor_data, rank, n_repeats=5, verbose=0):
    """
    Evaluates the R² (variance explained) of CP decomposition at a given rank.

    Parameters:
    tensor_data : torch.Tensor
        Input tensor data.
    rank : int
        Rank of the CP decomposition.
    n_repeats : int
        Number of repeats.
    verbose : int
        Verbosity level.

    Returns:
    best_r2 : float
        Best R² score across repeats.
    std_r2 : float
        Standard deviation of R² scores.
    """
    if tensor_data.dtype != torch.float32:
        tensor_data = tensor_data.to(torch.float32)

    r2_scores = []

    for i in range(n_repeats):
        try:
            cp_tensor = run_parafac(tensor_data, rank=rank, random_state=i)
            r2 = r_squared(tensor_data, cp_tensor)
            r2_scores.append(r2)
            
            if verbose > 1:
                print(f"  Run {i}: R² = {r2:.4f}")
             
        except Exception as e:
            if verbose:
                print(f"Run {i} failed: {e}")
    
    if not r2_scores:
        return 0.0, 0.0
    
    best_r2 = float(np.max(r2_scores))
    std_r2 = float(np.std(r2_scores))
    
    return best_r2, std_r2



def kron_mat_ten(matrices, X):
    """Applies Kronecker-like transformation to tensor."""
    Y = X
    for mode, M in enumerate(matrices):
        Y = mode_dot(Y, M, mode)
    return Y



def rank_selection(tensor_data, ranks=range(1, 11), n_repeats=10, verbose=1):
    """
    Run stability and R² analysis across multiple ranks.
    
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
        Dictionary with stability and R² results.
    """
    stabilities, stab_stds = [], []
    r2_scores, r2_stds = [], []
    
    for rank in ranks:
        if verbose:
            print(f"\n=== Rank {rank} ===")
        
        stab, stab_std = rank_stability(tensor_data, rank, n_repeats=n_repeats, verbose=verbose)
        stabilities.append(stab)
        stab_stds.append(stab_std)
        
        r2, r2_std = rank_r2(tensor_data, rank, n_repeats=n_repeats, verbose=verbose)
        r2_scores.append(r2)
        r2_stds.append(r2_std)
        
        if verbose:
            print(f"Rank {rank}: Stability={stab:.3f}±{stab_std:.3f}, R²={r2:.3f}±{r2_std:.3f}")
    
    return {
        'ranks': list(ranks),
        'stabilities': stabilities,
        'stab_stds': stab_stds,
        'r2_scores': r2_scores,
        'r2_stds': r2_stds
    }


