import numpy as np
import matplotlib.pyplot as plt
from tensorly.decomposition import parafac, constrained_parafac
from scipy.optimize import linear_sum_assignment
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
    Uses absolute value of factor correlations to handle sign ambiguity.
    
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


def run_parafac(tensor_data, rank, non_negative_modes=None, random_state=None,
                n_iter_max=2000, tol=1e-6):
    """
    Run CP decomposition with optional non-negativity constraints on specific modes.
    
    Parameters:
    tensor_data : torch.Tensor
        Input tensor.
    rank : int
        Rank of decomposition.
    non_negative_modes : list or None
        List of mode indices that should be non-negative (e.g., [1, 2] for modes 1 and 2).
    random_state : int or None
        Random seed.
    n_iter_max : int
        Maximum iterations.
    tol : float
        Convergence tolerance.
    
    Returns:
    cp_tensor : tuple
        (weights, factors)
    """
    if non_negative_modes is not None and len(non_negative_modes) > 0:
        # Use constrained_parafac for non-negativity on specific modes
        # API expects dictionary: {mode_index: True/False} or bool for all modes
        n_modes = len(tensor_data.shape)
        non_negative_dict = {mode: (mode in non_negative_modes) for mode in range(n_modes)}
        
        cp_tensor = constrained_parafac(
            tensor_data,
            rank=rank,
            init="random",
            n_iter_max=n_iter_max,
            n_iter_max_inner=10,
            tol_outer=tol,
            tol_inner=1e-4,
            random_state=random_state,
            non_negative=non_negative_dict,  # Dictionary {mode: bool}
            verbose=0
        )
    else:
        # Standard CP decomposition (faster when no constraints needed)
        cp_tensor = parafac(
            tensor_data,
            rank=rank,
            init="random",
            n_iter_max=n_iter_max,
            tol=tol,
            random_state=random_state,
            normalize_factors=True,
            linesearch=True
        )
    
    return cp_tensor


def rank_stability(tensor_data, rank, mask=None, n_repeats=10, 
                   non_negative_modes=None, verbose=0):
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
    non_negative_modes : list, optional
        Modes that should be non-negative (e.g., [1, 2] for frequency and time).
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
        nn_str = f" (non-negative modes: {non_negative_modes})" if non_negative_modes else ""
        print(f"--- Testing Rank {rank} with {n_repeats} repeats{nn_str} ---")

    device = tensor_data.device if hasattr(tensor_data, 'device') else 'cpu'
    if mask is not None and hasattr(mask, 'to'):
        mask = mask.to(device)

    models = []
    errors = []

    for i in range(n_repeats):
        try:
            cp_tensor = run_parafac(
                tensor_data, 
                rank=rank, 
                non_negative_modes=non_negative_modes,
                random_state=i
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
            if verbose:
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


def rank_fit(tensor_data, rank, mask=None, n_repeats=5, 
             non_negative_modes=None, verbose=0):
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
    non_negative_modes : list, optional
        Modes that should be non-negative.
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

    # Total sum of squares
    global_mean = torch.mean(tensor_data)
    sst = torch.sum((tensor_data - global_mean) ** 2)

    fits = []

    for i in range(n_repeats):
        try:
            cp_tensor = run_parafac(
                tensor_data,
                rank=rank,
                non_negative_modes=non_negative_modes,
                random_state=i
            )

            rec_tensor = tl.cp_to_tensor(cp_tensor)

            # Sum of squared residuals
            if mask is not None:
                ssr = torch.sum(((tensor_data - rec_tensor) * mask) ** 2)
            else:
                ssr = torch.sum((tensor_data - rec_tensor) ** 2)

            r2_score = 1 - (ssr / sst)
            fits.append(r2_score.detach().cpu().item())
            
            if verbose > 1:
                print(f"  Run {i}: R² = {fits[-1]:.4f}")
             
        except Exception as e:
            if verbose:
                print(f"Run {i} failed: {e}")
    
    if not fits:
        return 0.0, 0.0
    
    best_fit = float(np.max(fits))
    std_fit = float(np.std(fits))
    
    return best_fit, std_fit


def stability_plot(ranks, stabilities, stds, title_suffix=""):
    '''
    Plots the stability scores against ranks.
    '''
    plt.figure(figsize=(10, 6))
    plt.errorbar(ranks, stabilities, stds, marker='o', capsize=4, label='CP Stability Score')
    plt.xlabel("Rank", fontsize=14)
    plt.ylabel("Stability (a.u.)", fontsize=14)
    plt.ylim(0, 1.2)
    plt.axhline(0.9, 0, 100, ls='--', color='r')
    plt.text(min(ranks), 0.91, '0.9', color='r')
    plt.title(f"CP Rank Stability{title_suffix}", fontsize=14)
    plt.grid(alpha=0.5, ls='--')
    plt.minorticks_on()
    plt.xticks(ranks)
    plt.legend(framealpha=0, loc='upper right')
    plt.tight_layout()
    return plt.gcf()


def fit_plot(ranks, fits, stds, title_suffix=""):
    '''
    Plots the fit (R²) scores against ranks.
    '''
    plt.figure(figsize=(10, 6))
    plt.errorbar(ranks, fits, stds, marker='s', color='green', capsize=4, label='R² Score')
    plt.xlabel("Rank", fontsize=14)
    plt.ylabel("R² (Variance Explained)", fontsize=14)
    plt.ylim(0, 1.05)
    plt.title(f"CP Decomposition Fit vs Rank{title_suffix}", fontsize=14)
    plt.grid(alpha=0.5, ls='--')
    plt.minorticks_on()
    plt.xticks(ranks)
    plt.legend(framealpha=0, loc='lower right')
    plt.tight_layout()
    return plt.gcf()


def kron_mat_ten(matrices, X):
    '''
    Applies Kronecker-like transformation to tensor.
    '''
    Y = X
    for mode, M in enumerate(matrices):
        Y = mode_dot(Y, M, mode)
    return Y


def corcondia(tensor_data, rank=1, init='random', non_negative_modes=None):
    '''
    Computes CORCONDIA (Core Consistency Diagnostic) for CP decomposition.
    Works with 3-mode or higher order tensors.
    
    Parameters:
    tensor_data : torch.Tensor
        Input tensor (any order >= 3).
    rank : int
        Rank for decomposition.
    init : str
        Initialization method.
    non_negative_modes : list, optional
        Modes that should be non-negative.
    
    Returns:
    cc : float
        CORCONDIA score (100 = perfect trilinear structure).
    '''
    n_modes = len(tensor_data.shape)
    
    if n_modes < 3:
        raise ValueError(f"CORCONDIA requires at least 3-mode tensor, got {n_modes}-mode")
    
    # Run decomposition
    cp_tensor = run_parafac(
        tensor_data,
        rank=rank,
        non_negative_modes=non_negative_modes,
        random_state=42,
        n_iter_max=5000,
        tol=1e-8
    )
    
    _, factors = cp_tensor
    
    # SVD of each factor and compute transformations
    Us = []
    SIs = []
    Vhs = []
    
    for factor in factors:
        U, S, Vh = torch.linalg.svd(factor, full_matrices=False)
        Us.append(U[:, :rank])
        SI = torch.linalg.pinv(torch.diag(S[:rank]))
        SIs.append(SI)
        Vhs.append(Vh[:rank, :])
    
    # Compute core tensor G
    y = kron_mat_ten([U.T for U in Us], tensor_data)
    z = kron_mat_ten(SIs, y)
    G = kron_mat_ten(Vhs, z)
    
    # Ideal superdiagonal core tensor (works for any number of modes)
    ideal_shape = tuple([rank] * n_modes)
    C_ideal = torch.zeros(ideal_shape, device=tensor_data.device)
    for i in range(rank):
        idx = tuple([i] * n_modes)
        C_ideal[idx] = 1
    
    # Compute CORCONDIA with correct normalization
    diff_sq = torch.sum((G - C_ideal) ** 2)
    ideal_norm_sq = torch.sum(C_ideal ** 2)  # = rank
    cc = 100 * (1 - (diff_sq / ideal_norm_sq))
    
    return cc.detach().cpu().item()


def rank_selection(tensor_data, ranks=range(1, 11), n_repeats=10, 
                   non_negative_modes=None, verbose=1):
    '''
    Run stability and fit analysis across multiple ranks.
    
    Parameters:
    tensor_data : torch.Tensor
        Input tensor.
    ranks : range or list
        Ranks to test.
    n_repeats : int
        Number of repeats per rank.
    non_negative_modes : list, optional
        Modes that should be non-negative (e.g., [1, 2] for frequency and time).
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
    
    if verbose and non_negative_modes:
        print(f"Using non-negative constraints on modes: {non_negative_modes}")
    
    for rank in ranks:
        if verbose:
            print(f"\n=== Rank {rank} ===")
        
        stab, stab_std = rank_stability(
            tensor_data, rank, 
            n_repeats=n_repeats, 
            non_negative_modes=non_negative_modes,
            verbose=verbose
        )
        stabilities.append(stab)
        stab_stds.append(stab_std)
        
        fit, fit_std = rank_fit(
            tensor_data, rank, 
            n_repeats=n_repeats,
            non_negative_modes=non_negative_modes,
            verbose=verbose
        )
        fits.append(fit)
        fit_stds.append(fit_std)
        
        if verbose:
            print(f"Rank {rank}: Stability={stab:.3f}±{stab_std:.3f}, R²={fit:.3f}±{fit_std:.3f}")
    
    results = {
        'ranks': list(ranks),
        'stabilities': stabilities,
        'stab_stds': stab_stds,
        'fits': fits,
        'fit_stds': fit_stds,
        'non_negative_modes': non_negative_modes
    }
    
    return results


def suggest_rank(results, stability_threshold=0.85, fit_threshold=0.8):
    """
    Suggest optimal rank based on stability and fit criteria.
    
    Parameters:
    results : dict
        Output from rank_selection().
    stability_threshold : float
        Minimum acceptable stability (default 0.85).
    fit_threshold : float
        Minimum acceptable R² (default 0.8).
    
    Returns:
    suggested_rank : int or None
        Suggested rank, or None if no rank meets criteria.
    """
    ranks = results['ranks']
    stabilities = results['stabilities']
    fits = results['fits']
    
    # Find ranks that meet both criteria
    valid_ranks = []
    for i, r in enumerate(ranks):
        if stabilities[i] >= stability_threshold and fits[i] >= fit_threshold:
            valid_ranks.append((r, stabilities[i], fits[i]))
    
    if not valid_ranks:
        print(f"No rank meets criteria (stability >= {stability_threshold}, R² >= {fit_threshold})")
        # Suggest highest stable rank
        stable_ranks = [(r, s, f) for i, (r, s, f) in 
                        enumerate(zip(ranks, stabilities, fits)) if s >= stability_threshold]
        if stable_ranks:
            best = max(stable_ranks, key=lambda x: x[2])  # highest R² among stable
            print(f"Suggestion: Rank {best[0]} (stability={best[1]:.3f}, R²={best[2]:.3f})")
            return best[0]
        return None
    
    # Among valid ranks, prefer highest rank (more components) if R² still increasing
    # Or use elbow method: find where fit improvement slows
    best_rank = max(valid_ranks, key=lambda x: x[2])  # highest R²
    print(f"Suggested rank: {best_rank[0]} (stability={best_rank[1]:.3f}, R²={best_rank[2]:.3f})")
    
    return best_rank[0]