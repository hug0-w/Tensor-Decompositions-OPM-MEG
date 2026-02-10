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
    Implements Equation 14 from Williams et al. (2018):
    
    score = max_{σ∈U} (1/R) Σ_r [ (1 - |λ_r - λ_{σ(r)}| / max(λ_r, λ_{σ(r)}))
                                    · |w_r^T w'_{σ(r)}| · |b_r^T b'_{σ(r)}| · |a_r^T a'_{σ(r)}| ]
    
    Uses Hungarian algorithm for optimal permutation matching (exact solution,
    equivalent to exhaustive search but O(R³) instead of O(R!)).
    """
    lambdaA, factorsA = cp_normalize(cp_tensor_A)
    lambdaB, factorsB = cp_normalize(cp_tensor_B)

    if torch.is_tensor(lambdaA):
        lambdaA = lambdaA.detach().cpu().numpy().astype(np.float64)
    else:
        lambdaA = np.array(lambdaA, dtype=np.float64)

    if torch.is_tensor(lambdaB):
        lambdaB = lambdaB.detach().cpu().numpy().astype(np.float64)
    else:
        lambdaB = np.array(lambdaB, dtype=np.float64)

    factorsA = [f.detach().cpu().numpy().astype(np.float64) if torch.is_tensor(f)
                else np.array(f, dtype=np.float64) for f in factorsA]
    factorsB = [f.detach().cpu().numpy().astype(np.float64) if torch.is_tensor(f)
                else np.array(f, dtype=np.float64) for f in factorsB]

    rank = factorsA[0].shape[1]
    n_modes = len(factorsA)
    sim_matrix = np.zeros((rank, rank), dtype=np.float64)

    for i in range(rank):
        for j in range(rank):
            # Weight similarity (paper uses plain λ, assuming positive after normalization)
            # cp_normalize should produce positive weights, but we use abs() defensively
            lam_i = np.abs(lambdaA[i])
            lam_j = np.abs(lambdaB[j])
            max_lam = max(lam_i, lam_j)
            if max_lam == 0:
                weight_score = 1.0
            else:
                weight_score = 1.0 - (np.abs(lam_i - lam_j) / max_lam)

            # Factor similarity: product of absolute cosine similarities across ALL modes
            factor_score = 1.0
            for mode in range(n_modes):
                vec_a = factorsA[mode][:, i]
                vec_b = factorsB[mode][:, j]

                norm_a = np.linalg.norm(vec_a)
                norm_b = np.linalg.norm(vec_b)

                if norm_a > 0 and norm_b > 0:
                    cosine_sim = np.abs(np.dot(vec_a, vec_b) / (norm_a * norm_b))
                else:
                    cosine_sim = 0.0

                factor_score *= cosine_sim
            sim_matrix[i, j] = weight_score * factor_score

    # Hungarian algorithm for optimal matching (exact optimal for linear sum assignment)
    row_ind, col_ind = linear_sum_assignment(sim_matrix, maximize=True)
    score = sim_matrix[row_ind, col_ind].mean()

    return float(score)


def run_parafac(tensor_data, rank, random_state=None, n_iter_max=2000, tol=1e-8, init='random'):
    """
    Run CP decomposition.

    Parameters:
        tensor_data : torch.Tensor - Input tensor
        rank : int - Rank of decomposition
        random_state : int or None - Random seed
        n_iter_max : int - Maximum iterations
        tol : float - Convergence tolerance
        init : str - Initialization method ('random' or 'svd')

    Returns:
        cp_tensor : tuple (weights, factors)
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


def rank_stability_and_r2(tensor_data, rank, n_repeats=10, verbose=0):
    """
    Evaluates both stability and R² of CP decomposition at a given rank,
    following Williams et al. (2018) Figure 2F.

    Stability: each model's similarity to the best-fit model (lowest error).
    R²: from the best-fit model.

    Key differences from original code:
    - NO error_threshold filtering: ALL models are compared to best (as in paper)
    - Self-comparison excluded from similarity mean
    - R² reported from best model, plus mean/std across all models

    Parameters:
        tensor_data : torch.Tensor - Input tensor data
        rank : int - Rank of the CP decomposition
        n_repeats : int - Number of repeats per rank
        verbose : int - Verbosity level

    Returns:
        mean_stability : float - Mean similarity of all models to best
        std_stability : float - Std of similarities
        best_r2 : float - R² of the best-fit model
        all_similarities : list - Individual similarity scores (for plotting)
        all_errors : list - Individual errors (for error plot)
        all_r2 : list - Individual R² values
    """
    if tensor_data.dtype != torch.float32:
        tensor_data = tensor_data.to(torch.float32)

    if verbose:
        print(f"--- Testing Rank {rank} with {n_repeats} repeats ---")

    models = []
    errors = []
    r2_scores = []

    for i in range(n_repeats):
        try:
            cp_tensor = run_parafac(tensor_data, rank=rank, random_state=i)

            # Compute error
            error = rel_error(tensor_data, cp_tensor)
            error = float(error.detach().cpu().numpy())
            errors.append(error)

            # Compute R²
            r2 = r_squared(tensor_data, cp_tensor)
            r2_scores.append(r2)

            # Store normalized model on CPU
            weights, factors = cp_tensor
            norm_weights, norm_factors = cp_normalize((weights, factors))
            cpu_weights = norm_weights.detach().cpu() if hasattr(norm_weights, 'cpu') else norm_weights
            cpu_factors = [f.detach().cpu() if hasattr(f, 'cpu') else f for f in norm_factors]
            models.append((cpu_weights, cpu_factors))

        except Exception as e:
            if verbose:
                print(f"  Run {i} failed: {e}")
            # Append None placeholders so indices stay aligned
            errors.append(np.inf)
            r2_scores.append(np.nan)
            models.append(None)

    errors = np.array(errors)
    r2_scores = np.array(r2_scores)

    # Find valid (non-failed) models
    valid_mask = np.isfinite(errors)
    valid_indices = np.where(valid_mask)[0]

    if len(valid_indices) == 0:
        return 0.0, 0.0, 0.0, [], [], []

    # Best model = lowest error among valid models
    best_idx = valid_indices[np.argmin(errors[valid_indices])]
    best_model = models[best_idx]
    best_r2 = r2_scores[best_idx]

    # Stability: compare ALL valid models to best (paper: every dot compared to best)
    # Exclude self-comparison
    similarities = []
    for i in valid_indices:
        if i == best_idx:
            continue  # Don't include self-similarity
        score = similarity_score(best_model, models[i])
        similarities.append(score)

    if len(similarities) == 0:
        # Only one valid model — no stability information
        mean_stability = np.nan
        std_stability = np.nan
    else:
        mean_stability = float(np.mean(similarities))
        std_stability = float(np.std(similarities))

    if verbose:
        print(f"  {len(valid_indices)} models converged")
        print(f"  Rank {rank}: Best Error={errors[best_idx]:.4f} | Best R²={best_r2:.4f}")
        print(f"  Mean Stability={mean_stability:.4f} ± {std_stability:.4f}")

    return (mean_stability, std_stability, float(best_r2),
            similarities, errors[valid_indices].tolist(), r2_scores[valid_indices].tolist())


def rank_selection(tensor_data, ranks=range(1, 11), n_repeats=10, verbose=1):
    """
    Run stability and R² analysis across multiple ranks.
    Produces data for error plots and similarity plots as in Williams et al. (2018) Figure 2F.

    Parameters:
        tensor_data : torch.Tensor - Input tensor
        ranks : range or list - Ranks to test
        n_repeats : int - Number of repeats per rank
        verbose : int - Verbosity level

    Returns:
        results : dict with per-rank stability, R², and raw data for plotting
    """
    results = {
        'ranks': list(ranks),
        'mean_stabilities': [],
        'std_stabilities': [],
        'best_r2': [],
        # Raw data for scatter plots (paper Figure 2F style)
        'all_similarities': {},  # rank -> list of similarity scores
        'all_errors': {},        # rank -> list of errors per run
        'all_r2': {},            # rank -> list of R² per run
    }

    for rank in ranks:
        if verbose:
            print(f"\n=== Rank {rank} ===")

        stab, stab_std, best_r2, sims, errs, r2s = rank_stability_and_r2(
            tensor_data, rank, n_repeats=n_repeats, verbose=verbose
        )
        results['mean_stabilities'].append(stab)
        results['std_stabilities'].append(stab_std)
        results['best_r2'].append(best_r2)
        results['all_similarities'][rank] = sims
        results['all_errors'][rank] = errs
        results['all_r2'][rank] = r2s

    return results


def plot_rank_selection(results, figsize=(12, 5)):
    """
    Plot error and similarity plots as in Williams et al. (2018) Figure 2F.

    Parameters:
        results : dict from rank_selection()
        figsize : tuple
    """
    ranks = results['ranks']

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # --- Error plot (Figure 2F inset 1) ---
    ax = axes[0]
    for rank in ranks:
        errs = results['all_errors'].get(rank, [])
        ax.scatter([rank] * len(errs), errs, c='black', s=15, alpha=0.6, zorder=2)
    # Red line: minimum error at each rank
    min_errors = [min(results['all_errors'][r]) if results['all_errors'][r] else np.nan for r in ranks]
    ax.plot(ranks, min_errors, 'r-', linewidth=1.5, zorder=3)
    ax.set_xlabel('# components')
    ax.set_ylabel('Normalized reconstruction error')
    ax.set_title('Error plot')

    # --- Similarity plot (Figure 2F inset 2) ---
    ax = axes[1]
    for rank in ranks:
        sims = results['all_similarities'].get(rank, [])
        ax.scatter([rank] * len(sims), sims, c='black', s=15, alpha=0.6, zorder=2)
    # Red line: mean similarity at each rank
    ax.plot(ranks, results['mean_stabilities'], 'r-', linewidth=1.5, zorder=3)
    ax.set_xlabel('# components')
    ax.set_ylabel('Similarity')
    ax.set_title('Similarity plot')
    ax.set_ylim(-0.05, 1.05)

    plt.tight_layout()
    return fig