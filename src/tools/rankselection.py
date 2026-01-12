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

def normalise_cp_tensor(cp_tensor):
    '''
    Docstring for normalise_cp_tensor
    
    :param cp_tensor: Description
    '''
    
    # Get weights/ factors 
    weights, factors = cp_tensor
    
    # Take copies
    weights = weights.copy()
    new_factors = [f.copy() for f in factors]
    
    # Get rank of current decomposition
    rank = weights.shape[0]
    
    for r in range(rank):
        for mode in len(new_factors):
            
            # Get norm of r-th component
            norm = np.linalg.nrom(new_factors[mode][:,r])
            
            weights[r] *= norm
            
            # Normalise factor vectors for the r-th component
            if norm > 0:
                new_factors[mode][:,r] /= norm
                            
                
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
    
    sim_matrix = np.zeros((rank,rank))
    
    for i in range(rank):
        for j in range(rank):
            
            weights_score = 1 - ( np.abs(lambdaA[i] - lambdaB[j] ) / max(lambdaA[i], lambdaB[j], 1e-16))
     
            factor_score = 1.0
            for mode in range(len(factorsA)):
                
                dot_prod = np.dot(factorsA[mode][:,i], factorsB[mode][:,j])
                
                factor_score *= dot_prod

            sim_matrix[i,j] = weights_score * factor_score
            
    # Hungarian Algo
    row_ind, col_ind = linear_sum_assignment(sim_matrix, maximize=True)
    
    score = sim_matrix[row_ind, col_ind].mean()

    return score




def rank_stability(tensor_data, rank, mask=None, n_repeats=10, verbose=0):
    '''
    Evaluates stability significantly faster by avoiding manual reconstruction.
    '''
    
    if verbose:
        print(f"--- Testing Rank {rank} with {n_repeats} repeats (Fast Optimized) ---")

    device = tensor_data.device if hasattr(tensor_data, 'device') else 'cpu'
    if mask is not None and hasattr(mask, 'to'):
        mask = mask.to(device)

    # Pre-calculate norm once for relative->absolute error conversion (optional but useful for display)
    # If mask is used, we only care about the norm of the observed part
    if mask is not None:
        norm_tensor = torch.norm(tensor_data * mask).item()
    else:
        norm_tensor = torch.norm(tensor_data).item()

    models = []
    final_errors = []

    for i in range(n_repeats):
        try:
            # OPTIMIZATION: Use return_errors=True to get the loss history directly.
            # This avoids reconstructing the full huge tensor manually.
            cp_tensor_obj, error_history = non_negative_parafac(
                tensor_data,
                rank=rank,
                init="random",
                mask=mask,
                n_iter_max=5000,
                tol=1e-7,
                return_errors=True,  # <--- CRITICAL SPEEDUP
                random_state=i
            )
            
            # TensorLy returns relative error (norm(diff) / norm(data))
            # We take the final error from the history.
            rel_error = error_history[-1]
            final_errors.append(rel_error)

            # Store model on GPU (detached) for fast similarity calculation later
            # unpacking CPTensor object
            weights, factors = cp_tensor_obj 
            
            # Detach from graph but KEEP ON GPU to save transfer time
            weights = weights.detach()
            factors = [f.detach() for f in factors]
            
            models.append((weights, factors))

        except Exception as e:
            print(f"Run {i} failed: {e}")

    if not models:
        return 0.0, 0.0

    # Identify Best Model (lowest error)
    best_idx = np.argmin(final_errors)
    best_model = models[best_idx]
    
    # Calculate Similarity
    # Compare all models to the Best Model
    similarities = []
    for i, model in enumerate(models):
        if i == best_idx:
            similarities.append(1.0)
        else:
            # Use the vectorized GPU function from the previous step
            # Note: Ensure you have defined 'paper_similarity_score' as discussed
            score = paper_similarity_score(best_model, model)
            similarities.append(score)

    mean_stability = np.mean(similarities)
    std_stability = np.std(similarities)

    if verbose:
        # Convert relative error back to absolute for display if you prefer
        best_abs_error = final_errors[best_idx] * norm_tensor
        print(f"Rank {rank}: Best Relative Error={final_errors[best_idx]:.6f} (Abs: {best_abs_error:.4f}) | Mean Similarity={mean_stability:.4f}")

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


    plt.errorbar(ranks,stabilities,stds,marker='o',capsize=4, label='ALS Stabilty Score')

    plt.xlabel("Rank",fontsize=14)
    plt.ylabel("Stability (a.u.)",fontsize=14)
    plt.ylim(0,1.2)
    plt.axhline(0.9,0,100,ls='--',color='r')
    plt.text(min(ranks),0.91,'0.9',color='r')
    plt.title("CP Rank Stability (ALS)",fontsize=14)
    plt.grid(alpha=0.5,ls='--')
    plt.minorticks_on()
    plt.xticks(ranks)
    plt.legend(framealpha=0,loc='upper right')