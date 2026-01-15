import numpy as np
import matplotlib.pyplot as plt
from tensorly.decomposition import non_negative_parafac_hals
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
    reconstructed : torch.
    '''
    reconstructed = tl.cp_to_tensor(cp_tensor)

    numerator = torch.norm(full - reconstructed)
    denominator = torch.norm(full)

    return numerator / denominator




def similarity_score(cp_tensor_A, cp_tensor_B):
    '''
    Calculate similarity score
       Parameters:
    factorsA : list of np.ndarray
        List of factor matrices from the first run.
    factorsA : list of np.ndarray
        List of factor matrices from the second run.
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

    sim_matrix = np.zeros((rank, rank),dtype=np.float32)

    for i in range(rank):
        for j in range(rank):

            max_lam = max(lambdaA[i], lambdaB[j])
            if max_lam == 0:
                weight_score = np.float32(1.0) # Both are zero
            else:
                weight_score = np.float32(1.0) - (np.abs(lambdaA[i] - lambdaB[j]) / max_lam)

            factor_score = np.float32(1.0)
            for mode in range(len(factorsA)):

                dot_prod = np.dot(factorsA[mode][:, i], factorsB[mode][:, j])

                factor_score *= dot_prod

            sim_matrix[i, j] = weight_score * factor_score

    # Hungarian Algo
    row_ind, col_ind = linear_sum_assignment(sim_matrix, maximize=True)

    score = sim_matrix[row_ind, col_ind].mean(dtype=np.float32)

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

    if tensor_data.dtype != torch.float32:
        tensor_data = tensor_data.to(torch.float32)

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
            cp_tensor = non_negative_parafac_hals(
                tensor_data,
                rank=rank,
                init="random",
                n_iter_max=2000,
                tol=1e-6,
                random_state=i  # Ensure different random init
            )

            weights, factors = cp_tensor
            
            norm_weights, norm_factors = cp_normalize((weights, factors))
            cpu_weights = norm_weights.detach().cpu()
            cpu_factors = [f.detach().cpu() for f in norm_factors]
            models.append((cpu_weights, cpu_factors))
     
     
            # Compute Reconstruction Error to find the best model
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

    # Identify Best Model (global minimum candidate)
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

    mean_stability = np.mean(similarities,dtype=np.float32)
    std_stability = np.std(similarities,dtype=np.float32)

    if verbose:
        print(
            f"Rank {rank}: Best Error={errors[best_idx]:.4f} | Mean Similarity={mean_stability:.4f}")

    return mean_stability, std_stability


def rank_fit(tensor_data, rank, mask=None, n_repeats=5, verbose=0):
    '''


    '''
    
    if tensor_data.dtype != torch.float32:
        tensor_data = tensor_data.to(torch.float32)
    
    device = tensor_data.device if hasattr(tensor_data, 'device') else 'cpu'
    if mask is not None and hasattr(mask, 'to'):
        mask = mask.to(device)

    # We subtract the mean to ignore the static background offset
    global_mean = torch.mean(tensor_data)
    sst = torch.sum((tensor_data - global_mean) ** 2)

    fits = []

    for i in range(n_repeats):

        try:
             cp_tensor = non_negative_parafac_hals(
                tensor_data,
                rank=rank,
                init="random",
                n_iter_max=1000,
                tol=1e-5,
                random_state=i  # Ensure different random init
            )


             rec_tensor = tl.cp_to_tensor(cp_tensor)

             # SSR is the sum of squared errors
             ssr = torch.sum((tensor_data - rec_tensor) ** 2)

             relative_err = relative_err.detach().cpu().numpy()

             r2_score = 1 - (ssr / sst)
             
             fits.append(r2_score.detach().cpu().item())
             
        except Exception as e:
            print(f"Run {i} failed: {e}") 
    
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

# based on https://gist.github.com/willshiao/2c0d7cc1133d8fa31587e541fef480fb, adapted for 4-th order array

def kron_mat_ten(matrices, X):
    
    Y = X
     
    for mode, M in enumerate(matrices):
        Y = mode_dot(Y, M, mode)
    return Y

def corcondia(tensor_data, rank=1, init='random'):
    
      cp_tensor = non_negative_parafac_hals(
                tensor_data,
                rank=rank,
                init="random",
                n_iter_max=1000,
                tol=1e-5,
            )
      
      _, factors = cp_tensor
      
      reconstructed_tensor = tl.cp_to_tensor(cp_tensor)
      
      
      # SVD each factor
      
      U_list, S_inv_list, Vt_list = [], [], []
      
      for mode in factors:
          
          U,s,Vt = torch.linalg.svd(mode, full_matrices=False)

          U_list.append(U)
          S_inv_list.append(torch.diag(1/s))
          Vt_list.append(Vt.T)

      part1 = kron_mat_ten(U_list, reconstructed_tensor)
      part2 = kron_mat_ten(S_inv_list, part1)
      G = kron_mat_ten(Vt_list, part2)

      T = torch.zeros((rank,) * tl.ndim(tensor_data))
      idx = (torch.arange(rank),) * tl.ndim(tensor_data)
      T[idx] = 1.0

      result = 100.0 * (1.0 - torch.sum((G - T) ** 2) / float(rank))
    

      return result.detach().cpu().numpy()
    
    
    
    
    
    
    
    
    
  
