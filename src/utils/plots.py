import matplotlib.pyplot as plt


def plot_factors_subplots(A, B, C, D):
    """
    Plots the factor matrices A, B, C, and D.

    Parameters
    ----------
    A : np.ndarray
        Factor matrix for mode 0 (e.g., voxels).
    B : np.ndarray
        Factor matrix for mode 1 (e.g., channels).
    C : np.ndarray
        Factor matrix for mode 2 (e.g., frequencies).
    D : np.ndarray
        Factor matrix for mode 3 (e.g., time).
    """

    R = A.shape[1]

    fig, axes = plt.subplots(R, 4, figsize=(18, 10), sharex=False)

    for r in range(R):
        # Mode 0 (trials)
        axes[r, 0].bar(range(A.shape[0]), A[:, r], color='tab:blue')
        axes[r, 0].set_title(f'Component {r+1} – Trials')
        axes[r, 0].set_ylabel('Amplitude')
        axes[r, 0].set_xlabel('Trial Number')
        axes[r, 0].set_xticklabels('')

        # Mode 1 (channels)
        axes[r, 1].plot(B[:, r], color='tab:orange', ls='', marker='o')
        axes[r, 1].set_title(f'Component {r+1} – Channels')
        axes[r, 1].set_ylabel('Amplitude')
        axes[r, 1].set_xlabel('Channel')

        # Mode 2 (freq)
        axes[r, 2].plot(C[:, r], color='tab:green', ls='', marker='o')
        axes[r, 2].set_title(f'Component {r+1} – Frequency')
        axes[r, 2].set_ylabel('Amplitude')
        axes[r, 2].set_xlabel('Frequency Band (Hz)')

        # Mode 3 (time)
        axes[r, 3].plot(D[:, r])
        axes[r, 3].set_title(f'Component {r+1} – Time')
        axes[r, 3].set_ylabel('Amplitude')
        axes[r, 3].set_xlabel('Time (s)')

    for ax in axes[-1, :]:
        ax.set_xlabel('Index')

    plt.tight_layout()
    plt.show()
    
def plot_factors(A, B, C, D):    
    """
    Plots the factor matrices A, B, C, and D without subplots,
    each component in a separate figure.

    Parameters
    ----------
    A : np.ndarray
        Factor matrix for mode 0 (e.g., trials).
    B : np.ndarray
        Factor matrix for mode 1 (e.g., channels).
    C : np.ndarray
        Factor matrix for mode 2 (e.g., frequencies).
    D : np.ndarray
        Factor matrix for mode 3 (e.g., time).
    """

    R = A.shape[1]

    for r in range(R):
        fig, axes = plt.subplots(1, 4, figsize=(18, 4))
        fig.suptitle(f'Component {r+1}', fontsize=16)

        # Mode 0 (trials)
        axes[0].bar(range(A.shape[0]), A[:, r], color='tab:blue')
        axes[0].set_title('Trials')
        axes[0].set_ylabel('Amplitude')
        axes[0].set_xlabel('Trial Number')

        # Mode 1 (channels)
        axes[1].plot(B[:, r], color='tab:orange', ls='', marker='o')
        axes[1].set_title('Channels')
        axes[1].set_ylabel('Amplitude')
        axes[1].set_xlabel('Channel')

        # Mode 2 (freq)
        axes[2].plot(C[:, r], color='tab:green', ls='', marker='o')
        axes[2].set_title('Frequency')
        axes[2].set_ylabel('Amplitude')
        freq_bands = ['1-4 Hz', '4-8 Hz', '8-12 Hz', '13 - 20 Hz', '21-30 Hz' , '31-45 Hz', '46-70 Hz']
        axes[2].set_xticks(range(len(freq_bands)))
        axes[2].set_xticklabels(freq_bands, rotation=45, ha='right')
        axes[2].set_xlabel('Frequency Band (Hz)')

        # Mode 3 (time)
        axes[3].plot(D[:, r])
        axes[3].set_title('Time')
        axes[3].set_ylabel('Amplitude')
        axes[3].set_xlabel('Time (s)')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) 
        plt.show
    
    
def plot_single_mode(factor_matrix, mode_name):
    """
    Plots a single factor matrix.

    Parameters
    ----------
    factor_matrix : np.ndarray
        The factor matrix to plot.
    mode_name : str
        The name of the mode (e.g., 'Channels', 'Frequencies', 'Time').
    """

    R = factor_matrix.shape[1]

    for r in range(R):
        plt.figure(figsize=(6, 4))
        plt.plot(factor_matrix[:, r], marker='o')
        plt.title(f'Component {r+1} – {mode_name}')
        plt.ylabel('Amplitude')
        plt.xlabel(mode_name)
        plt.grid()
        plt.show()
        
        
def plot_single_component_modes(A_component, B_component, C_component, D_component, component_number):
    """
    Plots the modes of a single component (A, B, C, D) in a 1x4 subplot.

    Parameters
    ----------
    A_component : np.ndarray
        The r-th column of factor matrix A (e.g., trials).
    B_component : np.ndarray
        The r-th column of factor matrix B (e.g., channels).
    C_component : np.ndarray
        The r-th column of factor matrix C (e.g., frequencies).
    D_component : np.ndarray
        The r-th column of factor matrix D (e.g., time).
    component_number : int
        The index of the component being plotted (for titles).
    """
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    fig.suptitle(f'Component {component_number}', fontsize=16)

    # Mode 0 (trials)
    axes[0].bar(range(len(A_component)), A_component, color='tab:blue')
    axes[0].set_title('Trials')
    axes[0].set_ylabel('Amplitude')
    axes[0].set_xlabel('Trial Number')

    # Mode 1 (channels)
    axes[1].plot(B_component, color='tab:orange', ls='', marker='o')
    axes[1].set_title('Channels')
    axes[1].set_ylabel('Amplitude')
    axes[1].set_xlabel('Channel')

    # Mode 2 (freq)
    axes[2].plot(C_component, color='tab:green', ls='', marker='o')
    axes[2].set_title('Frequency')
    axes[2].set_ylabel('Amplitude')
    freq_bands = ['1-4 Hz', '4-8 Hz', '8-12 Hz', '13 - 20 Hz', '21-30 Hz' , '31-45 Hz', '46-70 Hz']
    axes[2].set_xticks(range(len(freq_bands)))
    axes[2].set_xticklabels(freq_bands, rotation=45, ha='right')
    axes[2].set_xlabel('Frequency Band (Hz)')

    # Mode 3 (time)
    axes[3].plot(D_component)
    axes[3].set_title('Time')
    axes[3].set_ylabel('Amplitude')
    axes[3].set_xlabel('Time (s)')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()