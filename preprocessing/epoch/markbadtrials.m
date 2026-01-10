% Load epoched SPM MEEG object
D = spm_eeg_load('D:\Reward\Epoching\be_mfffCorregistration&Filter.mat');

% Trials to reject
bad_idx = [44 112 113 149 200];

% Mark bad trials (flag only; does not delete data)
D = badtrials(D, bad_idx, true);
save(D);

% Remove all trials currently marked as bad (creates new dataset)
S = struct('D', D);
Dclean = spm_eeg_remove_bad_trials(S);
