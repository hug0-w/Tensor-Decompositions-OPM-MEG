% Epoch OUTCOME trials from NI-TRIG-2 using spm_opm_epoch_trigger

clear; clc;

Dfile = 'D:\Reward\Epoching\mfffCorregistration&Filter.mat';

spm('defaults','eeg');

mD = spm_eeg_load(Dfile);

S = [];
S.D               = mD;
S.timewin         = [-500 1500];     % ms
S.triggerChannels = {'NI-TRIG-2'};
S.condLabels      = {'Outcome'};
S.bc              = 0;               % no baseline correction here 
S.prefix          = 'e_outcome_';

eD = spm_opm_epoch_trigger(S);

S = [];
S.D = eD;
S.timewin = [-500 -50];
eD = spm_eeg_bc(S);


fprintf('OUTCOME epochs: %d\n', eD.ntrials);
if eD.ntrials ~= 200
    warning('OUTCOME: expected 200 trials, got %d.', eD.ntrials);
end
