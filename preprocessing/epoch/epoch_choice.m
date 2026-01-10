% Epoch CHOICE trials from NI-TRIG-4, NI-TRIG-5, NI-TRIG-7
% Uses spm_opm_epoch_trigger (no thresholding)

clear; clc;

Dfile = 'D:\Reward\Epoching\mfffCorregistration&Filter.mat';

spm('defaults','eeg');

mD = spm_eeg_load(Dfile);

S = [];
S.D               = mD;
S.timewin         = [-500 1500];                 % ms
S.triggerChannels = {'NI-TRIG-4','NI-TRIG-5','NI-TRIG-7'};
S.condLabels      = {'Choice','Choice','Choice'};
S.bc              = 0;
S.prefix          = 'e_choice_';

eD = spm_opm_epoch_trigger(S);

S = [];
S.D = eD;
S.timewin = [-500 -50];
eD = spm_eeg_bc(S);

fprintf('CHOICE epochs: %d\n', eD.ntrials);
if eD.ntrials ~= 200
    warning('CHOICE: expected 200 trials, got %d.', eD.ntrials);
end
