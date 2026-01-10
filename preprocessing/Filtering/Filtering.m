%% Written by Hugo Wharton (07/12/2025)
addpath('D:\spm') % spm path
spm('defaults','EEG')

% Define paths
savepath = 'D:\Reward\Corregistration&Filter\';
% Ensure savepath exists
if ~exist(savepath, 'dir')
    mkdir(savepath);
end

%% Load OPM data object
filename = 'D:\Reward\Corregistration&Filter\Corregistration&Filter.mat';
D = spm_eeg_load(filename);

%% 2. PHASE 1: FILTERING
disp('Starting Phase 1: Filtering...');

% A. High-Pass Filter (1 Hz) -> Creates 'f' prefix
S = [];
S.D = D;
S.channels = 'MEG';
S.freq = [1];
S.band = 'high';
S.prefix = 'f'; 
fD = spm_eeg_ffilter(S);

% B. Low-Pass Filter (70 Hz) -> Creates 'ff' prefix
S = [];
S.D = fD;
S.freq = [70];
S.band = 'low';
fD = spm_eeg_ffilter(S);

% C. Notch Filter (48-52 Hz) -> Creates 'fff' prefix
S = [];
S.D = fD;
S.freq = [48, 52];
S.band = 'stop';
fD = spm_eeg_ffilter(S);

%% 3. PHASE 2: MARK BAD CHANNELS
disp('Starting Phase 2: Marking Bad Channels...');

bad_channel_list = {'G2-DH-Y', 'G2-A9-Y', 'G2-DH-Z', 'G2-MW-Y', 'G2-MW-Z', 'G2-A9-Z'};

% Robust method to find indices and mark bad
bad_indices = fD.indchannel(bad_channel_list);
if ~isempty(bad_indices)
    fD = fD.badchannels(bad_indices, 1);
    fD.save(); % Save changes to the current 'fff' file
else
    warning('None of the specified bad channels were found in the dataset.');
end

%% 4. PHASE 3: OPM AMM & MOVE
disp('Starting Phase 3: AMM and Final Move...');

%% Here we use AMM since #channels = 130 (>99)

S = [];
S.D = fD;                
S.corrLim = 0.95;
mD = spm_opm_amm(S); % Creates 'amm_' or similar prefix

% Move the final file to savepath
% Note: Intermediate files (f, ff, fff) remain in the directory of the original D 

S = [];
S.D = mD;
S.outfile = fullfile(savepath, mD.fname); 

disp(['Processing complete. Final file located at: ' fullfile(savepath, mD.fname)]);