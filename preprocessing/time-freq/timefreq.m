in_file    = 'D:\Reward\FTA\Choice\avg_choice_rbe_mfffCorregistration&Filter.dat'; 
out_mat    = 'tf_choice_spm.mat';
chantype   = 'MEGMAG';
timewin_ms = [-500 1500];
freqs_hz   = 5:2:70;
method     = 'morlet';
% ----------------------------------

%% Load data
D = spm_eeg_load(in_file);

%% Select good channels for the TF computation
chan_idx_orig = D.indchantype(chantype, 'GOOD');
chan_lbl      = D.chanlabels(chan_idx_orig);

%% Time–frequency analysis
S = [];
S.D = D;
S.channels = chan_lbl; 
S.timewin = timewin_ms;
S.method = method;
S.frequencies = freqs_hz;
S.phase = 0; 
Dtf = spm_eeg_tf(S);


base_idx = find(Dtf.time < 0);
if isempty(base_idx)
    error('No baseline time points found (time < 0).');
end

data = Dtf(:, :, :, 1); 

tf_avg = squeeze(mean(log(data), 1));
baseline = mean(log(data(:, :, base_idx)), 3); 
baseline_avg = mean(baseline, 1);              

tf = tf_avg - baseline_avg'; 

time = Dtf.time;
freq = Dtf.frequencies;

save(out_mat, 'tf', 'time', 'freq', 'chantype', '-v7');
fprintf('Saved %s (freq x time)\n', out_mat);