% spm_tf_power_time_ms_freqbins_stream_ds500_nobase_with_mne.m
%
% Epoched SPM MEEG -> Downsample to 500 Hz -> SPM TF (no trial averaging)
% -> power tensor streamed to disk (v7.3):
%   Pow(trial, chan, fbin, time)  (single)
%
% Also stores MNE-friendly sensor metadata for topoplots:
%   mne.ch_names  (cellstr, aligned to Pow channel dimension)
%   mne.ch_pos_m  (Nx3 double, meters, aligned to Pow channel dimension)
%   mne.coord_frame (string)
%
% No baseline correction here.

%% ---------- USER SETTINGS ----------
in_file     = 'D:\Reward\FTA\Choice\choice_rbe_mfffCorregistration&Filter.mat';
out_mat     = 'choice_pow_trial_chan_fbin_time_ds500_tpos.mat';

chantype    = 'MEGMAG';
timewin_ms  = [-500 1500];       % TF window (ms)
freqs_hz    = 1:2:70;            % TF frequencies to compute (Hz)
method      = 'morlet';

ds_fs       = 500;               % target sampling rate (Hz)

% Define frequency bins as [low high] in Hz
fbins = [
    1 4;
    5 7;
    8 12;
    13 20;
    21 30;
    31 45;
    46 70
];

%% Load epoched data
eD = spm_eeg_load(in_file);

%% Downsample to 500 Hz 
if abs(eD.fsample - ds_fs) > 1e-6
    Sd = [];
    Sd.D = eD;
    Sd.fsample_new = ds_fs;   
    Dds = spm_eeg_downsample(Sd);
else
    Dds = eD;
end

%% Select good channels
chan_idx = Dds.indchantype(chantype, 'GOOD');
chan_lbl = Dds.chanlabels(chan_idx);

%% ---------- EXPORT MNE-FRIENDLY SENSOR INFO ----------
% Expectation:
%   sensMEG.label    : all MEG channel labels (cellstr)
%   sensMEG.chanpos  : Nx3 channel positions (in mm)  OR
%   sensMEG.pnt      : Nx3 channel positions (in mm)
%


sensMEG = Dds.sensors('MEG');

if isfield(sensMEG, 'chanpos')
    pos_all = sensMEG.chanpos;
else
    pos_all = sensMEG.pnt;
end

lab_all = sensMEG.label;

[tf, idx] = ismember(chan_lbl, lab_all);

ch_pos_m = nan(numel(chan_lbl), 3);
ch_pos_m(tf,:) = pos_all(idx(tf), :) / 1000;  %  mm -> m

mne = struct();
mne.ch_names     = chan_lbl(:);
mne.ch_pos_m     = ch_pos_m;      % meters
mne.coord_frame  = 'head';        % set to what you use in MNE
mne.sensors_meg  = sensMEG;       % optional: keep full sensor struct

%% Time–frequency analysis on downsampled epoched data
S = [];
S.D           = Dds;
S.channels    = chan_lbl;
S.timewin     = timewin_ms;
S.method      = method;
S.frequencies = freqs_hz;

Dtf = spm_eeg_tf(S);

Sc = [];
Sc.D = Dtf;
Sc.timewin = [0 1500];   
Dtf_pos = spm_eeg_crop(Sc);

% Dtf(channel, freq, time, trial)
freq = Dtf_pos.frequencies;   % Hz
time = Dtf_pos.time;          % seconds

nTrials = Dtf_pos.ntrials;
nChan   = numel(chan_lbl);
nTime   = numel(time);
%nBins   = size(fbins,1);
nFreqs = numel(freq);

%% Precompute TF frequency indices for each bin
%bin_fidx = cell(nBins,1);
%for b = 1:nBins
%    bin_fidx{b} = find(freq >= fbins(b,1) & freq <= fbins(b,2));
%end

%% Stream output
m = matfile(out_mat, 'Writable', true);

% Preallocate on disk: Trial x Chan x Fbin x Time
%m.Pow = zeros(nTrials, nChan, nBins, nTime, 'single');
m.Pow = zeros(nTrials, nChan, nFreqs, nTime, 'single');


% Save metadata
m.freq       = freq;
m.time       = time;
m.chan_lbl   = chan_lbl;
%m.fbins      = fbins;
m.timewin_ms = timewin_ms;
m.freqs_hz   = freqs_hz;
m.chantype   = chantype;
m.method     = method;
m.ds_fs      = ds_fs;
m.fs_orig    = eD.fsample;

% Save MNE metadata
m.mne        = mne;

%% Extract power per trial and bin frequencies
for tr = 1:nTrials
    X = Dtf_pos(:,:,:,tr);  % chan x freq x time

    % Ensure power
    if ~isreal(X)
        X = abs(X).^2;  % complex coeffs -> power
    end

    % Bin frequencies: chan x bin x time
    %Xbin = zeros(nChan, nBins, nTime, 'single');
    %for b = 1:nBins
    %    Xbin(:,b,:) = single(mean(X(:, bin_fidx{b}, :), 2));  % average across freqs in bin
    %end

    % Write this trial only
    m.Pow(tr,:,:,:) = reshape(single(X), [1 nChan nFreqs nTime]);
    %m.Pow(tr,:,:,:) = reshape(Xbin, [1 nChan nBins nTime]);
end

fprintf('Saved %s\nPow size = [%d trials x %d chan x %d fbin x %d time]\n', ...
    out_mat, nTrials, nChan, nFreqs, nTime);
fprintf('Downsample: %.1f Hz -> %.1f Hz\n', eD.fsample, Dds.fsample);
