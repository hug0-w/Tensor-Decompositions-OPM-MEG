%% Written by Hugo Wharton (07/12/2025)

addpath('D:\spm') %spm path
spm('defaults','EEG')


%% Load OPM data object
filename = 'D:\Reward\Corregistration&Filter\mfffCorregistration&Filter.mat';
D = spm_eeg_load(filename);

%% Power spectral density 

S=[];
S.triallength = 5000; 
S.plot=1;
S.D=D;
S.channels='MEGMAG';
spm_opm_psd(S);
ylim([1,1e5])
xlim([0,100])