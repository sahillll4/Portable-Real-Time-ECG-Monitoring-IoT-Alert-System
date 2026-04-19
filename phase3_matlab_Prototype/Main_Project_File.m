%  PHASE-1 : ECG ARRHYTHMIA DETECTOR (MATLAB)
clc; clear; close all;

%% ======= STEP-1 : LOAD ECG FROM .MAT FILE =======

mat_file_name = '114m.mat';   
duration_sec = 60;            
disp(['Loading file: ' mat_file_name]);

try
    data = load(mat_file_name);

    if isfield(data, 'val')
        raw = data.val(1,:);  
    else
        error('Variable "val" not found in .mat file');
    end

    % Auto-detect sampling frequency if stored
    if isfield(data, 'Fs')
        fs = data.Fs;
    else
        fs = 360;  % fallback for MIT-BIH cases
    end

    % Limit to selected duration
    raw = raw(1:min(length(raw), fs * duration_sec));

catch ME
    disp('❌ Could not load ECG file');
    disp(ME.message);
    return;
end

tm = (0:length(raw)-1) / fs;
disp('✅ File loaded successfully');

%% ======= STEP-2 : PRE-PROCESSING =======
disp('Filtering ECG...');

% Bandpass (5-15Hz)
[b_bp, a_bp] = butter(4, [5 15] / (fs/2), 'bandpass');
ecg_bp = filtfilt(b_bp, a_bp, raw);

% Notch Filter at 50 Hz
notch = 50;
wo = notch / (fs/2);
bw = wo / 35;
[b_notch, a_notch] = iirnotch(wo, bw);
ecg_notch = filtfilt(b_notch, a_notch, ecg_bp);

% Smoothing (moving average 20ms)
win = round(0.02 * fs);
ecg_smooth = movmean(ecg_notch, win);

disp('✅ Pre-processing complete');

% Plot pre-processing
figure;
subplot(3,1,1); plot(tm, raw); title('Raw ECG'); ylabel('mV');
subplot(3,1,2); plot(tm, ecg_bp,'k'); title('Bandpass (5-15 Hz)');
subplot(3,1,3); plot(tm, ecg_smooth,'g'); title('After Notch + Smoothing');
xlabel('Time (s)');

%% ======= STEP-3 : PAN-TOMPKINS QRS DETECTION =======
disp('Detecting R-peaks using Pan-Tompkins...');

% Derivative
diff_sig = diff(ecg_smooth);

% Squaring
sq_sig = diff_sig.^2;

% Moving Window Integration (150 ms)
win_int = round(0.150 * fs);
int_sig = movsum(sq_sig, win_int);

% Dynamic threshold
th = 0.25 * max(int_sig);

[pks, locs] = findpeaks(int_sig, 'MinPeakHeight', th, ...
                                    'MinPeakDistance', round(0.25*fs));
tm_int = (0:length(int_sig)-1) / fs;
figure;
plot(tm, ecg_smooth); hold on;
plot(tm_int(locs), ecg_smooth(locs), 'ro','MarkerFaceColor','r','MarkerSize',6);
title('R-Peak Detection (Pan-Tompkins)');
xlabel('Time (s)'); ylabel('Amplitude');
grid on;
disp(['✅ R-peaks detected: ' num2str(length(locs))]);

%% ======= STEP-4 : HEART RATE, IRREGULARITY & CLASSIFICATION =======
rr = diff(locs)/fs;         % RR intervals in seconds
bpm = 60 ./ rr;             % BPM per beat
avg_bpm = mean(bpm);

disp('--------------------------------------------');
disp(['✅ Average Heart Rate  : ' num2str(avg_bpm, '%.2f') ' BPM']);
disp(['✅ Total Beats         : ' num2str(length(locs))]);

% Heart Rhythm Classification
if avg_bpm < 60
    rhythm = 'Bradycardia';
elseif avg_bpm > 100
    rhythm = 'Tachycardia';
else
    rhythm = 'Normal Sinus Rhythm';
end
disp(['✅ Classification       : ' rhythm]);

% Detailed beat information
for i = 1:length(bpm)
    t = tm_int(locs(i+1));
    fprintf('Beat @ %.2fs → %.1f BPM\n', t, bpm(i));
end

%% ======= STEP-5 : BPM HISTOGRAM =======
figure;
histogram(bpm, 20);
title('Heart Rate Distribution (BPM)');
xlabel('BPM'); ylabel('Count');
grid on;

disp('✅ Phase-1 Complete ✅');
