function ECG_GUI_Live
    %% =========== ECG LIVE DISPLAY GUI ============
    clc; close all;
    fig = figure('Name','ECG Live Monitor','NumberTitle','off','Position',[200 100 900 600]);

    ax1 = subplot(2,1,1,'Parent',fig);
    title(ax1,'ECG Signal (Live)');
    xlabel(ax1,'Time (s)'); ylabel('Amplitude');

    ax2 = subplot(2,1,2,'Parent',fig);
    title(ax2,'Filtered ECG + R-peaks');
    xlabel(ax2,'Time (s)'); ylabel('Amplitude');

    txt_hr = uicontrol('Style','text','Position',[720 500 150 30], ...
        'String','HR: ---','FontSize',11);
    txt_avg = uicontrol('Style','text','Position',[720 460 150 30], ...
        'String','Avg HR: ---','FontSize',11);
    txt_class = uicontrol('Style','text','Position',[720 420 150 30], ...
        'String','Status: ---','FontSize',11);

    btn_load = uicontrol('Style','pushbutton','String','Load ECG','Position',[720 350 150 35], ...
        'FontSize',10,'Callback',@loadECG);
    btn_start = uicontrol('Style','pushbutton','String','Start','Position',[720 300 150 35], ...
        'FontSize',10,'Callback',@startLive,'Enable','off');
    btn_stop = uicontrol('Style','pushbutton','String','Stop','Position',[720 250 150 35], ...
        'FontSize',10,'Callback',@stopLive,'Enable','off');

    tmr = timer('ExecutionMode','fixedRate','Period',0.12,'TimerFcn',@updateECG);

    ecg = []; fs = 360; idx = 1;
    win_len = fs * 3;

    function loadECG(~,~)
        [file,path] = uigetfile('*.mat','Choose MIT-BIH .mat file');
        if isequal(file,0), return; end

        data = load(fullfile(path,file));
        if isfield(data,'val')
            ecg = data.val(1,:);
        else
            errordlg('MAT missing variable "val"'); return;
        end
        
        if isfield(data,'Fs'); fs = data.Fs; else fs = 360; end
        idx = 1;

        set(btn_start,'Enable','on');
        disp(['✅ Loaded: ' file]);
    end

    function startLive(~,~)
        if isempty(ecg)
            errordlg('Load a file first'); return;
        end
        set(btn_stop,'Enable','on');
        start(tmr);
    end

    function stopLive(~,~)
        stop(tmr);
    end

    function updateECG(~,~)
        if idx + win_len > length(ecg)
            stop(tmr); return;
        end
        sig = ecg(idx:idx+win_len);
        t = (0:length(sig)-1)/fs;

        % Preprocessing
        [b,a] = butter(4,[5 15]/(fs/2),'bandpass'); f1 = filtfilt(b,a,sig);
        [bn,an] = iirnotch(50/(fs/2), (50/(fs/2))/35); f2 = filtfilt(bn,an,f1);
        f3 = movmean(f2,round(0.02*fs));

        % Pan-Tompkins
        d = diff(f3); sq = d.^2;
        int = movsum(sq,round(0.15*fs));
        [pks,locs] = findpeaks(int,'MinPeakHeight',0.25*max(int),'MinPeakDistance',round(0.25*fs));

        if length(locs) > 1
            rr = diff(locs)/fs;
            bpm = 60 ./ rr;
            avg_bpm = mean(bpm);

            if avg_bpm < 60
                status = 'Bradycardia';
            elseif avg_bpm > 100
                status = 'Tachycardia';
            else
                status = 'Normal';
            end

            set(txt_hr,'String',['HR: ' num2str(bpm(end),'%.1f')]);
            set(txt_avg,'String',['Avg HR: ' num2str(avg_bpm,'%.1f')]);
            set(txt_class,'String',['Status: ' status]);
        end

        % Update plots
        plot(ax1,t,sig); grid(ax1,'on');
        plot(ax2,t,f3); hold(ax2,'on');
        plot(ax2,t(locs),f3(locs),'ro','MarkerFaceColor','r'); hold(ax2,'off'); grid(ax2,'on');

        idx = idx + round(0.12*fs);
    end
end
