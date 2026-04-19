"""
ECG Real-Time Backend — Phase 4
ESP32 → Filter → Pan-Tompkins → ML → Firebase → Telegram
"""

import serial
import time, joblib, numpy as np, pywt, warnings
from collections import deque
from scipy.signal import butter, filtfilt, iirnotch
import firebase_admin
from firebase_admin import credentials, db
import requests, threading

warnings.filterwarnings('ignore')

# ── CONFIG ────────────────────────────────────────────────────
SERIAL_PORT   = "COM3"          # Windows: COM3 / Linux: /dev/ttyUSB0
BAUD_RATE     = 115200
FS            = 360

MODEL_DIR     = "./"            # folder containing this script
RF_MODEL      = MODEL_DIR + "ecg_rf_model.pkl"
SVM_MODEL     = MODEL_DIR + "ecg_svm_model.pkl"
CNN_MODEL     = MODEL_DIR + "ecg_cnn_model.h5"
SCALER        = MODEL_DIR + "ecg_scaler.pkl"

# ── CREDENTIALS (ADD YOUR OWN BEFORE RUNNING) ──
FIREBASE_URL  = "ADD_YOUR_FIREBASE_DATABASE_URL"
FIREBASE_KEY  = MODEL_DIR + "ADD_FIREBASE_SERVICE_ACCOUNT_JSON"

TG_TOKEN      = "ADD_TELEGRAM_BOT_TOKEN"
TG_CHAT_ID    = "ADD_TELEGRAM_CHAT_ID"

WIN_PRE       = 90
WIN_POST      = 110
WAVEFORM_BUF  = 5 * FS         # 5 sec of samples for dashboard

# ── LOAD MODELS ───────────────────────────────────────────────
print("Loading ML models...")
try:
    rf     = joblib.load(RF_MODEL)
    svm    = joblib.load(SVM_MODEL)
    scaler = joblib.load(SCALER)
    from tensorflow import keras
    cnn    = keras.models.load_model(CNN_MODEL)
    print("✅ All models loaded")
except Exception as e:
    print(f"❌ Model load failed: {e}")
    print("   Copy .pkl and .h5 files from Colab to this folder.")
    raise SystemExit(1)

# ── FIREBASE ──────────────────────────────────────────────────
print("Connecting to Firebase...")
try:
    cred = credentials.Certificate(FIREBASE_KEY)
    firebase_admin.initialize_app(cred, {'databaseURL': FIREBASE_URL})
    ecg_ref    = db.reference('/ecg_live')
    alerts_ref = db.reference('/alerts')
    print("✅ Firebase connected")
except Exception as e:
    print(f"❌ Firebase failed: {e}")
    raise SystemExit(1)

# ── TELEGRAM ──────────────────────────────────────────────────
def tg_send(msg):
    try:
        requests.post(
            f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
            json={"chat_id": TG_CHAT_ID, "text": msg},
            timeout=5
        )
    except Exception:
        pass

def tg_async(msg):
    threading.Thread(target=tg_send, args=(msg,), daemon=True).start()

# ── FILTERS ───────────────────────────────────────────────────
def make_filters():
    b_hp, a_hp = butter(4, 0.5/(FS/2), btype='high')
    b_lp, a_lp = butter(4, 40/(FS/2),  btype='low')
    b_n,  a_n  = iirnotch(50/(FS/2), 30)
    return (b_hp,a_hp), (b_lp,a_lp), (b_n,a_n)

HPF, LPF, NOTCH = make_filters()

def apply_filters(buf):
    s = filtfilt(*HPF, buf)
    s = filtfilt(*NOTCH, s)
    s = filtfilt(*LPF, s)
    return s

# ── PAN-TOMPKINS (Python, on filtered buffer) ─────────────────
def find_r_peaks(sig):
    diff = np.diff(sig, prepend=sig[0])
    sq   = diff ** 2
    win  = int(0.15 * FS)
    ma   = np.convolve(sq, np.ones(win)/win, mode='same')
    thr  = np.mean(ma) * 0.5
    ref  = int(0.2 * FS)
    peaks, in_p, last = [], False, 0
    for i, v in enumerate(ma):
        if v > thr and not in_p and (i - last) > ref:
            in_p = True; last = i
        elif v < thr and in_p:
            lo = max(0, last-15); hi = min(len(sig), last+15)
            peaks.append(lo + int(np.argmax(sig[lo:hi])))
            in_p = False
    return np.array(peaks)

# ── FEATURE EXTRACTION (matches Phase 2) ─────────────────────
def wavelet_features(beat, wavelet='db4', level=4):
    coeffs = pywt.wavedec(beat, wavelet, level=level)
    feats  = []
    for c in coeffs:
        feats += [np.mean(np.abs(c)), np.std(c), np.sum(c**2)]
    return feats

def extract_features(beat, rr_pre, rr_post, rr_mean):
    ratio = rr_pre / rr_post if rr_post > 0 else 1.0
    r_amp = float(np.max(beat))
    qrs_w = int(np.sum(np.abs(beat) > 0.5 * r_amp))
    wt    = wavelet_features(beat)
    return [rr_pre, rr_post, rr_mean, ratio, r_amp, qrs_w] + wt

# ── ML INFERENCE ─────────────────────────────────────────────
LABEL = {0:'N', 1:'S', 2:'V'}

def classify_beat(beat, rr_pre, rr_post, rr_mean):
    feats  = np.array(extract_features(beat, rr_pre, rr_post, rr_mean)).reshape(1,-1)
    fscale = scaler.transform(feats)
    # Ensemble vote: RF + SVM + CNN
    rf_pred  = rf.predict(fscale)[0]
    svm_pred = svm.predict(fscale)[0]
    cnn_prob = cnn.predict(beat.reshape(1,200,1), verbose=0)[0]
    cnn_pred = LABEL[int(np.argmax(cnn_prob))]
    votes = [rf_pred, svm_pred, cnn_pred]
    final = max(set(votes), key=votes.count)
    return final

# ── FIREBASE PUSH (async) ─────────────────────────────────────
_fb_lock = threading.Lock()

def push_firebase(bpm, beat_class, status, waveform):
    def _push():
        with _fb_lock:
            try:
                ecg_ref.set({
                    'bpm':        round(float(bpm), 1),
                    'beat_class': beat_class,
                    'status':     status,
                    'waveform':   waveform[-360:],   # last 1 sec
                    'ts':         int(time.time()*1000)
                })
            except Exception:
                pass
    threading.Thread(target=_push, daemon=True).start()

def push_alert(beat_class, bpm):
    def _push():
        try:
            alerts_ref.push({
                'type': f'{beat_class}_beat',
                'bpm':  round(float(bpm),1),
                'time': time.strftime('%H:%M:%S')
            })
        except Exception:
            pass
    threading.Thread(target=_push, daemon=True).start()

# ── MAIN LOOP ─────────────────────────────────────────────────
def main():
    print(f"🔌 Opening serial port {SERIAL_PORT} @ {BAUD_RATE}...")
    try:
        import serial as _serial
        ser = _serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=2)
        time.sleep(2)
        print("✅ Serial connected")
    except Exception as e:
        print(f"❌ Serial failed: {e}")
        print("   Available ports:")
        import serial.tools.list_ports
        for p in serial.tools.list_ports.comports():
            print(f"   {p}")
        raise SystemExit(1)

    # Wait for ESP32 READY
    for _ in range(50):
        line = ser.readline().decode('utf-8','ignore').strip()
        if line == 'READY':
            break

    print("▶️  Streaming ECG... Press Ctrl+C to stop\n")
    print(f"{'Time':10} | {'BPM':6} | {'Beat':5} | {'Status':10} | Latency")
    print("─" * 55)

    raw_buf  = deque(maxlen=5*FS)      # 5 sec raw
    filt_buf = deque(maxlen=5*FS)
    wave_buf = deque(maxlen=WAVEFORM_BUF)

    rr_history   = deque(maxlen=8)
    beat_count   = 0
    abn_consec   = 0
    last_r_idx   = 0
    last_fb_time = 0
    current_bpm  = 0.0
    current_cls  = 'N'

    # Latency tracking (paper table)
    lat_filter, lat_pt, lat_ml = [], [], []

    sample_idx = 0

    try:
        while True:
            line = ser.readline().decode('utf-8','ignore').strip()
            if not line or line == 'L':
                continue

            # Parse: "raw,bpm_esp,proc_us"
            parts = line.split(',')
            if len(parts) < 2:
                continue
            try:
                raw_val  = int(parts[0])
                bpm_esp  = int(parts[1])
            except ValueError:
                continue

            sample_idx += 1
            raw_buf.append(raw_val)
            wave_buf.append(raw_val)

            # Need at least 2 sec to filter
            if len(raw_buf) < 2 * FS:
                continue

            # ── Filter ───────────────────────────────────────
            t0  = time.perf_counter()
            sig = apply_filters(np.array(raw_buf))
            t_filter = (time.perf_counter() - t0) * 1000

            # ── Pan-Tompkins ─────────────────────────────────
            t1 = time.perf_counter()
            peaks = find_r_peaks(sig)
            t_pt = (time.perf_counter() - t1) * 1000

            # Fix: cap last_r_idx to buffer bounds
            buf_last_r = last_r_idx % len(sig)
            new_peaks = peaks[peaks > buf_last_r]
            if len(new_peaks) < 2:
                new_peaks = peaks[-4:]
            if len(new_peaks) < 2:
                continue

            for pi, r_idx in enumerate(new_peaks[:-1]):
                if r_idx < WIN_PRE or r_idx + WIN_POST > len(sig):
                    continue

                rr_pre  = (r_idx - new_peaks[pi-1]) if pi > 0 else (rr_history[-1] if rr_history else FS)
                rr_post = new_peaks[pi+1] - r_idx
                rr_history.append(rr_post)
                rr_mean = float(np.mean(rr_history)) if rr_history else float(FS)
                current_bpm = 360.0 * 60.0 / rr_mean if rr_mean > 0 else 0

                beat = sig[r_idx - WIN_PRE : r_idx + WIN_POST]

                # ── ML Inference ─────────────────────────────
                t2 = time.perf_counter()
                current_cls = classify_beat(beat, rr_pre, rr_post, rr_mean)
                t_ml = (time.perf_counter() - t2) * 1000

                lat_filter.append(t_filter)
                lat_pt.append(t_pt)
                lat_ml.append(t_ml)
                total_lat = t_filter + t_pt + t_ml

                beat_count  += 1
                is_abnormal  = (current_cls in ['S','V'])
                abn_consec   = (abn_consec + 1) if is_abnormal else 0
                status       = "ALERT" if is_abnormal else "Normal"

                ts = time.strftime('%H:%M:%S')
                print(f"{ts:10} | {current_bpm:5.1f} | {current_cls:5} | {status:10} | {total_lat:.1f}ms")

                # Telegram: V-beat or 3+ consecutive abnormal
                if current_cls == 'V' or abn_consec >= 3:
                    msg = (f"🚨 ECG ALERT\n"
                           f"Beat: {current_cls}\nBPM: {current_bpm:.1f}\n"
                           f"Time: {ts}\nConsecutive abnormal: {abn_consec}")
                    tg_async(msg)
                    push_alert(current_cls, current_bpm)

                last_r_idx = int(new_peaks[-1])

            # ── Firebase every 5 sec ──────────────────────────
            now = time.time()
            if now - last_fb_time >= 5:
                last_fb_time = now
                push_firebase(current_bpm, current_cls,
                              "ALERT" if abn_consec > 0 else "Normal",
                              list(wave_buf))

            # ── Print latency table every 100 beats ──────────
            if beat_count > 0 and beat_count % 100 == 0:
                print(f"\n📊 LATENCY REPORT (Paper Table):")
                print(f"   Filter:   {np.mean(lat_filter):.2f} ms avg")
                print(f"   Pan-Tomp: {np.mean(lat_pt):.2f} ms avg")
                print(f"   ML Infer: {np.mean(lat_ml):.2f} ms avg")
                print(f"   Total:    {np.mean(lat_filter)+np.mean(lat_pt)+np.mean(lat_ml):.2f} ms avg\n")

    except KeyboardInterrupt:
        print("\n\n⏹ Stopped.")
        if lat_ml:
            print(f"\n📊 FINAL LATENCY TABLE (use in paper):")
            print(f"   Filter:   {np.mean(lat_filter):.2f} ms")
            print(f"   Pan-Tomp: {np.mean(lat_pt):.2f} ms")
            print(f"   ML Infer: {np.mean(lat_ml):.2f} ms")
            print(f"   Total:    {np.mean(lat_filter)+np.mean(lat_pt)+np.mean(lat_ml):.2f} ms")
        ser.close()

if __name__ == '__main__':
    main()
