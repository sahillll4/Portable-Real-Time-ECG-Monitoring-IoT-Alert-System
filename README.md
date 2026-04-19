# 🚀 Real-Time Single-Lead ECG Monitoring System

End-to-end arrhythmia detection using **AD8232 + ESP32 + Machine Learning + IoT dashboard + Telegram alerts**

---
<img width="2378" height="1283" alt="image" src="https://github.com/user-attachments/assets/9624933e-d354-4465-b97a-53c3c95b1143" />

## 📌 Overview

This project implements a **real-time ECG monitoring system** that acquires cardiac signals, processes them using embedded signal processing and machine learning, and visualizes results on a live dashboard with alerting.

```
AD8232 → ESP32 → Python Backend → Firebase → Web Dashboard
                      ↓
               ML Classifier
                      ↓
              Telegram Alerts
```

- **Beat Classification:** AAMI standard → N (Normal), S, V  
- **Dataset:** MIT-BIH Arrhythmia Database  
- **Latency:** ~22–33 ms (real-time pipeline)

---
<img width="2048" height="1358" alt="image" src="https://github.com/user-attachments/assets/caedf97d-5dd5-4960-9d2a-f7e401b622dc" />

## 🧠 Key Features

- Real-time ECG acquisition (360 Hz)
- Pan-Tompkins QRS detection on ESP32
- ML-based arrhythmia classification (RF + SVM + CNN ensemble)
- Firebase-powered live dashboard
- Telegram-based alert system
- MATLAB-based algorithm prototyping (pre-deployment validation)

---

## 🏗️ Project Structure

```
ecg_project/
├── esp32/
│   ├── ecg_monitor.ino
│   └── config.h
│
├── phase2_ml_models/
│   ├── ecg_rf_model.pkl
│   ├── ecg_svm_model.pkl
│   ├── ecg_cnn_model.h5
│   └── ecg_scaler.pkl
│
├── phase3_matlab/
│   └── ecg_prototype.m
│
├── phase4_backend/
│   ├── realtime_backend.py
│   ├── requirements.txt
│   └── firebase_key.json
│
└── phase5_dashboard/
    └── index.html
```

---

## 🔌 Hardware Setup

| Component | Role |
|----------|------|
| AD8232 | ECG signal acquisition |
| ESP32 | Signal processing + Wi-Fi |
| Electrodes | RA, LA, RL placement |
| Buzzer + LEDs | Local alerts |
<img width="2114" height="947" alt="image" src="https://github.com/user-attachments/assets/ae0dccf3-9e81-433b-8c1a-423e23e883f7" />

### Wiring

| AD8232 | ESP32 |
|--------|------|
| OUTPUT | GPIO 34 |
| LO+ | GPIO 32 |
| LO- | GPIO 33 |
| 3.3V | 3.3V |
| GND | GND |

---

## 📊 MATLAB Prototype (Phase 3)

Before deploying to hardware, the signal processing pipeline was validated in MATLAB.

### ✔ Purpose

- Validate filtering & noise removal
- Implement Pan-Tompkins algorithm
- Verify QRS detection accuracy
- Compute heart rate (BPM)

---
<img width="1745" height="1238" alt="image" src="https://github.com/user-attachments/assets/95534394-f6f3-41a1-8a23-8ed0f82d48ef" />

### 🔁 Processing Pipeline

```
Raw ECG → Bandpass Filter → Differentiation → Squaring → Moving Window Integration → Peak Detection → BPM
```

---

### 💻 MATLAB Code (Core Implementation)

```matlab
clc; clear; close all;

fs = 360;
ecg = csvread('ecg_signal.csv');

t = (0:length(ecg)-1)/fs;

% Bandpass Filter (5–15 Hz)
[b, a] = butter(2, [5 15]/(fs/2), 'bandpass');
filtered_ecg = filtfilt(b, a, ecg);

% Differentiation
diff_ecg = diff(filtered_ecg);
diff_ecg(end+1) = diff_ecg(end);

% Squaring
squared_ecg = diff_ecg.^2;

% Moving Window Integration
window_size = round(0.150 * fs);
mwi_ecg = movmean(squared_ecg, window_size);

% Peak Detection
threshold = 0.6 * max(mwi_ecg);
[pks, locs] = findpeaks(mwi_ecg, ...
    'MinPeakHeight', threshold, ...
    'MinPeakDistance', round(0.2 * fs));

% BPM Calculation
rr_intervals = diff(locs) / fs;
bpm = 60 ./ rr_intervals;

fprintf('Average BPM: %.2f\n', mean(bpm));
```

---

### 📌 Outcome

- Accurate QRS detection  
- Stable BPM calculation  
- Validated pipeline before ESP32 deployment  

---
<img width="1686" height="1194" alt="image" src="https://github.com/user-attachments/assets/67c3e81d-ed02-4157-874d-b435f62e7213" />

## 🤖 Machine Learning (Phase 2)

Models trained on MIT-BIH dataset:

| Model | Strength |
|------|--------|
| Random Forest | High precision |
| SVM | Balanced performance |
| 1D-CNN | Best V-class detection |

### ✅ Deployment Strategy

- Ensemble voting (RF + SVM + CNN)
- CNN prioritized for ventricular arrhythmia detection

---

## ⚡ System Latency

| Stage | Latency |
|------|--------|
| Filtering | 8–10 ms |
| QRS Detection | 2–3 ms |
| ML Inference | 12–20 ms |
| **Total** | **~22–33 ms** |

---

## 🚨 Alert System

| Condition | Action |
|----------|--------|
| Ventricular beat detected | Immediate alert |
| 3+ abnormal beats | Alert |
| BPM > 110 (10 min) | High HR alert |
| BPM < 45 (10 min) | Low HR alert |

---

## 🖥️ Setup Instructions

### 1️⃣ ESP32

- Configure `config.h`
- Upload via Arduino IDE

---

### 2️⃣ Python Backend

```bash
cd phase4_backend
py -3.11 -m pip install -r requirements.txt
py -3.11 realtime_backend.py
```

---

### 3️⃣ Dashboard

- Open `index.html` in browser  
- Configure Firebase credentials  

---

### 4️⃣ Telegram Bot

- Create bot via `@BotFather`
- Add token + chat ID in config files  

---

## 📦 Dependencies

- Python: numpy, scipy, tensorflow, sklearn, firebase-admin
- Embedded: Arduino ESP32 libraries
- Dashboard: Firebase SDK + Chart.js

---

## 📚 References

- Moody GB, Mark RG — MIT-BIH Arrhythmia Database  
- Pan J, Tompkins WJ — QRS Detection Algorithm  
- ANSI/AAMI EC57 Standard  

---

## 💡 Why This Project Stands Out

- ✔ Real-time embedded + ML integration  
- ✔ End-to-end pipeline (hardware → cloud → UI)  
- ✔ MATLAB → ESP32 validation flow  
- ✔ Clinically relevant arrhythmia detection  

---

## 🔮 Future Improvements

- Improve S-class detection (data imbalance handling)
- Add wearable form factor
- Edge ML deployment on ESP32
- Mobile app integration

---

## 👨‍💻 Author

**Sahil**  
B.Tech Electronics & Communication

---

## ⭐ If you found this useful

Give it a ⭐ on GitHub

