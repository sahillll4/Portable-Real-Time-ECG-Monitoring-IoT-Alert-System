#include <Arduino.h>
#include <WiFi.h>
#include <HTTPClient.h>
#include "config.h"

// ─────────────────────────────────────────────────────────────
//  Pan-Tompkins state
// ─────────────────────────────────────────────────────────────
#define PT_WIN        54            // 150ms @ 360Hz (moving avg window)
#define RR_BUF        8             // last N RR intervals for BPM average
#define REFRACTORY    72            // 200ms in samples

static int   pt_buf[PT_WIN];       // moving-average buffer
static int   pt_idx      = 0;
static long  pt_sum      = 0;
static float pt_threshold= 0;
static int   pt_above_ct = 0;
static bool  pt_in_peak  = false;
static long  last_r      = 0;      // sample index of last R-peak
static long  rr_buf[RR_BUF];
static int   rr_idx      = 0;
static bool  rr_ready    = false;
static long  sample_idx  = 0;

// ─────────────────────────────────────────────────────────────
//  BPM + alert state
// ─────────────────────────────────────────────────────────────
static float current_bpm = 0;
static bool  alert_active = false;
static unsigned long last_firebase_ms = 0;
static unsigned long last_bpm_time    = 0;

// ─────────────────────────────────────────────────────────────
//  Baseline wander removal (1-pole HPF, fc≈0.5Hz)
// ─────────────────────────────────────────────────────────────
static float hpf_prev_in  = 0;
static float hpf_prev_out = 0;
// α = exp(-2π × 0.5 / 360) ≈ 0.9913
#define HPF_ALPHA 0.9913f

inline float hpf(float x) {
    float y = HPF_ALPHA * (hpf_prev_out + x - hpf_prev_in);
    hpf_prev_in  = x;
    hpf_prev_out = y;
    return y;
}

// ─────────────────────────────────────────────────────────────
//  Pan-Tompkins: feed one sample, returns true on R-peak detect
//  Also writes measured timing via micros() for paper latency
// ─────────────────────────────────────────────────────────────
bool pt_update(float filtered_sample, unsigned long *proc_us) {
    unsigned long t0 = micros();

    // Derivative + square
    static float prev = 0;
    float diff = filtered_sample - prev;
    prev = filtered_sample;
    float sq = diff * diff;

    // Moving average
    pt_sum -= pt_buf[pt_idx];
    int sq_int = (int)(sq * 1000);         // scale to int
    pt_buf[pt_idx] = sq_int;
    pt_sum += sq_int;
    pt_idx = (pt_idx + 1) % PT_WIN;
    float ma = (float)pt_sum / PT_WIN;

    // Adaptive threshold (decays slowly when no peak)
    pt_threshold = 0.9975f * pt_threshold + 0.0025f * ma;
    float thresh = pt_threshold * 0.5f;

    bool r_detected = false;

    if (ma > thresh) {
        pt_in_peak = true;
        pt_above_ct++;
    } else if (pt_in_peak) {
        pt_in_peak = false;
        long refractory_ok = sample_idx - last_r;
        if (refractory_ok > REFRACTORY && last_r > 0) {
            // Valid R-peak
            long rr = sample_idx - last_r;
            rr_buf[rr_idx % RR_BUF] = rr;
            rr_idx++;
            if (rr_idx >= RR_BUF) rr_ready = true;

            // Compute BPM from mean RR
            int count = rr_ready ? RR_BUF : rr_idx;
            long rr_sum = 0;
            for (int i = 0; i < count; i++)
                rr_sum += rr_buf[i];
            float rr_mean = (float)rr_sum / count;
            current_bpm = (rr_mean > 0) ? (360.0f * 60.0f / rr_mean) : 0;

            r_detected = true;
            last_bpm_time = millis();
        }
        last_r = sample_idx;
        pt_above_ct = 0;
    }

    *proc_us = micros() - t0;
    return r_detected;
}

// ─────────────────────────────────────────────────────────────
//  Alert: buzzer + LED
// ─────────────────────────────────────────────────────────────
void trigger_alert(bool on) {
    alert_active = on;
    digitalWrite(BUZZER_PIN,    on ? HIGH : LOW);
    digitalWrite(LED_RED_PIN,   on ? HIGH : LOW);
    digitalWrite(LED_GREEN_PIN, on ? LOW  : HIGH);
}

// ─────────────────────────────────────────────────────────────
//  Firebase push (non-blocking HTTP PATCH)
// ─────────────────────────────────────────────────────────────
void push_firebase(float bpm, const char* status) {
    if (WiFi.status() != WL_CONNECTED) return;

    HTTPClient http;
    String url = "https://";
    url += FIREBASE_HOST;
    url += "/ecg_live.json?auth=";
    url += FIREBASE_AUTH;

    String body = "{\"bpm\":";
    body += String(bpm, 1);
    body += ",\"status\":\"";
    body += status;
    body += "\",\"ts\":";
    body += String(millis());
    body += "}";

    http.begin(url);
    http.addHeader("Content-Type", "application/json");
    http.PATCH(body);                  // fire-and-forget
    http.end();
}

// ─────────────────────────────────────────────────────────────
//  Setup
// ─────────────────────────────────────────────────────────────
void setup() {
    Serial.begin(115200);

    pinMode(LO_PLUS_PIN,  INPUT);
    pinMode(LO_MINUS_PIN, INPUT);
    pinMode(BUZZER_PIN,   OUTPUT);
    pinMode(LED_RED_PIN,  OUTPUT);
    pinMode(LED_GREEN_PIN,OUTPUT);

    trigger_alert(false);

    // Connect Wi-Fi
    Serial.print("Connecting WiFi");
    WiFi.begin(WIFI_SSID, WIFI_PASS);
    int tries = 0;
    while (WiFi.status() != WL_CONNECTED && tries++ < 20) {
        delay(500); Serial.print(".");
    }
    Serial.println(WiFi.status() == WL_CONNECTED ? " OK" : " FAILED (offline mode)");

    Serial.println("READY");           // Python backend watches for this
}

// ─────────────────────────────────────────────────────────────
//  Main loop — hard-timed at 360Hz
// ─────────────────────────────────────────────────────────────
void loop() {
    static unsigned long next_sample_us = micros();
    unsigned long now_us = micros();

    if (now_us < next_sample_us) return;   // busy-wait remainder
    next_sample_us += SAMPLE_US;
    sample_idx++;

    // ── Lead-off detection ──────────────────────────────────
    if (digitalRead(LO_PLUS_PIN) || digitalRead(LO_MINUS_PIN)) {
        Serial.println("L");               // "L" = leads off
        return;
    }

    // ── Sample ADC ──────────────────────────────────────────
    int raw = analogRead(ECG_PIN);         // 0–4095 (12-bit)

    // ── Filter ──────────────────────────────────────────────
    float filtered = hpf((float)raw);

    // ── Pan-Tompkins ────────────────────────────────────────
    unsigned long proc_us = 0;
    bool r_peak = pt_update(filtered, &proc_us);

    // ── Serial output: "raw,bpm,proc_us\n" ─────────────────
    // Python backend parses this stream
    Serial.print(raw);
    Serial.print(',');
    Serial.print((int)current_bpm);
    Serial.print(',');
    Serial.println(proc_us);              // ← paper latency measurement

    // ── Local alert (immediate, no network needed) ───────────
    if (r_peak && current_bpm > 0) {
        bool abnormal = (current_bpm < BPM_LOW || current_bpm > BPM_HIGH);
        trigger_alert(abnormal);
    }

    // ── BPM timeout: no beat for 3 sec → flatline alert ─────
    if (millis() - last_bpm_time > 3000 && last_bpm_time > 0) {
        trigger_alert(true);
    }

    // ── Firebase push every 5 seconds ───────────────────────
    unsigned long now_ms = millis();
    if (now_ms - last_firebase_ms >= FIREBASE_INTERVAL_MS) {
        last_firebase_ms = now_ms;
        const char* status = alert_active ? "ALERT" : "Normal";
        push_firebase(current_bpm, status);   // runs in ~100ms
    }
}