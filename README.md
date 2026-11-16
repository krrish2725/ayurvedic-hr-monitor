# Ayurvedic Heart Rate Monitor

Webcam-based heart rate estimation with Ayurvedic pulse-style insights. The app uses remote photoplethysmography (rPPG) from a live camera feed, estimates BPM, then maps that rate to a dominant dosha (Vata, Pitta, or Kapha) with lifestyle recommendations.

**Repo:** [krrish2725/ayurvedic-hr-monitor](https://github.com/krrish2725/ayurvedic-hr-monitor)

> This is a wellness demo, not a medical device. Readings can be noisy and should not be used for diagnosis or clinical decisions.

## Features

- Live webcam feed in the browser
- Face detection with OpenCV Haar cascades
- Blink-based liveness check before signal collection
- rPPG from facial color changes (green channel, bandpass-filtered)
- Heart rate from FFT plus time-domain peak intervals
- Dosha badge, description, and recommendations after a short measurement
- PPG plot area on the results page

## How it works

1. Sit still in good lighting with your face in frame.
2. Click **Start Measurement** and blink when prompted (liveness).
3. The server samples mean RGB values from the face region at about 30 fps.
4. After ~20 seconds, the green channel is filtered (about 45–240 BPM).
5. BPM is estimated from frequency peaks and beat intervals, then mapped to Kapha / Vata / Pitta ranges.

Typical mapping used by the app:

| Approximate BPM | Dominant dosha |
|-----------------|----------------|
| Below ~70       | Kapha          |
| ~70–90          | Vata           |
| Above ~90       | Pitta          |

Pulse-quality labels in the analysis follow traditional Nadi Pariksha language (snake-like, frog-like, swan-like). That mapping is heuristic, based on heart rate alone.

## Tech stack

- **Backend:** Flask
- **Vision / signal:** OpenCV, NumPy, SciPy
- **Frontend:** HTML, Bootstrap 5, jQuery
- **Plots:** Matplotlib

## Project layout

```text
ppg/ayurvedic_hr_monitor/
├── app.py                 # Flask app, camera loop, HR + dosha logic
├── requirements.txt
├── templates/index.html
└── static/
    ├── css/style.css
    └── js/main.js
```

## Requirements

- Python 3.10+ (tested with 3.13)
- A webcam
- Windows, macOS, or Linux

Camera access is local to the machine running Flask. The browser only displays the MJPEG stream from `/video_feed`.

## Setup and run

From the repository root:

```bash
cd ppg/ayurvedic_hr_monitor
python -m pip install -r requirements.txt
python app.py
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000).

### Tips for a usable reading

- Use even indoor light; avoid strong backlight
- Keep your face centered and relatively still
- Allow the blink check to finish before the 20-second capture
- If BPM is missing or implausible, retry with better light and less motion

## API endpoints

| Method | Path                 | Purpose                                      |
|--------|----------------------|----------------------------------------------|
| GET    | `/`                  | Main UI                                      |
| GET    | `/video_feed`        | MJPEG camera stream                          |
| POST   | `/start_measurement` | Begin liveness check and signal capture      |
| POST   | `/stop_measurement`  | Stop capture, compute BPM and dosha insights |

## Disclaimer

Heart rate from a webcam is an estimate. Ayurvedic dosha suggestions here are informational only and are not a substitute for a qualified practitioner or medical care.
