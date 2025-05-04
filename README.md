Here's a **README.md** file for your fatigue analysis project using facial expression, eye movement, and rPPG-based metrics:

---

# Fatigue Analyzer using DeepFace and rPPG

This Python-based real-time fatigue analyzer utilizes a webcam to detect a person's **fatigue level**, **heart rate**, **HRV**, and **emotional state** using facial expressions and eye analysis. It provides a **Total Meditation Score (TMS)** to estimate a user's mental relaxation and fatigue.

## Features

* 🎭 Emotion detection using [DeepFace](https://github.com/serengil/deepface)
* 👀 Eye-based fatigue estimation (blink rate & eye closure)
* ❤️ rPPG signal processing to estimate heart rate and HRV
* 📊 Live HRV plotting (optional)
* 🎯 Total Meditation Score (TMS) in the format `Fatigue-HRV-Expression`
* 🧠 Facial relaxation scoring
* 👁️ Face and eye detection using Haar cascades

---


## Installation

### 🔧 Requirements

* Python 3.7+
* OpenCV
* NumPy
* Matplotlib
* DeepFace

### 📦 Install dependencies

```bash
pip install opencv-python numpy matplotlib deepface
```

---

## Usage

### 🔍 Run the Analyzer

```bash
python fatigue_analyzer.py
```

### 🎮 Controls

* Press **`q`** to quit the analyzer
* Press **`p`** to toggle HRV plotting window

---

## Output Metrics

* **Emotion**: Real-time dominant emotion detected from face
* **Fatigue**: Based on blink rate, eye closure, and emotion
* **HR / HRV**: Heart rate and variability estimated via rPPG signal (simulated in this version)
* **TMS (Total Meditation Score)**: Composite score in format `x-y-z`

  * `x`: Fatigue score (0–10)
  * `y`: HRV score (0–10)
  * `z`: Relaxation score from facial expression (0–10)

---

## Limitations

* Current HR/HRV values are **simulated** using sine/cosine functions.
* Eye detection may be less accurate with glasses or low light.
* DeepFace emotion detection may require GPU for faster performance.

---

## Future Enhancements

* Replace simulated rPPG with real rPPG signal processing (e.g., CHROM/ICA methods)
* Improve fatigue detection using facial landmarks (e.g., dlib)
* Add drowsiness alerts
* Store session history and generate reports

---

## License

MIT License

---

## Author

Created by Sana Agarkar
Feel free to contribute, report bugs, or suggest improvements.


