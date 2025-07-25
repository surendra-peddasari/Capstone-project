# Home Security System

This capstone project is a **Home Security System** that uses facial recognition and detection to enhance security at residential entrances or other sensitive locations.

## 🔧 Features

- Face detection using Haar cascades.
- Face recognition using pre-trained models (`trainer.yml`).
- Script-based implementation for real-time use.
- Includes a virtual environment setup and necessary scripts.

## 🗂️ Project Structure

```
Home-Security-System-main/
│
├── HSS.py                         # Main Python script for the security system
├── trainer.yml                    # Pre-trained face recognition model
├── haarcascade_frontalface_default.xml  # Haar cascade model for face detection
├── output.pdf                     # Project documentation
├── Scripts/                       # Scripts and executable files (e.g., activate, face_recognition.exe)
└── pyvenv.cfg                     # Python virtual environment config
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- OpenCV
- NumPy

### Setup

1. Clone or download this repository.
2. Set up the virtual environment (optional but recommended).
3. Install dependencies:
   ```bash
   pip install opencv-python numpy
   ```
4. Run the main script:
   ```bash
   python HSS.py
   ```

## 📄 Documentation

See `output.pdf` for a detailed explanation of the system design, components, and functionalities.

## 📬 Contact

Project by **Surendra Peddasari**  
For questions or collaboration, please reach out via GitHub.