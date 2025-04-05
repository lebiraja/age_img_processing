# Real-Time Age and Gender Detection

This project demonstrates **real-time age and gender detection** using deep learning and computer vision with Python and OpenCV. By utilizing pre-trained Caffe models, the application can accurately predict a person’s age range and gender from a live webcam feed or static image.

---

## 🚀 Features

- 🔍 Real-time face detection
- 🧠 Age group prediction
- 🧑‍🤝‍🧑 Gender classification
- 💡 Lightweight and easy to set up
- 📷 Webcam-based detection

---

## 📁 Project Structure

### 🔸 GitHub Repository Files

- **`test2.py`**  
  The main Python script that handles webcam input, face detection, and age & gender predictions.

- **`deploy_gender.prototxt`**  
  Model architecture file for gender classification.

- **`deploy_age.prototxt`**  
  Model architecture file for age prediction.

- **`deploy.prototxt`**  
  A general deployment file (may be used for additional configurations).

### 🔸 Google Drive Files

🔗 **[Download Pre-trained Caffe Models Here](https://drive.google.com/drive/folders/16J2QFyq8oqgdmMddJYRo9vupEu4oM0KW?usp=drive_link)**

- **`age_net.caffemodel`**  
  Pre-trained model for age group prediction.

- **`gender_net.caffemodel`**  
  Pre-trained model for gender classification.

> ⚠️ Place these models in the same directory as your script or update the file paths in `test2.py` accordingly.

---

## 🧠 Labels Used

### 📊 Age Groups:
```python
['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
```

### 🚻 Gender:
```python
['Male', 'Female']
```

---

## ⚙️ Installation & Requirements

Install the dependencies using pip:

```bash
pip install opencv-python numpy
```

Ensure you are using **Python 3.x**.

---

## ▶️ How to Run

1. **Clone the repository**:
```bash
git clone https://github.com/your-username/age_img_processing.git
cd age_img_processing

```

2. **Download the pre-trained models** from the Google Drive link and place them in the same directory.

3. **Run the script**:
```bash
python test2.py
```

A window should open, displaying your webcam feed with detected faces, and the predicted age & gender for each detected face.

---

## 🛠️ Customization

You can modify `test2.py` to:
- Load and process images instead of video
- Save results to a file
- Integrate with other applications (e.g., web apps)

---

## 📌 Notes
- This project uses the Caffe deep learning framework with OpenCV's DNN module.
- The models are relatively lightweight and run in real-time on most modern systems.

---

## 📄 License

This project is open-source and free to use under the MIT License.

---

## 🙌 Acknowledgements
- Pre-trained models from [Caffe Model Zoo](https://github.com/BVLC/caffe/wiki/Model-Zoo)
- OpenCV DNN module for handling deep learning models
