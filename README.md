

---

```markdown
# Real-Time Age and Gender Detection

This project performs **real-time age and gender detection** using deep learning models in Python and OpenCV. It utilizes pre-trained Caffe models to predict the gender and age of individuals from a webcam feed or a static image.

## 🚀 Features

- Real-time video feed processing
- Age prediction from face detection
- Gender classification from facial features
- Easy to use and extend
- Lightweight and fast

---

## 📁 File Structure

### GitHub Repository Files:

- `test2.py`  
  Main script that performs real-time age and gender detection using the webcam. It loads the necessary models and processes video frames to detect faces and predict age and gender.

- `deploy_gender.prototxt`  
  Network configuration file for the gender detection model.

- `deploy_age.prototxt`  
  Network configuration file for the age detection model.

- `deploy.prototxt`  
  General deployment file (optional use or can be a shared config across both networks depending on your implementation).

### Google Drive Files:

🔗 **Download Pre-trained Caffe Models:**  
[Click here to access the models](https://drive.google.com/drive/folders/16J2QFyq8oqgdmMddJYRo9vupEu4oM0KW?usp=drive_link)

- `age_net.caffemodel`  
  Pre-trained Caffe model for age prediction.

- `gender_net.caffemodel`  
  Pre-trained Caffe model for gender classification.

> 💡 Note: These models are not stored in the GitHub repo due to size limits. Please download them from the Drive link and place them in the same directory as your script or update the paths accordingly.

---

## 🧠 Age & Gender Labels

### Age Groups:
```
['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
```

### Gender:
```
['Male', 'Female']
```

---

## ⚙️ Requirements

Install the required packages before running the script:

```bash
pip install opencv-python numpy
```

Ensure you have Python 3.x installed.

---

## 🧪 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name
   ```

2. Download the Caffe models from the [Google Drive link](https://drive.google.com/drive/folders/16J2QFyq8oqgdmMddJYRo9vupEu4oM0KW?usp=drive_link) and place them in the same folder.

3. Run the detection script:
   ```bash
   python test2.py
   ```

The webcam window will open and begin predicting age and gender for any detected faces.

---

## 🧰 Customization

- You can modify the model paths in `test2.py` if you place the model files in a different directory.
- To use a static image instead of webcam, replace the `cv2.VideoCapture(0)` with the path to your image and modify the loop accordingly.

---

## 📷 Example Output

The application will show the live webcam feed with bounding boxes around detected faces and display their predicted age and gender above them.

---

## 📌 Credits

- [OpenCV](https://opencv.org/)
- Caffe deep learning models trained by [Levi and Hassner, 2015](https://talhassner.github.io/home/projects/Adience/Adience-data.html)
- Dataset: [Adience Benchmark](https://www.openu.ac.il/home/hassner/Adience/data.html)

---

## 🛡️ License

This project is for educational purposes. Please cite appropriate datasets and models if used in research or development.

---

## 🙌 Contribution

Feel free to fork, improve, and make pull requests. Suggestions and issues are welcome!

```

---

Let me know if you want a markdown version to copy directly or need a README with images, badges, or enhancements!
