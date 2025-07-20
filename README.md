# FurSure Backend – Cat vs Dog Classifier API

Welcome to the backend of **FurSure**, a lightweight image classification API that distinguishes between images of **cats** and **dogs** using a trained deep learning model.

> This backend serves predictions via a Flask API and is designed to be integrated with a React/Next.js frontend or any client that supports HTTP.

---

## 🔧 Tech Stack

- **Python 3.10+**
- **Flask** – For serving the API
- **TensorFlow / Keras** – For the trained CNN model
- **Flask-CORS** – Cross-origin support
- **Pillow** – For image preprocessing
- **Gunicorn** (for deployment on platforms like Render)

---

## 🚀 Features

- 📸 Accepts image uploads via POST requests.
- 🧠 Predicts if the image is a **cat** or **dog** using a pre-trained model.
- ⚡ Returns a JSON response with the label and confidence score.
- 🌐 Easily deployable on platforms like **Render**, **Railway**, or **Fly.io**.

---

## 📦 API Endpoint

### `POST /predict`

#### Request:

- `multipart/form-data` with a field named `image`

#### Example:

```bash
curl -X POST http://localhost:5000/predict \
  -F image=@/path/to/image.jpg
```

#### Response:

```json
{
  "prediction": "dog",
  "confidence": 0.91
}
```

---

## 📌 Known Challenges

- Training the model to generalize well on low-quality or ambiguous images.
- Optimizing for speed vs accuracy.
- Handling different image sizes & formats robustly.

---

## 📈 Future Improvements

- 📷 **Live Camera Feed** support.
- 🐾 Real-time mobile deployment via TensorFlow Lite.
- 🧠 Switch to more accurate models (like EfficientNet, MobileNetV3).
- 🔐 User-level analytics & prediction history (for login-based systems).
- ☁️ Auto-deployment CI/CD with GitHub Actions.

---

## 🤝 Contributing

Feel free to open issues or pull requests to contribute to this project!

---

## 👨‍💻 Author

**Devansh Tyagi**
[LinkedIn](https://www.linkedin.com/in/tyagi-devansh) | [GitHub](https://github.com/devanshtyagi26)
