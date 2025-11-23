# 🐾 Animals-10 RNN Classifier System

A complete end-to-end deep learning system for classifying animal images using an RNN-based CNN-LSTM hybrid model, with a FastAPI backend and a stunning modern web interface.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13+-orange.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## 🌟 Features

- **Advanced RNN Model**: CNN-LSTM hybrid architecture for image classification
- **10 Animal Classes**: Butterfly, Cat, Chicken, Cow, Dog, Elephant, Horse, Sheep, Spider, Squirrel
- **Modern Web Interface**: Glassmorphism effects, smooth animations, fully responsive
- **Real-time Predictions**: Fast inference with confidence scores
- **RESTful API**: FastAPI backend with automatic documentation
- **Drag & Drop Upload**: Intuitive file upload with preview
- **Beautiful Visualizations**: Animated probability bars and gradient effects

## 📁 Project Structure

```
Project 10/
├── Animals-10/                 # Dataset directory (10 animal classes)
├── backend/                    # FastAPI backend
│   ├── app.py                 # Main API application
│   └── requirements.txt       # Backend dependencies
├── frontend/                   # Web interface
│   ├── index.html             # Main HTML file
│   ├── styles.css             # Styling with animations
│   └── script.js              # JavaScript for interactions
├── data_preprocessing.py       # Data loading and preprocessing
├── model_architecture.py       # CNN-LSTM model definition
├── train_model.py             # Training script
├── evaluate_model.py          # Model evaluation
├── requirements.txt           # Main Python dependencies
├── animal_classifier.h5       # Trained model (after training)
├── saved_model/               # SavedModel format (after training)
└── README.md                  # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager
- (Optional) GPU with CUDA for faster training

### Installation

1. **Clone or navigate to the project directory**
   ```bash
   cd "f:/Data Analytics A to Z/PROJECTS   ............................ALL/Project 10"
   ```

2. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install backend dependencies**
   ```bash
   cd backend
   pip install -r requirements.txt
   cd ..
   ```

## 🎓 Training the Model

### Step 1: Verify Dataset

Ensure the `Animals-10` directory contains 10 subdirectories (one per class):
- butterfly
- cat
- chicken
- cow
- dog
- elephant
- horse
- sheep
- spider
- squirrel

### Step 2: Train the Model

```bash
python train_model.py
```

**Training Configuration:**
- Image size: 64x64 pixels
- Batch size: 32
- Max epochs: 100 (with early stopping)
- Optimizer: Adam
- Target accuracy: ≥70%

**What happens during training:**
- Data is automatically split into train/validation/test sets
- Model uses early stopping to prevent overfitting
- Best model is saved automatically
- Training history plot is generated
- Training info is saved to `training_info.txt`

**Expected Output:**
- `animal_classifier.h5` - Trained model in H5 format
- `saved_model/` - Model in SavedModel format
- `best_model.h5` - Best checkpoint during training
- `training_history.png` - Accuracy/loss plots
- `training_info.txt` - Training summary

**Training Time:**
- With GPU: ~10-30 minutes (depending on dataset size)
- With CPU: ~1-3 hours

### Step 3: Evaluate the Model (Optional)

```bash
python evaluate_model.py
```

This generates:
- `confusion_matrix.png` - Confusion matrix visualization
- `per_class_accuracy.png` - Per-class accuracy chart
- `evaluation_results.txt` - Detailed metrics

## 🖥️ Running the Application

### Step 1: Start the Backend API

```bash
cd backend
python app.py
```

The API will start at `http://localhost:8000`

**API Endpoints:**
- `GET /` - API information
- `GET /health` - Health check
- `GET /classes` - List all classes
- `POST /predict` - Upload image for classification
- `GET /docs` - Interactive API documentation (Swagger UI)

**Alternative: Using Uvicorn directly**
```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

### Step 2: Open the Frontend

1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```

2. Open `index.html` in your web browser:
   - **Option 1**: Double-click `index.html`
   - **Option 2**: Use a local server (recommended):
     ```bash
     python -m http.server 8080
     ```
     Then open `http://localhost:8080` in your browser

### Step 3: Use the Application

1. **Upload an Image**:
   - Drag and drop an animal image onto the upload area
   - Or click to browse and select a file
   - Supported formats: JPG, PNG, JPEG

2. **Classify**:
   - Click the "Classify Image" button
   - Wait for the AI model to process (usually < 1 second)

3. **View Results**:
   - See the predicted animal class with confidence score
   - View probability distribution across all classes
   - Add notes about the classification

## 🎨 Frontend Features

### Design Highlights
- **Glassmorphism Effects**: Modern frosted glass aesthetic
- **Animated Gradients**: Dynamic background with floating orbs
- **Smooth Transitions**: All interactions have fluid animations
- **Responsive Layout**: Works perfectly on mobile, tablet, and desktop
- **Dark Theme**: Easy on the eyes with vibrant accent colors

### Animations
- Fade-in effects on page load
- Slide-up animations for sections
- Hover effects on cards and buttons
- Animated probability bars
- Ripple effect on button clicks
- Smooth loader animation

## 🧠 Model Architecture

### CNN-LSTM Hybrid

```
Input (64x64x3 image)
    ↓
Reshape to sequence (64, 192)
    ↓
Reshape to image (64, 64, 3)
    ↓
CNN Encoder:
  - Conv2D(32) + BatchNorm + MaxPool + Dropout
  - Conv2D(64) + BatchNorm + MaxPool + Dropout
  - Conv2D(128) + BatchNorm + MaxPool + Dropout
    ↓
Reshape to sequences (8, 1024)
    ↓
LSTM Layers:
  - LSTM(128, return_sequences=True) + Dropout
  - LSTM(64) + Dropout
    ↓
Dense Classifier:
  - Dense(128) + BatchNorm + Dropout
  - Dense(10, softmax)
    ↓
Output (10 class probabilities)
```

**Key Features:**
- Combines spatial feature extraction (CNN) with sequence processing (LSTM)
- Batch normalization for stable training
- Dropout for regularization
- Adam optimizer with learning rate scheduling

## 📊 Performance

**Expected Metrics:**
- Test Accuracy: ≥70%
- Top-3 Accuracy: ≥85%
- Inference Time: < 100ms per image

**Per-Class Performance:**
Results vary by class. Check `per_class_accuracy.png` after evaluation.

## 🔧 Troubleshooting

### Issue: "Model file not found"
**Solution**: Make sure you've trained the model first using `python train_model.py`

### Issue: "API connection failed"
**Solution**: 
1. Verify the backend is running at `http://localhost:8000`
2. Check if port 8000 is available
3. Update `API_URL` in `frontend/script.js` if using a different port

### Issue: "CORS error in browser"
**Solution**: The backend already has CORS enabled. If issues persist, use a local server for the frontend instead of opening the HTML file directly.

### Issue: "Out of memory during training"
**Solution**: 
1. Reduce batch size in `train_model.py` (e.g., `BATCH_SIZE = 16`)
2. Set `MAX_IMAGES_PER_CLASS = 500` for faster training with less data
3. Use a machine with more RAM or GPU memory

### Issue: "Low accuracy after training"
**Solution**:
1. Train for more epochs (increase `EPOCHS` in `train_model.py`)
2. Try different hyperparameters (learning rate, dropout rate)
3. Ensure dataset quality and balance
4. Consider data augmentation adjustments

## 🎯 API Usage Examples

### Using cURL

```bash
# Health check
curl http://localhost:8000/health

# Get classes
curl http://localhost:8000/classes

# Predict (replace with your image path)
curl -X POST "http://localhost:8000/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@path/to/your/image.jpg"
```

### Using Python

```python
import requests

# Predict
url = "http://localhost:8000/predict"
files = {"file": open("path/to/image.jpg", "rb")}
response = requests.post(url, files=files)
result = response.json()

print(f"Predicted: {result['prediction']['class']}")
print(f"Confidence: {result['prediction']['confidence']:.2%}")
```

### Using JavaScript (Fetch API)

```javascript
const formData = new FormData();
formData.append('file', fileInput.files[0]);

const response = await fetch('http://localhost:8000/predict', {
    method: 'POST',
    body: formData
});

const data = await response.json();
console.log(data.prediction);
```

## 📝 Dataset Information

**Animals-10 Dataset**
- **Source**: Kaggle
- **Classes**: 10 animal categories
- **Images**: ~28,000 total images
- **Format**: JPG/PNG
- **License**: Check Kaggle dataset page

## 🛠️ Technology Stack

**Machine Learning:**
- TensorFlow 2.13+
- Keras
- NumPy
- scikit-learn
- Matplotlib

**Backend:**
- FastAPI
- Uvicorn
- Python-multipart
- Pillow

**Frontend:**
- HTML5
- CSS3 (with CSS Variables)
- Vanilla JavaScript (ES6+)
- Google Fonts (Poppins, Inter)

## 🎨 Customization

### Change API Port

Edit `backend/app.py`:
```python
uvicorn.run(app, host="0.0.0.0", port=YOUR_PORT)
```

And update `frontend/script.js`:
```javascript
const API_URL = 'http://localhost:YOUR_PORT';
```

### Modify Model Architecture

Edit `model_architecture.py` to change:
- Number of CNN layers
- LSTM units
- Dropout rates
- Dense layer sizes

### Customize UI Theme

Edit CSS variables in `frontend/styles.css`:
```css
:root {
    --primary-gradient: your-gradient;
    --accent-color: your-color;
    /* ... */
}
```

## 📄 License

This project is created for educational purposes. Please check the Animals-10 dataset license on Kaggle.

## 🤝 Contributing

This is a personal project, but suggestions and improvements are welcome!

## 📧 Support

If you encounter any issues:
1. Check the Troubleshooting section
2. Review the console/terminal output for error messages
3. Ensure all dependencies are installed correctly

## 🎉 Acknowledgments

- Animals-10 dataset from Kaggle
- TensorFlow and Keras teams
- FastAPI framework
- Google Fonts

---

**Built with ❤️ for Project #10**

*Powered by CNN-LSTM Deep Learning | Real-time AI Classification*
