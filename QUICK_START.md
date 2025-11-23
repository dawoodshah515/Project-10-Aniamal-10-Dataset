# 🚀 Quick Start Guide - Animals-10 Classifier

## ⚡ Fast Track (3 Steps)

### 1️⃣ Install Dependencies
```bash
# Windows
setup.bat

# Linux/Mac
chmod +x setup.sh
./setup.sh
```

### 2️⃣ Train the Model
```bash
python train_model.py
```
⏱️ **Time**: 10-30 min (GPU) or 1-3 hours (CPU)  
🎯 **Target**: ≥70% accuracy

### 3️⃣ Run the System
```bash
# Terminal 1: Start Backend
cd backend
python app.py

# Terminal 2: Open Frontend
cd frontend
# Then open index.html in your browser
```

---

## 📋 What You Get

✅ **ML Model**: CNN-LSTM hybrid for 10 animal classes  
✅ **Backend API**: FastAPI with /predict endpoint  
✅ **Frontend**: Modern UI with glassmorphism & animations  
✅ **Documentation**: Complete guides & examples

---

## 🐾 Supported Animals

1. 🦋 Butterfly
2. 🐱 Cat
3. 🐔 Chicken
4. 🐄 Cow
5. 🐕 Dog
6. 🐘 Elephant
7. 🐴 Horse
8. 🐑 Sheep
9. 🕷️ Spider
10. 🐿️ Squirrel

---

## 🔗 Quick Links

- **API Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health
- **Frontend**: Open `frontend/index.html`

---

## 🆘 Common Issues

**"Model not found"**  
→ Run `python train_model.py` first

**"API connection failed"**  
→ Make sure backend is running on port 8000

**"Out of memory"**  
→ Reduce `BATCH_SIZE` in `train_model.py`

---

## 📚 Full Documentation

See [README.md](README.md) for complete details!
