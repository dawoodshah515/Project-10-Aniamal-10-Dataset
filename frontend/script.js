// ===================================
// Configuration
// ===================================
const API_URL = 'http://localhost:8000';
const CLASSES = ['butterfly', 'cat', 'chicken', 'cow', 'dog', 
                 'elephant', 'horse', 'sheep', 'spider', 'squirrel'];

// ===================================
// DOM Elements
// ===================================
const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
const previewContainer = document.getElementById('previewContainer');
const imagePreview = document.getElementById('imagePreview');
const removeBtn = document.getElementById('removeBtn');
const predictBtn = document.getElementById('predictBtn');
const loaderOverlay = document.getElementById('loaderOverlay');
const resultsSection = document.getElementById('resultsSection');
const predictedClass = document.getElementById('predictedClass');
const confidenceValue = document.getElementById('confidenceValue');
const confidenceFill = document.getElementById('confidenceFill');
const predictionsGrid = document.getElementById('predictionsGrid');
const notesTextarea = document.getElementById('notesTextarea');

// ===================================
// State
// ===================================
let selectedFile = null;

// ===================================
// Event Listeners
// ===================================

// Drop zone click
dropZone.addEventListener('click', () => {
    fileInput.click();
});

// File input change
fileInput.addEventListener('change', (e) => {
    handleFileSelect(e.target.files[0]);
});

// Drag and drop events
dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('drag-over');
});

dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('drag-over');
});

dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('drag-over');
    
    const file = e.dataTransfer.files[0];
    handleFileSelect(file);
});

// Remove button
removeBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    resetUpload();
});

// Predict button
predictBtn.addEventListener('click', async () => {
    if (selectedFile) {
        await classifyImage(selectedFile);
    }
});

// Notes textarea - smooth resize
notesTextarea.addEventListener('input', function() {
    this.style.height = 'auto';
    this.style.height = this.scrollHeight + 'px';
});

// ===================================
// File Handling Functions
// ===================================

function handleFileSelect(file) {
    // Validate file
    if (!file) return;
    
    if (!file.type.startsWith('image/')) {
        showError('Please select a valid image file (JPG, PNG, JPEG)');
        return;
    }
    
    if (file.size > 10 * 1024 * 1024) { // 10MB limit
        showError('File size must be less than 10MB');
        return;
    }
    
    // Store file
    selectedFile = file;
    
    // Show preview
    const reader = new FileReader();
    reader.onload = (e) => {
        imagePreview.src = e.target.result;
        previewContainer.style.display = 'block';
        predictBtn.disabled = false;
        
        // Hide results if showing
        resultsSection.style.display = 'none';
    };
    reader.readAsDataURL(file);
}

function resetUpload() {
    selectedFile = null;
    fileInput.value = '';
    previewContainer.style.display = 'none';
    imagePreview.src = '';
    predictBtn.disabled = true;
    resultsSection.style.display = 'none';
}

// ===================================
// API Functions
// ===================================

async function classifyImage(file) {
    try {
        // Show loader
        showLoader();
        
        // Create form data
        const formData = new FormData();
        formData.append('file', file);
        
        // Make API request
        const response = await fetch(`${API_URL}/predict`, {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error(`API error: ${response.status}`);
        }
        
        const data = await response.json();
        
        // Hide loader
        hideLoader();
        
        // Display results
        displayResults(data);
        
    } catch (error) {
        hideLoader();
        showError(`Classification failed: ${error.message}. Make sure the backend server is running.`);
        console.error('Error:', error);
    }
}

// ===================================
// UI Functions
// ===================================

function showLoader() {
    loaderOverlay.classList.add('active');
}

function hideLoader() {
    loaderOverlay.classList.remove('active');
}

function showError(message) {
    // Create error notification
    const errorDiv = document.createElement('div');
    errorDiv.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        background: linear-gradient(135deg, #f5576c 0%, #f093fb 100%);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 12px;
        box-shadow: 0 10px 30px rgba(245, 87, 108, 0.3);
        z-index: 10000;
        animation: slideInRight 0.3s ease-out;
        max-width: 400px;
        font-weight: 500;
    `;
    errorDiv.textContent = message;
    
    document.body.appendChild(errorDiv);
    
    // Remove after 5 seconds
    setTimeout(() => {
        errorDiv.style.animation = 'slideOutRight 0.3s ease-out';
        setTimeout(() => errorDiv.remove(), 300);
    }, 5000);
}

function displayResults(data) {
    if (!data.success) {
        showError('Classification failed');
        return;
    }
    
    const prediction = data.prediction;
    const allPredictions = data.all_predictions;
    
    // Update top prediction
    predictedClass.textContent = prediction.class;
    const confidencePercent = (prediction.confidence * 100).toFixed(1);
    confidenceValue.textContent = `${confidencePercent}%`;
    
    // Animate confidence bar
    setTimeout(() => {
        confidenceFill.style.width = `${confidencePercent}%`;
    }, 100);
    
    // Update all predictions grid
    predictionsGrid.innerHTML = '';
    allPredictions.forEach((pred, index) => {
        const item = createPredictionItem(pred, index);
        predictionsGrid.appendChild(item);
    });
    
    // Show results section with animation
    resultsSection.style.display = 'block';
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    
    // Add success notification
    showSuccessNotification(prediction.class, confidencePercent);
}

function createPredictionItem(prediction, index) {
    const item = document.createElement('div');
    item.className = 'prediction-item';
    item.style.setProperty('--index', index);
    
    const percent = (prediction.confidence * 100).toFixed(1);
    
    item.innerHTML = `
        <span class="prediction-name">${prediction.class}</span>
        <div class="prediction-prob">
            <span class="prob-value">${percent}%</span>
            <div class="prob-bar">
                <div class="prob-fill" style="width: 0"></div>
            </div>
        </div>
    `;
    
    // Animate probability bar
    setTimeout(() => {
        const fill = item.querySelector('.prob-fill');
        fill.style.width = `${percent}%`;
    }, 100 + (index * 50));
    
    return item;
}

function showSuccessNotification(className, confidence) {
    const notification = document.createElement('div');
    notification.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 12px;
        box-shadow: 0 10px 30px rgba(67, 233, 123, 0.3);
        z-index: 10000;
        animation: slideInRight 0.3s ease-out;
        font-weight: 600;
    `;
    notification.innerHTML = `
        <div style="display: flex; align-items: center; gap: 0.5rem;">
            <span style="font-size: 1.5rem;">✓</span>
            <div>
                <div>Classification Complete!</div>
                <div style="font-size: 0.9rem; opacity: 0.9; font-weight: 400;">
                    ${className} (${confidence}%)
                </div>
            </div>
        </div>
    `;
    
    document.body.appendChild(notification);
    
    // Remove after 4 seconds
    setTimeout(() => {
        notification.style.animation = 'slideOutRight 0.3s ease-out';
        setTimeout(() => notification.remove(), 300);
    }, 4000);
}

// ===================================
// Add notification animations to CSS dynamically
// ===================================
const style = document.createElement('style');
style.textContent = `
    @keyframes slideInRight {
        from {
            transform: translateX(400px);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
    
    @keyframes slideOutRight {
        from {
            transform: translateX(0);
            opacity: 1;
        }
        to {
            transform: translateX(400px);
            opacity: 0;
        }
    }
`;
document.head.appendChild(style);

// ===================================
// Initialize
// ===================================
console.log('Animals-10 Classifier initialized');
console.log('API URL:', API_URL);
console.log('Supported classes:', CLASSES);
