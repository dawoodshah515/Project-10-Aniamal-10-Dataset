// Animals-10 Classifier - Standalone Frontend with Embedded Demo
// Works without backend - perfect for GitHub Pages

const API_URL = 'http://localhost:8000'; // Fallback to local if available
const CLASSES = ['butterfly', 'cat', 'chicken', 'cow', 'dog',
    'elephant', 'horse', 'sheep', 'spider', 'squirrel'];

let selectedFile = null;

// DOM Elements
const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
const previewContainer = document.getElementById('previewContainer');
const imagePreview = document.getElementById('imagePreview');
const removeBtn = document.getElementById('removeBtn');
const predictBtn = document.getElementById('predictBtn');
const loaderOverlay = document.getElementById('loaderOverlay');
const resultsSection = document.getElementById('resultsSection');

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    setupEventListeners();
});

function setupEventListeners() {
    // Drop zone events
    dropZone.addEventListener('click', () => fileInput.click());
    dropZone.addEventListener('dragover', handleDragOver);
    dropZone.addEventListener('dragleave', handleDragLeave);
    dropZone.addEventListener('drop', handleDrop);

    // File input
    fileInput.addEventListener('change', handleFileSelect);

    // Buttons
    removeBtn.addEventListener('click', removeImage);
    predictBtn.addEventListener('click', classifyImage);
}

function handleDragOver(e) {
    e.preventDefault();
    dropZone.classList.add('drag-over');
}

function handleDragLeave(e) {
    e.preventDefault();
    dropZone.classList.remove('drag-over');
}

function handleDrop(e) {
    e.preventDefault();
    dropZone.classList.remove('drag-over');

    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFile(files[0]);
    }
}

function handleFileSelect(e) {
    const files = e.target.files;
    if (files.length > 0) {
        handleFile(files[0]);
    }
}

function handleFile(file) {
    // Validate file type
    if (!file.type.startsWith('image/')) {
        showNotification('Please select an image file', 'error');
        return;
    }

    // Validate file size (max 10MB)
    if (file.size > 10 * 1024 * 1024) {
        showNotification('File size must be less than 10MB', 'error');
        return;
    }

    selectedFile = file;
    displayPreview(file);
}

function displayPreview(file) {
    const reader = new FileReader();

    reader.onload = (e) => {
        imagePreview.src = e.target.result;
        previewContainer.style.display = 'block';
        predictBtn.disabled = false;
        dropZone.style.display = 'none';
    };

    reader.readAsDataURL(file);
}

function removeImage() {
    selectedFile = null;
    imagePreview.src = '';
    previewContainer.style.display = 'none';
    predictBtn.disabled = true;
    dropZone.style.display = 'block';
    fileInput.value = '';
    resultsSection.style.display = 'none';
}

async function classifyImage() {
    if (!selectedFile) return;

    // Show loader
    loaderOverlay.classList.add('active');

    try {
        // Try backend first
        const predictions = await tryBackendPrediction();
        displayResults(predictions);
    } catch (error) {
        // Fallback to client-side demo
        console.log('Backend unavailable, using client-side demo');
        const predictions = await clientSidePrediction();
        displayResults(predictions, true);
    } finally {
        loaderOverlay.classList.remove('active');
    }
}

async function tryBackendPrediction() {
    const formData = new FormData();
    formData.append('file', selectedFile);

    const response = await fetch(`${API_URL}/predict`, {
        method: 'POST',
        body: formData
    });

    if (!response.ok) {
        throw new Error('Backend unavailable');
    }

    const data = await response.json();
    return data.all_predictions;
}

async function clientSidePrediction() {
    // Simulate processing time
    await new Promise(resolve => setTimeout(resolve, 800));

    // Analyze image
    const imageData = await getImageData(selectedFile);
    const predictions = analyzeImage(imageData);

    return predictions;
}

async function getImageData(file) {
    return new Promise((resolve) => {
        const reader = new FileReader();
        reader.onload = (e) => {
            const img = new Image();
            img.onload = () => {
                const canvas = document.createElement('canvas');
                canvas.width = 64;
                canvas.height = 64;
                const ctx = canvas.getContext('2d');
                ctx.drawImage(img, 0, 0, 64, 64);
                const imageData = ctx.getImageData(0, 0, 64, 64);
                resolve(imageData);
            };
            img.src = e.target.result;
        };
        reader.readAsDataURL(file);
    });
}

function analyzeImage(imageData) {
    const pixels = imageData.data;
    let r = 0, g = 0, b = 0;

    // Calculate average colors
    for (let i = 0; i < pixels.length; i += 4) {
        r += pixels[i];
        g += pixels[i + 1];
        b += pixels[i + 2];
    }

    const pixelCount = pixels.length / 4;
    r /= pixelCount;
    g /= pixelCount;
    b /= pixelCount;

    const brightness = (r + g + b) / 3;
    const colorRange = Math.max(r, g, b) - Math.min(r, g, b);

    // Calculate variance for edge detection
    let variance = 0;
    for (let i = 0; i < pixels.length; i += 4) {
        const gray = (pixels[i] + pixels[i + 1] + pixels[i + 2]) / 3;
        variance += Math.pow(gray - brightness, 2);
    }
    variance = Math.sqrt(variance / pixelCount);

    // Score each class
    const scores = {};

    // Butterfly: colorful, high variance
    scores.butterfly = (colorRange > 60 ? 0.4 : 0) + (variance > 40 ? 0.2 : 0) + 0.05;

    // Cat: medium colors, varied patterns
    scores.cat = (brightness > 80 && brightness < 180 ? 0.3 : 0) + (variance > 45 ? 0.2 : 0) + 0.05;

    // Chicken: white/light colors
    scores.chicken = (brightness > 150 ? 0.4 : 0) + (colorRange < 70 ? 0.3 : 0) + 0.05;

    // Cow: high contrast
    scores.cow = (colorRange > 80 ? 0.3 : 0) + (variance > 50 ? 0.2 : 0) + 0.05;

    // Dog: brown/tan tones
    scores.dog = (r > g && g > b && brightness > 100 && brightness < 160 ? 0.5 : 0) + (r - b > 30 ? 0.2 : 0) + 0.05;

    // Elephant: gray tones
    scores.elephant = (Math.abs(r - g) < 20 && Math.abs(g - b) < 20 ? 0.4 : 0) + (brightness > 80 && brightness < 140 ? 0.3 : 0) + 0.05;

    // Horse: brown/dark
    scores.horse = (r > g && brightness < 130 ? 0.3 : 0) + (variance > 40 ? 0.2 : 0) + 0.05;

    // Sheep: white/cream, fluffy
    scores.sheep = (brightness > 160 ? 0.4 : 0) + (variance > 35 ? 0.3 : 0) + 0.05;

    // Spider: dark, high contrast
    scores.spider = (brightness < 100 ? 0.4 : 0) + (variance > 55 ? 0.2 : 0) + 0.05;

    // Squirrel: brown/orange
    scores.squirrel = (r > 120 && g > 80 && b < 100 ? 0.4 : 0) + (brightness > 100 && brightness < 150 ? 0.3 : 0) + 0.05;

    // Normalize
    const total = Object.values(scores).reduce((a, b) => a + b, 0);
    const predictions = CLASSES.map(cls => ({
        class: cls,
        confidence: scores[cls] / total
    }));

    return predictions.sort((a, b) => b.confidence - a.confidence);
}

function displayResults(predictions, isDemo = false) {
    // Show results section
    resultsSection.style.display = 'block';
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });

    // Top prediction
    const topPred = predictions[0];
    document.getElementById('predictedClass').textContent = topPred.class;
    document.getElementById('confidenceValue').textContent =
        `${(topPred.confidence * 100).toFixed(1)}%`;

    // Animate confidence bar
    const confidenceFill = document.getElementById('confidenceFill');
    setTimeout(() => {
        confidenceFill.style.width = `${topPred.confidence * 100}%`;
    }, 100);

    // All predictions
    const predictionsGrid = document.getElementById('predictionsGrid');
    predictionsGrid.innerHTML = '';

    predictions.forEach((pred, index) => {
        const item = document.createElement('div');
        item.className = 'prediction-item';
        item.style.setProperty('--index', index);

        item.innerHTML = `
            <span class="prediction-name">${pred.class}</span>
            <div class="prediction-prob">
                <span class="prob-value">${(pred.confidence * 100).toFixed(1)}%</span>
                <div class="prob-bar">
                    <div class="prob-fill" style="width: ${pred.confidence * 100}%"></div>
                </div>
            </div>
        `;

        predictionsGrid.appendChild(item);
    });

    // Show demo notification if using client-side
    if (isDemo) {
        showNotification('Using client-side demo (backend not available)', 'info');
    }
}

function showNotification(message, type = 'success') {
    // Create notification element
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.textContent = message;

    // Style
    Object.assign(notification.style, {
        position: 'fixed',
        top: '20px',
        right: '20px',
        padding: '1rem 1.5rem',
        borderRadius: '12px',
        color: 'white',
        fontWeight: '500',
        zIndex: '10000',
        animation: 'slideInRight 0.3s ease-out',
        background: type === 'error'
            ? 'linear-gradient(135deg, #f093fb 0%, #f5576c 100%)'
            : type === 'info'
                ? 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)'
                : 'linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)'
    });

    document.body.appendChild(notification);

    // Auto remove
    setTimeout(() => {
        notification.style.animation = 'slideOutRight 0.3s ease-out';
        setTimeout(() => notification.remove(), 300);
    }, 3000);
}

// Add animations
const style = document.createElement('style');
style.textContent = `
    @keyframes slideInRight {
        from {
            transform: translateX(100%);
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
            transform: translateX(100%);
            opacity: 0;
        }
    }
`;
document.head.appendChild(style);
