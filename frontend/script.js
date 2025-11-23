// Animals-10 Classifier - TensorFlow.js Version
// Uses MobileNet for real AI predictions in the browser!

const API_URL = 'http://localhost:8000';
const CLASSES = ['butterfly', 'cat', 'chicken', 'cow', 'dog',
    'elephant', 'horse', 'sheep', 'spider', 'squirrel'];

// Mapping MobileNet classes to our 10 animals
const ANIMAL_MAPPING = {
    'butterfly': ['butterfly', 'monarch', 'sulphur butterfly', 'lycaenid'],
    'cat': ['tabby', 'tiger cat', 'persian cat', 'siamese cat', 'egyptian cat', 'lynx'],
    'chicken': ['hen', 'cock', 'rooster'],
    'cow': ['ox', 'cow'],
    'dog': ['golden retriever', 'german shepherd', 'labrador', 'terrier', 'beagle', 'collie', 'dalmatian', 'poodle', 'pug', 'husky', 'chihuahua', 'malamute', 'doberman', 'rottweiler', 'boxer', 'bulldog', 'corgi', 'sheepdog'],
    'elephant': ['indian elephant', 'african elephant', 'tusker'],
    'horse': ['sorrel', 'zebra', 'horse'], // zebra often confused with horse in simple models
    'sheep': ['ram', 'ewe', 'bighorn', 'sheep'],
    'spider': ['tarantula', 'garden spider', 'black and gold garden spider', 'barn spider', 'wolf spider'],
    'squirrel': ['fox squirrel', 'squirrel']
};

let selectedFile = null;
let tfModel = null;
let isModelLoading = false;

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
    loadTFModel(); // Start loading model immediately
});

async function loadTFModel() {
    try {
        console.log('Loading MobileNet model...');
        isModelLoading = true;
        tfModel = await mobilenet.load();
        isModelLoading = false;
        console.log('MobileNet model loaded!');
    } catch (error) {
        console.error('Error loading model:', error);
        isModelLoading = false;
    }
}

function setupEventListeners() {
    dropZone.addEventListener('click', () => fileInput.click());
    dropZone.addEventListener('dragover', handleDragOver);
    dropZone.addEventListener('dragleave', handleDragLeave);
    dropZone.addEventListener('drop', handleDrop);
    fileInput.addEventListener('change', handleFileSelect);
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
    if (e.dataTransfer.files.length > 0) handleFile(e.dataTransfer.files[0]);
}

function handleFileSelect(e) {
    if (e.target.files.length > 0) handleFile(e.target.files[0]);
}

function handleFile(file) {
    if (!file.type.startsWith('image/')) {
        showNotification('Please select an image file', 'error');
        return;
    }
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

    loaderOverlay.classList.add('active');

    // Update loader text if model is still loading
    if (isModelLoading) {
        document.querySelector('.loader-text').textContent = "Loading AI Model...";
    }

    try {
        // Wait for model if needed
        if (!tfModel) {
            await loadTFModel();
        }
        document.querySelector('.loader-text').textContent = "Analyzing image...";

        // Try backend first (optional, for local dev)
        try {
            const predictions = await tryBackendPrediction();
            displayResults(predictions);
        } catch (e) {
            // Use TensorFlow.js
            console.log('Using TensorFlow.js model');
            const predictions = await tfPrediction();
            displayResults(predictions, true);
        }
    } catch (error) {
        console.error(error);
        showNotification('Classification failed', 'error');
    } finally {
        loaderOverlay.classList.remove('active');
        document.querySelector('.loader-text').textContent = "Analyzing image...";
    }
}

async function tryBackendPrediction() {
    const formData = new FormData();
    formData.append('file', selectedFile);
    const response = await fetch(`${API_URL}/predict`, {
        method: 'POST',
        body: formData
    });
    if (!response.ok) throw new Error('Backend unavailable');
    const data = await response.json();
    return data.all_predictions;
}

async function tfPrediction() {
    if (!tfModel) throw new Error('Model not loaded');

    // Get predictions from MobileNet (returns top 3 imagenet classes)
    const imgElement = document.getElementById('imagePreview');
    const rawPredictions = await tfModel.classify(imgElement, 5);

    console.log('Raw MobileNet predictions:', rawPredictions);

    // Map MobileNet classes to our 10 animals
    let scores = {};
    CLASSES.forEach(c => scores[c] = 0.01); // Base score

    rawPredictions.forEach(pred => {
        const className = pred.className.toLowerCase();
        const probability = pred.probability;

        // Check against our mapping
        for (const [animal, keywords] of Object.entries(ANIMAL_MAPPING)) {
            if (keywords.some(k => className.includes(k))) {
                scores[animal] += probability;
            }
        }
    });

    // Normalize scores
    let total = Object.values(scores).reduce((a, b) => a + b, 0);
    let finalPredictions = CLASSES.map(cls => ({
        class: cls,
        confidence: scores[cls] / total
    }));

    return finalPredictions.sort((a, b) => b.confidence - a.confidence);
}

function displayResults(predictions, isClientSide = false) {
    resultsSection.style.display = 'block';
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });

    const topPred = predictions[0];
    document.getElementById('predictedClass').textContent = topPred.class;
    document.getElementById('confidenceValue').textContent =
        `${(topPred.confidence * 100).toFixed(1)}%`;

    const confidenceFill = document.getElementById('confidenceFill');
    setTimeout(() => {
        confidenceFill.style.width = `${topPred.confidence * 100}%`;
    }, 100);

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

    if (isClientSide) {
        showNotification('Using Browser AI (TensorFlow.js)', 'info');
    }
}

function showNotification(message, type = 'success') {
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.textContent = message;

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
    setTimeout(() => {
        notification.style.animation = 'slideOutRight 0.3s ease-out';
        setTimeout(() => notification.remove(), 300);
    }, 3000);
}

// Add animations
const style = document.createElement('style');
style.textContent = `
    @keyframes slideInRight {
        from { transform: translateX(100%); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    @keyframes slideOutRight {
        from { transform: translateX(0); opacity: 1; }
        to { transform: translateX(100%); opacity: 0; }
    }
`;
document.head.appendChild(style);
