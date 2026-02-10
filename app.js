// ============================================
// TITANIC BINARY CLASSIFIER WITH SIGMOID GATE
// TensorFlow.js Browser Application
// ============================================

// Global variables for data and model
let trainData = null;
let testData = null;
let trainFeatures = null;
let trainLabels = null;
let validationFeatures = null;
let validationLabels = null;
let testFeatures = null;
let testPassengerIds = null;
let model = null;
let predictions = null;
let validationPredictions = null;
let featureNames = [];
let scaler = { mean: {}, std: {} };

// ============================================
// DATA SCHEMA (SWAP THIS FOR OTHER DATASETS)
// ============================================
const TARGET = 'Survived';  // Binary target: 0 or 1
const FEATURES = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'];
const IDENTIFIER = 'PassengerId';  // Exclude from features

// ============================================
// 1. DATA LOADING AND INSPECTION
// ============================================

async function loadData() {
    try {
        showStatus('Loading data...', 'info');
        
        const trainFile = document.getElementById('trainFile').files[0];
        const testFile = document.getElementById('testFile').files[0];
        
        if (!trainFile) {
            alert('Please select training data file (train.csv)');
            return;
        }
        
        // Load training data
        trainData = await parseCSV(trainFile);
        console.log('Train data loaded:', trainData.length, 'rows');
        
        // Load test data if provided
        if (testFile) {
            testData = await parseCSV(testFile);
            console.log('Test data loaded:', testData.length, 'rows');
        }
        
        // Display preview
        displayDataPreview(trainData);
        
        // Display statistics
        displayDataStats(trainData);
        
        // Visualize survival by Sex and Pclass
        visualizeSurvivalDistribution(trainData);
        
        // Enable preprocessing button
        document.getElementById('preprocessBtn').disabled = false;
        
        showStatus('Data loaded successfully!', 'success');
        
    } catch (error) {
        showStatus('Error loading data: ' + error.message, 'error');
        console.error(error);
    }
}

// Parse CSV file with proper comma/quote handling
function parseCSV(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = function(e) {
            try {
                const text = e.target.result;
                const lines = text.split('\n').filter(line => line.trim());
                
                if (lines.length === 0) {
                    reject(new Error('CSV file is empty'));
                    return;
                }
                
                // Parse header
                const headers = parseCSVLine(lines[0]);
                
                // Parse data rows
                const data = [];
                for (let i = 1; i < lines.length; i++) {
                    const values = parseCSVLine(lines[i]);
                    if (values.length === headers.length) {
                        const row = {};
                        headers.forEach((header, index) => {
                            row[header] = values[index];
                        });
                        data.push(row);
                    }
                }
                
                resolve(data);
            } catch (error) {
                reject(error);
            }
        };
        reader.onerror = () => reject(new Error('Failed to read file'));
        reader.readAsText(file);
    });
}

// Parse single CSV line handling quotes and commas correctly
function parseCSVLine(line) {
    const result = [];
    let current = '';
    let inQuotes = false;
    
    for (let i = 0; i < line.length; i++) {
        const char = line[i];
        
        if (char === '"') {
            inQuotes = !inQuotes;
        } else if (char === ',' && !inQuotes) {
            result.push(current.trim());
            current = '';
        } else {
            current += char;
        }
    }
    
    result.push(current.trim());
    return result;
}

// Display first 10 rows of data
function displayDataPreview(data) {
    const preview = document.getElementById('dataPreview');
    const displayRows = data.slice(0, 10);
    
    if (displayRows.length === 0) {
        preview.innerHTML = '<p>No data to display</p>';
        return;
    }
    
    const headers = Object.keys(displayRows[0]);
    
    let html = '<div class="info-box"><strong>Data Preview (First 10 Rows)</strong></div>';
    html += '<div class="preview-table"><table><thead><tr>';
    headers.forEach(h => html += `<th>${h}</th>`);
    html += '</tr></thead><tbody>';
    
    displayRows.forEach(row => {
        html += '<tr>';
        headers.forEach(h => {
            const value = row[h] || '';
            html += `<td>${value}</td>`;
        });
        html += '</tr>';
    });
    
    html += '</tbody></table></div>';
    preview.innerHTML = html;
}

// Display dataset statistics
function displayDataStats(data) {
    const preview = document.getElementById('dataPreview');
    const shape = `${data.length} rows × ${Object.keys(data[0]).length} columns`;
    
    // Calculate missing values
    const headers = Object.keys(data[0]);
    const missing = {};
    headers.forEach(h => {
        const missingCount = data.filter(row => !row[h] || row[h] === '').length;
        missing[h] = ((missingCount / data.length) * 100).toFixed(1) + '%';
    });
    
    let statsHtml = '<div class="info-box">';
    statsHtml += `<strong>Dataset Shape:</strong> ${shape}<br>`;
    statsHtml += '<strong>Missing Values:</strong> ';
    statsHtml += Object.entries(missing)
        .filter(([k, v]) => parseFloat(v) > 0)
        .map(([k, v]) => `${k}: ${v}`)
        .join(', ') || 'None';
    statsHtml += '</div>';
    
    preview.innerHTML += statsHtml;
}

// Visualize survival distribution by Sex and Pclass
function visualizeSurvivalDistribution(data) {
    // Survival by Sex
    const sexData = { male: { survived: 0, died: 0 }, female: { survived: 0, died: 0 } };
    data.forEach(row => {
        const sex = (row.Sex || '').toLowerCase().trim();
        const survived = parseInt(row.Survived);
        if (sex === 'male') {
            if (survived === 1) sexData.male.survived++;
            else sexData.male.died++;
        } else if (sex === 'female') {
            if (survived === 1) sexData.female.survived++;
            else sexData.female.died++;
        }
    });
    
    // Format data for grouped bar chart (Sex)
    const sexChartData = [
        { index: 0, value: sexData.male.died, label: 'Male Died' },
        { index: 1, value: sexData.male.survived, label: 'Male Survived' },
        { index: 2, value: sexData.female.died, label: 'Female Died' },
        { index: 3, value: sexData.female.survived, label: 'Female Survived' },
    ];
    
    tfvis.render.barchart(
        { name: 'Survival by Sex', tab: 'Data Inspection' },
        sexChartData,
        { 
            width: 500, 
            height: 300,
            xLabel: 'Category',
            yLabel: 'Count'
        }
    );
    
    // Survival by Pclass
    const pclassData = { 
        1: { survived: 0, died: 0 }, 
        2: { survived: 0, died: 0 }, 
        3: { survived: 0, died: 0 } 
    };
    
    data.forEach(row => {
        const pclass = parseInt(row.Pclass);
        const survived = parseInt(row.Survived);
        if ([1, 2, 3].includes(pclass)) {
            if (survived === 1) pclassData[pclass].survived++;
            else pclassData[pclass].died++;
        }
    });
    
    // Format data for grouped bar chart (Pclass)
    const pclassChartData = [
        { index: 0, value: pclassData[1].died, label: 'Class 1 Died' },
        { index: 1, value: pclassData[1].survived, label: 'Class 1 Survived' },
        { index: 2, value: pclassData[2].died, label: 'Class 2 Died' },
        { index: 3, value: pclassData[2].survived, label: 'Class 2 Survived' },
        { index: 4, value: pclassData[3].died, label: 'Class 3 Died' },
        { index: 5, value: pclassData[3].survived, label: 'Class 3 Survived' },
    ];
    
    tfvis.render.barchart(
        { name: 'Survival by Pclass', tab: 'Data Inspection' },
        pclassChartData,
        { 
            width: 600, 
            height: 300,
            xLabel: 'Category',
            yLabel: 'Count'
        }
    );
}

// ============================================
// 2. DATA PREPROCESSING
// ============================================

function preprocessData() {
    try {
        showStatus('Preprocessing data...', 'info');
        
        const addFamilySize = document.getElementById('addFamilySize').checked;
        
        // Separate features and labels for training
        const processed = processDataset(trainData, true, addFamilySize);
        trainFeatures = processed.features;
        trainLabels = processed.labels;
        featureNames = processed.featureNames;
        
        // Split into train/validation (80/20 stratified)
        const split = stratifiedSplit(trainFeatures, trainLabels, 0.8);
        trainFeatures = split.trainX;
        trainLabels = split.trainY;
        validationFeatures = split.valX;
        validationLabels = split.valY;
        
        // Process test data if available
        if (testData) {
            const testProcessed = processDataset(testData, false, addFamilySize);
            testFeatures = testProcessed.features;
            testPassengerIds = testProcessed.passengerIds;
        }
        
        // Display preprocessing info
        const info = document.getElementById('preprocessInfo');
        info.innerHTML = `
            <div class="info-box">
                <strong>Preprocessing Complete!</strong><br>
                Train samples: ${trainFeatures.shape[0]}<br>
                Validation samples: ${validationFeatures.shape[0]}<br>
                ${testData ? `Test samples: ${testFeatures.shape[0]}<br>` : ''}
                Features: ${featureNames.length} (${featureNames.join(', ')})<br>
                Feature shape: ${trainFeatures.shape[1]} dimensions
            </div>
        `;
        
        console.log('Train features shape:', trainFeatures.shape);
        console.log('Train labels shape:', trainLabels.shape);
        console.log('Feature names:', featureNames);
        
        // Enable model building
        document.getElementById('buildBtn').disabled = false;
        showStatus('Preprocessing completed successfully!', 'success');
        
    } catch (error) {
        showStatus('Error during preprocessing: ' + error.message, 'error');
        console.error(error);
    }
}

// Process dataset: imputation, standardization, encoding
function processDataset(data, isTraining, addFamilySize) {
    const features = [];
    const labels = [];
    const passengerIds = [];
    const localFeatureNames = [];
    
    // Calculate statistics for imputation (only on training data)
    if (isTraining) {
        const ages = data.map(r => parseFloat(r.Age)).filter(a => !isNaN(a));
        const fares = data.map(r => parseFloat(r.Fare)).filter(f => !isNaN(f));
        const embarked = data.map(r => r.Embarked).filter(e => e && e.trim());
        
        scaler.ageMedian = median(ages);
        scaler.fareMedian = median(fares);
        scaler.embarkedMode = mode(embarked);
        
        // Calculate mean and std for Age and Fare
        scaler.mean.Age = mean(ages);
        scaler.std.Age = std(ages);
        scaler.mean.Fare = mean(fares);
        scaler.std.Fare = std(fares);
    }
    
    data.forEach(row => {
        const featureVector = [];
        
        // Pclass (numeric)
        const pclass = parseInt(row.Pclass) || 3;
        
        // Sex (encode)
        const sex = (row.Sex || 'male').toLowerCase();
        const sexMale = sex === 'male' ? 1 : 0;
        const sexFemale = sex === 'female' ? 1 : 0;
        
        // Age (impute and standardize)
        let age = parseFloat(row.Age);
        if (isNaN(age)) age = scaler.ageMedian;
        const ageStd = (age - scaler.mean.Age) / scaler.std.Age;
        
        // SibSp, Parch (numeric)
        const sibSp = parseInt(row.SibSp) || 0;
        const parch = parseInt(row.Parch) || 0;
        
        // Fare (impute and standardize)
        let fare = parseFloat(row.Fare);
        if (isNaN(fare)) fare = scaler.fareMedian;
        const fareStd = (fare - scaler.mean.Fare) / scaler.std.Fare;
        
        // Embarked (impute and encode)
        let embarked = (row.Embarked || scaler.embarkedMode).trim();
        if (!embarked) embarked = scaler.embarkedMode;
        const embarkedS = embarked === 'S' ? 1 : 0;
        const embarkedC = embarked === 'C' ? 1 : 0;
        const embarkedQ = embarked === 'Q' ? 1 : 0;
        
        // Add features
        featureVector.push(pclass);  // Pclass
        featureVector.push(sexMale);  // Sex_male
        featureVector.push(sexFemale);  // Sex_female
        featureVector.push(ageStd);  // Age (standardized)
        featureVector.push(sibSp);  // SibSp
        featureVector.push(parch);  // Parch
        featureVector.push(fareStd);  // Fare (standardized)
        featureVector.push(embarkedS);  // Embarked_S
        featureVector.push(embarkedC);  // Embarked_C
        featureVector.push(embarkedQ);  // Embarked_Q
        
        // Optional: FamilySize and IsAlone
        if (addFamilySize) {
            const familySize = sibSp + parch + 1;
            const isAlone = familySize === 1 ? 1 : 0;
            featureVector.push(familySize);
            featureVector.push(isAlone);
        }
        
        features.push(featureVector);
        
        // Labels (only for training data)
        if (isTraining) {
            labels.push(parseInt(row[TARGET]) || 0);
        }
        
        // PassengerId (for test data)
        if (!isTraining && row[IDENTIFIER]) {
            passengerIds.push(row[IDENTIFIER]);
        }
    });
    
    // Build feature names (only once)
    if (isTraining) {
        localFeatureNames.push('Pclass', 'Sex_male', 'Sex_female', 'Age', 'SibSp', 'Parch', 
                               'Fare', 'Embarked_S', 'Embarked_C', 'Embarked_Q');
        if (addFamilySize) {
            localFeatureNames.push('FamilySize', 'IsAlone');
        }
    }
    
    return {
        features: tf.tensor2d(features),
        labels: isTraining ? tf.tensor2d(labels, [labels.length, 1]) : null,
        passengerIds: passengerIds,
        featureNames: localFeatureNames
    };
}

// Stratified train/validation split
function stratifiedSplit(features, labels, trainRatio) {
    const data = [];
    const labelsArray = labels.arraySync();
    const featuresArray = features.arraySync();
    
    for (let i = 0; i < labelsArray.length; i++) {
        data.push({ features: featuresArray[i], label: labelsArray[i][0] });
    }
    
    // Shuffle
    data.sort(() => Math.random() - 0.5);
    
    // Separate by class
    const class0 = data.filter(d => d.label === 0);
    const class1 = data.filter(d => d.label === 1);
    
    // Split each class
    const train0 = class0.slice(0, Math.floor(class0.length * trainRatio));
    const val0 = class0.slice(Math.floor(class0.length * trainRatio));
    const train1 = class1.slice(0, Math.floor(class1.length * trainRatio));
    const val1 = class1.slice(Math.floor(class1.length * trainRatio));
    
    const trainData = [...train0, ...train1].sort(() => Math.random() - 0.5);
    const valData = [...val0, ...val1].sort(() => Math.random() - 0.5);
    
    return {
        trainX: tf.tensor2d(trainData.map(d => d.features)),
        trainY: tf.tensor2d(trainData.map(d => [d.label])),
        valX: tf.tensor2d(valData.map(d => d.features)),
        valY: tf.tensor2d(valData.map(d => [d.label]))
    };
}

// Helper functions
function median(arr) {
    const sorted = [...arr].sort((a, b) => a - b);
    const mid = Math.floor(sorted.length / 2);
    return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

function mean(arr) {
    return arr.reduce((a, b) => a + b, 0) / arr.length;
}

function std(arr) {
    const m = mean(arr);
    const variance = arr.reduce((sum, val) => sum + Math.pow(val - m, 2), 0) / arr.length;
    return Math.sqrt(variance);
}

function mode(arr) {
    const freq = {};
    arr.forEach(v => freq[v] = (freq[v] || 0) + 1);
    return Object.keys(freq).reduce((a, b) => freq[a] > freq[b] ? a : b);
}

// ============================================
// 3. MODEL BUILDING WITH SIGMOID GATE
// ============================================

function buildModel() {
    try {
        showStatus('Building model with Sigmoid Gate...', 'info');
        
        const inputShape = trainFeatures.shape[1];
        
        // Create custom model with Sigmoid Gate layer
        const input = tf.input({ shape: [inputShape] });
        
        // SIGMOID GATE LAYER - learns feature importance
        // Each feature gets a sigmoid gate value (0 to 1)
        const gateWeights = tf.layers.dense({
            units: inputShape,
            activation: 'sigmoid',
            name: 'sigmoid_gate',
            useBias: false  // No bias for pure feature masking
        }).apply(input);
        
        // Element-wise multiplication (Hadamard product)
        // Masks input features by gate values
        const gatedInput = tf.layers.multiply({ name: 'gated_input' })
            .apply([input, gateWeights]);
        
        // Main classification network
        const hidden = tf.layers.dense({
            units: 16,
            activation: 'relu',
            name: 'hidden_layer'
        }).apply(gatedInput);
        
        const output = tf.layers.dense({
            units: 1,
            activation: 'sigmoid',
            name: 'output_layer'
        }).apply(hidden);
        
        // Create model
        model = tf.model({ inputs: input, outputs: output });
        
        // Compile model
        model.compile({
            optimizer: 'adam',
            loss: 'binaryCrossentropy',
            metrics: ['accuracy']
        });
        
        // Display model summary
        const info = document.getElementById('modelInfo');
        info.innerHTML = '<div class="info-box"><strong>Model Architecture:</strong></div>';
        
        tfvis.show.modelSummary(
            { name: 'Model Summary', tab: 'Model' },
            model
        );
        
        console.log('Model built successfully');
        model.summary();
        
        // Enable training button
        document.getElementById('trainBtn').disabled = false;
        showStatus('Model built successfully with Sigmoid Gate!', 'success');
        
    } catch (error) {
        showStatus('Error building model: ' + error.message, 'error');
        console.error(error);
    }
}

// ============================================
// 4. MODEL TRAINING
// ============================================

async function trainModel() {
    try {
        showStatus('Training model...', 'info');
        document.getElementById('trainBtn').disabled = true;
        
        const epochs = 50;
        const batchSize = 32;
        
        // Training callbacks
        const callbacks = {
            onEpochEnd: async (epoch, logs) => {
                console.log(`Epoch ${epoch + 1}/${epochs}: loss = ${logs.loss.toFixed(4)}, ` +
                           `accuracy = ${logs.acc.toFixed(4)}, ` +
                           `val_loss = ${logs.val_loss.toFixed(4)}, ` +
                           `val_accuracy = ${logs.val_acc.toFixed(4)}`);
            },
            ...tfvis.show.fitCallbacks(
                { name: 'Training Performance', tab: 'Training' },
                ['loss', 'acc', 'val_loss', 'val_acc'],
                { callbacks: ['onEpochEnd'] }
            )
        };
        
        // Train the model
        const history = await model.fit(trainFeatures, trainLabels, {
            epochs: epochs,
            batchSize: batchSize,
            validationData: [validationFeatures, validationLabels],
            callbacks: callbacks,
            shuffle: true
        });
        
        const info = document.getElementById('trainingInfo');
        const finalLoss = history.history.loss[history.history.loss.length - 1];
        const finalAcc = history.history.acc[history.history.acc.length - 1];
        const finalValLoss = history.history.val_loss[history.history.val_loss.length - 1];
        const finalValAcc = history.history.val_acc[history.history.val_acc.length - 1];
        
        info.innerHTML = `
            <div class="info-box">
                <strong>Training Complete!</strong><br>
                Final Training Loss: ${finalLoss.toFixed(4)}<br>
                Final Training Accuracy: ${(finalAcc * 100).toFixed(2)}%<br>
                Final Validation Loss: ${finalValLoss.toFixed(4)}<br>
                Final Validation Accuracy: ${(finalValAcc * 100).toFixed(2)}%
            </div>
        `;
        
        // Display feature importance
        displayFeatureImportance();
        
        // Enable evaluation button
        document.getElementById('evalBtn').disabled = false;
        if (testData) {
            document.getElementById('predictBtn').disabled = false;
        }
        
        showStatus('Training completed successfully!', 'success');
        
    } catch (error) {
        showStatus('Error during training: ' + error.message, 'error');
        console.error(error);
        document.getElementById('trainBtn').disabled = false;
    }
}

// ============================================
// 5. FEATURE IMPORTANCE (SIGMOID GATE)
// ============================================

function displayFeatureImportance() {
    try {
        // Get the sigmoid gate layer
        const gateLayer = model.getLayer('sigmoid_gate');
        const gateWeights = gateLayer.getWeights()[0];
        const gateValues = gateWeights.arraySync();
        
        // Calculate mean importance for each feature across all gate neurons
        const importance = [];
        for (let i = 0; i < featureNames.length; i++) {
            const values = gateValues.map(row => row[i]);
            const meanValue = mean(values);
            importance.push({ name: featureNames[i], value: meanValue });
        }
        
        // Sort by importance
        importance.sort((a, b) => b.value - a.value);
        
        // Display
        const container = document.getElementById('featureImportance');
        let html = '<div class="feature-importance">';
        html += '<p><strong>Feature Importance Values</strong> (higher = more important):</p>';
        
        importance.forEach(feat => {
            const percentage = (feat.value * 100).toFixed(1);
            html += `
                <div class="feature-bar">
                    <div class="feature-name">${feat.name}</div>
                    <div class="feature-bar-bg">
                        <div class="feature-bar-fill" style="width: ${percentage}%"></div>
                    </div>
                    <div class="feature-value">${feat.value.toFixed(3)}</div>
                </div>
            `;
        });
        
        html += '</div>';
        container.innerHTML = html;
        
        console.log('Feature Importance:', importance);
        
    } catch (error) {
        console.error('Error displaying feature importance:', error);
    }
}

// ============================================
// 6. MODEL EVALUATION AND METRICS
// ============================================

async function evaluateModel() {
    try {
        showStatus('Evaluating model...', 'info');
        
        // Get predictions on validation set
        validationPredictions = model.predict(validationFeatures);
        const predArray = await validationPredictions.array();
        const labelArray = await validationLabels.array();
        
        // Calculate ROC curve and AUC
        const rocData = calculateROC(predArray.map(p => p[0]), labelArray.map(l => l[0]));
        const auc = calculateAUC(rocData);
        
        // Plot ROC curve
        tfvis.render.linechart(
            { name: 'ROC Curve', tab: 'Evaluation' },
            { values: rocData.map(p => ({ x: p.fpr, y: p.tpr })) },
            {
                xLabel: 'False Positive Rate',
                yLabel: 'True Positive Rate',
                width: 500,
                height: 400
            }
        );
        
        // Display AUC
        const metricsInfo = document.getElementById('metricsInfo');
        metricsInfo.innerHTML = `
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-value">${auc.toFixed(4)}</div>
                    <div class="metric-label">ROC-AUC Score</div>
                </div>
            </div>
        `;
        
        // Show threshold slider
        document.getElementById('thresholdSliderContainer').style.display = 'block';
        
        // Add threshold slider listener
        const slider = document.getElementById('thresholdSlider');
        slider.addEventListener('input', function() {
            const threshold = this.value / 100;
            document.getElementById('thresholdValue').textContent = threshold.toFixed(2);
            updateConfusionMatrix(predArray.map(p => p[0]), labelArray.map(l => l[0]), threshold);
        });
        
        // Initial confusion matrix at 0.5 threshold
        updateConfusionMatrix(predArray.map(p => p[0]), labelArray.map(l => l[0]), 0.5);
        
        showStatus('Evaluation completed!', 'success');
        
    } catch (error) {
        showStatus('Error during evaluation: ' + error.message, 'error');
        console.error(error);
    }
}

// Calculate ROC curve points
function calculateROC(predictions, labels) {
    const thresholds = [...new Set(predictions)].sort((a, b) => b - a);
    const rocPoints = [];
    
    thresholds.forEach(threshold => {
        let tp = 0, fp = 0, tn = 0, fn = 0;
        
        for (let i = 0; i < predictions.length; i++) {
            const pred = predictions[i] >= threshold ? 1 : 0;
            const actual = labels[i];
            
            if (pred === 1 && actual === 1) tp++;
            else if (pred === 1 && actual === 0) fp++;
            else if (pred === 0 && actual === 0) tn++;
            else if (pred === 0 && actual === 1) fn++;
        }
        
        const tpr = tp / (tp + fn) || 0;
        const fpr = fp / (fp + tn) || 0;
        
        rocPoints.push({ threshold, tpr, fpr, tp, fp, tn, fn });
    });
    
    // Add endpoints
    rocPoints.unshift({ threshold: 1, tpr: 0, fpr: 0 });
    rocPoints.push({ threshold: 0, tpr: 1, fpr: 1 });
    
    return rocPoints;
}

// Calculate AUC using trapezoidal rule
function calculateAUC(rocPoints) {
    let auc = 0;
    for (let i = 1; i < rocPoints.length; i++) {
        const width = rocPoints[i].fpr - rocPoints[i - 1].fpr;
        const height = (rocPoints[i].tpr + rocPoints[i - 1].tpr) / 2;
        auc += width * height;
    }
    return auc;
}

// Update confusion matrix and metrics based on threshold
function updateConfusionMatrix(predictions, labels, threshold) {
    let tp = 0, fp = 0, tn = 0, fn = 0;
    
    for (let i = 0; i < predictions.length; i++) {
        const pred = predictions[i] >= threshold ? 1 : 0;
        const actual = labels[i];
        
        if (pred === 1 && actual === 1) tp++;
        else if (pred === 1 && actual === 0) fp++;
        else if (pred === 0 && actual === 0) tn++;
        else if (pred === 0 && actual === 1) fn++;
    }
    
    // Calculate metrics
    const accuracy = (tp + tn) / (tp + tn + fp + fn);
    const precision = tp / (tp + fp) || 0;
    const recall = tp / (tp + fn) || 0;
    const f1 = 2 * (precision * recall) / (precision + recall) || 0;
    
    // Display confusion matrix and metrics
    const container = document.getElementById('confusionMatrix');
    container.innerHTML = `
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-top: 20px;">
            <div>
                <h3>Confusion Matrix</h3>
                <table style="margin-top: 10px;">
                    <thead>
                        <tr>
                            <th></th>
                            <th>Predicted Positive</th>
                            <th>Predicted Negative</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <th>Actual Positive</th>
                            <td style="background: #c8e6c9; text-align: center; font-weight: bold;">${tp}</td>
                            <td style="background: #ffccbc; text-align: center; font-weight: bold;">${fn}</td>
                        </tr>
                        <tr>
                            <th>Actual Negative</th>
                            <td style="background: #ffccbc; text-align: center; font-weight: bold;">${fp}</td>
                            <td style="background: #c8e6c9; text-align: center; font-weight: bold;">${tn}</td>
                        </tr>
                    </tbody>
                </table>
            </div>
            <div>
                <h3>Performance Metrics</h3>
                <div class="metrics-grid" style="margin-top: 10px;">
                    <div class="metric-card">
                        <div class="metric-value">${(accuracy * 100).toFixed(2)}%</div>
                        <div class="metric-label">Accuracy</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">${(precision * 100).toFixed(2)}%</div>
                        <div class="metric-label">Precision</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">${(recall * 100).toFixed(2)}%</div>
                        <div class="metric-label">Recall</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">${f1.toFixed(4)}</div>
                        <div class="metric-label">F1 Score</div>
                    </div>
                </div>
            </div>
        </div>
    `;
}

// ============================================
// 7. PREDICTION AND EXPORT
// ============================================

async function makePredictions() {
    try {
        showStatus('Making predictions on test set...', 'info');
        
        if (!testFeatures) {
            alert('No test data loaded');
            return;
        }
        
        // Get predictions
        predictions = model.predict(testFeatures);
        const predArray = await predictions.array();
        
        const info = document.getElementById('predictionInfo');
        info.innerHTML = `
            <div class="info-box">
                <strong>Predictions Complete!</strong><br>
                Test samples: ${predArray.length}<br>
                Ready to download results
            </div>
        `;
        
        // Enable download buttons
        document.getElementById('downloadSubmissionBtn').disabled = false;
        document.getElementById('downloadProbsBtn').disabled = false;
        document.getElementById('saveModelBtn').disabled = false;
        
        showStatus('Predictions completed!', 'success');
        
    } catch (error) {
        showStatus('Error making predictions: ' + error.message, 'error');
        console.error(error);
    }
}

// Download submission.csv
async function downloadSubmission() {
    try {
        const threshold = parseFloat(document.getElementById('thresholdValue').textContent);
        const predArray = await predictions.array();
        
        let csv = 'PassengerId,Survived\n';
        predArray.forEach((pred, i) => {
            const survived = pred[0] >= threshold ? 1 : 0;
            csv += `${testPassengerIds[i]},${survived}\n`;
        });
        
        downloadCSV(csv, 'submission.csv');
        showStatus('Submission file downloaded!', 'success');
        
    } catch (error) {
        alert('Error downloading submission: ' + error.message);
    }
}

// Download probabilities.csv
async function downloadProbabilities() {
    try {
        const predArray = await predictions.array();
        
        let csv = 'PassengerId,Probability\n';
        predArray.forEach((pred, i) => {
            csv += `${testPassengerIds[i]},${pred[0].toFixed(6)}\n`;
        });
        
        downloadCSV(csv, 'probabilities.csv');
        showStatus('Probabilities file downloaded!', 'success');
        
    } catch (error) {
        alert('Error downloading probabilities: ' + error.message);
    }
}

// Save model
async function saveModel() {
    try {
        await model.save('downloads://titanic-tfjs');
        showStatus('Model saved successfully!', 'success');
    } catch (error) {
        alert('Error saving model: ' + error.message);
    }
}

// Helper function to download CSV
function downloadCSV(content, filename) {
    const blob = new Blob([content], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    window.URL.revokeObjectURL(url);
}

// ============================================
// UTILITY FUNCTIONS
// ============================================

function showStatus(message, type) {
    const status = document.getElementById('status');
    status.textContent = message;
    status.className = `status-${type}`;
    status.style.display = 'block';
    
    if (type === 'success') {
        setTimeout(() => {
            status.style.display = 'none';
        }, 5000);
    }
}

console.log('Titanic Classifier with Sigmoid Gate loaded successfully!');
