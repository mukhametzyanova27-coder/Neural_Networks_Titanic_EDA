// ============================================================================
// TITANIC BINARY CLASSIFIER - TensorFlow.js
// Dataset: Kaggle Titanic (train.csv, test.csv)
// Target: Survived (0/1)
// Features: Pclass, Sex, Age, SibSp, Parch, Fare, Embarked
// Identifier: PassengerId (excluded from training)
// 
// REUSABILITY NOTE: To adapt for other datasets:
// 1. Update DATA_SCHEMA (targetColumn, featureColumns, identifierColumn)
// 2. Modify preprocessing logic if needed (imputeData, encodeFeatures)
// 3. Adjust model architecture in buildModel() if necessary
// ============================================================================

// DATA SCHEMA - Modify for other datasets
const DATA_SCHEMA = {
    targetColumn: 'Survived',
    identifierColumn: 'PassengerId',
    featureColumns: ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'],
    categoricalFeatures: ['Sex', 'Pclass', 'Embarked'],
    numericFeatures: ['Age', 'Fare', 'SibSp', 'Parch']
};

// Global variables
let trainData = null;
let testData = null;
let preprocessedTrain = null;
let preprocessedTest = null;
let model = null;
let trainResults = null;
let predictions = null;
let rocData = null;

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

// Improved CSV parser that handles quotes and commas inside values
function parseCSV(text) {
    const lines = text.trim().split('\n');
    if (lines.length < 2) {
        throw new Error('CSV file is empty or has only headers');
    }
    
    // Parse header line
    const headers = parseCSVLine(lines[0]);
    const data = [];
    
    // Parse data lines
    for (let i = 1; i < lines.length; i++) {
        if (lines[i].trim() === '') continue; // Skip empty lines
        
        const values = parseCSVLine(lines[i]);
        if (values.length !== headers.length) {
            console.warn(`Line ${i + 1} has ${values.length} values but expected ${headers.length}. Skipping.`);
            continue;
        }
        
        const row = {};
        headers.forEach((header, index) => {
            row[header] = values[index];
        });
        data.push(row);
    }
    
    if (data.length === 0) {
        throw new Error('No valid data rows found in CSV');
    }
    
    console.log(`Parsed CSV: ${data.length} rows, ${headers.length} columns`);
    console.log('Headers:', headers);
    
    return { headers, data };
}

// Parse a single CSV line handling quotes and escaped commas
function parseCSVLine(line) {
    const result = [];
    let current = '';
    let inQuotes = false;
    
    for (let i = 0; i < line.length; i++) {
        const char = line[i];
        const nextChar = line[i + 1];
        
        if (char === '"') {
            if (inQuotes && nextChar === '"') {
                current += '"';
                i++;
            } else {
                inQuotes = !inQuotes;
            }
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

function calculateMissingPercentage(data, column) {
    const missing = data.filter(row => !row[column] || row[column] === '').length;
    return ((missing / data.length) * 100).toFixed(2);
}

function getMedian(values) {
    const sorted = values.filter(v => !isNaN(v) && v !== null).sort((a, b) => a - b);
    if (sorted.length === 0) return 0;
    const mid = Math.floor(sorted.length / 2);
    return sorted.length % 2 === 0 ? (sorted[mid - 1] + sorted[mid]) / 2 : sorted[mid];
}

function getMode(values) {
    const freq = {};
    values.forEach(v => {
        if (v && v !== '') {
            freq[v] = (freq[v] || 0) + 1;
        }
    });
    
    if (Object.keys(freq).length === 0) return '';
    return Object.keys(freq).reduce((a, b) => freq[a] > freq[b] ? a : b);
}

function showStatus(elementId, message, type = 'info') {
    const element = document.getElementById(elementId);
    if (element) {
        element.innerHTML = `<div class="status-message status-${type}">${message}</div>`;
    }
}

// ============================================================================
// 1. DATA LOADING AND INSPECTION
// ============================================================================

async function loadData() {
    try {
        const trainFile = document.getElementById('trainFile').files[0];
        const testFile = document.getElementById('testFile').files[0];
        
        console.log('Load button clicked');
        console.log('Train file:', trainFile);
        console.log('Test file:', testFile);
        
        if (!trainFile || !testFile) {
            alert('Пожалуйста, выберите оба файла: train.csv и test.csv');
            showStatus('dataStatus', 'Ошибка: файлы не выбраны', 'error');
            return;
        }
        
        showStatus('dataStatus', 'Загрузка файлов данных...', 'info');
        
        // Read files
        console.log('Reading train file...');
        const trainText = await trainFile.text();
        console.log('Train file read, length:', trainText.length);
        
        console.log('Reading test file...');
        const testText = await testFile.text();
        console.log('Test file read, length:', testText.length);
        
        // Parse CSV
        console.log('Parsing train CSV...');
        trainData = parseCSV(trainText);
        console.log('Train data parsed:', trainData.data.length, 'rows');
        
        console.log('Parsing test CSV...');
        testData = parseCSV(testText);
        console.log('Test data parsed:', testData.data.length, 'rows');
        
        // Validate schema
        if (!trainData.headers.includes(DATA_SCHEMA.targetColumn)) {
            throw new Error(`Столбец '${DATA_SCHEMA.targetColumn}' не найден в обучающих данных`);
        }
        
        // Check for required feature columns
        const missingFeatures = DATA_SCHEMA.featureColumns.filter(col => !trainData.headers.includes(col));
        if (missingFeatures.length > 0) {
            console.warn('Missing features:', missingFeatures);
        }
        
        // Display preview
        console.log('Displaying preview...');
        displayDataPreview();
        
        // Enable preprocessing
        document.getElementById('preprocessBtn').disabled = false;
        
        showStatus('dataStatus', `✓ Загружено ${trainData.data.length} обучающих примеров, ${testData.data.length} тестовых примеров`, 'success');
        
    } catch (error) {
        showStatus('dataStatus', `Ошибка загрузки данных: ${error.message}`, 'error');
        console.error('Error in loadData:', error);
        alert(`Ошибка: ${error.message}\n\nПроверьте консоль браузера (F12) для подробностей`);
    }
}

function displayDataPreview() {
    const previewDiv = document.getElementById('dataPreview');
    
    // Data shape
    let html = '<h3>Размерность данных</h3>';
    html += `<p>Обучающая выборка: ${trainData.data.length} строк × ${trainData.headers.length} столбцов</p>`;
    html += `<p>Тестовая выборка: ${testData.data.length} строк × ${testData.headers.length} столбцов</p>`;
    
    // Missing values
    html += '<h3>Пропущенные значения (%)</h3>';
    html += '<table><tr><th>Столбец</th><th>Train</th><th>Test</th></tr>';
    DATA_SCHEMA.featureColumns.forEach(col => {
        if (trainData.headers.includes(col)) {
            const trainMissing = calculateMissingPercentage(trainData.data, col);
            const testMissing = calculateMissingPercentage(testData.data, col);
            html += `<tr><td>${col}</td><td>${trainMissing}%</td><td>${testMissing}%</td></tr>`;
        }
    });
    html += '</table>';
    
    // Preview table
    html += '<h3>Предпросмотр обучающих данных (первые 5 строк)</h3>';
    html += '<div style="overflow-x: auto;"><table><tr>';
    trainData.headers.forEach(h => html += `<th>${h}</th>`);
    html += '</tr>';
    trainData.data.slice(0, 5).forEach(row => {
        html += '<tr>';
        trainData.headers.forEach(h => html += `<td>${row[h] || 'NaN'}</td>`);
        html += '</tr>';
    });
    html += '</table></div>';
    
    previewDiv.innerHTML = html;
    
    // Visualize survival by Sex and Pclass
    try {
        visualizeSurvival();
    } catch (error) {
        console.error('Error in visualizeSurvival:', error);
    }
}

async function visualizeSurvival() {
    // Survival by Sex
    const sexCounts = { 'male': { survived: 0, died: 0 }, 'female': { survived: 0, died: 0 } };
    trainData.data.forEach(row => {
        const sex = row['Sex'];
        const survived = parseInt(row[DATA_SCHEMA.targetColumn]);
        if (sex && !isNaN(survived)) {
            if (survived === 1) {
                if (!sexCounts[sex]) sexCounts[sex] = { survived: 0, died: 0 };
                sexCounts[sex].survived++;
            } else {
                if (!sexCounts[sex]) sexCounts[sex] = { survived: 0, died: 0 };
                sexCounts[sex].died++;
            }
        }
    });
    
    const sexData = [
        { index: 0, value: sexCounts.male.survived, label: 'Male Survived' },
        { index: 1, value: sexCounts.male.died, label: 'Male Died' },
        { index: 2, value: sexCounts.female.survived, label: 'Female Survived' },
        { index: 3, value: sexCounts.female.died, label: 'Female Died' }
    ];
    
    tfvis.render.barchart(
        { name: 'Survival by Sex', tab: 'Data Exploration' },
        sexData,
        { width: 400, height: 250 }
    );
    
    // Survival by Pclass
    const pclassCounts = { '1': { survived: 0, died: 0 }, '2': { survived: 0, died: 0 }, '3': { survived: 0, died: 0 } };
    trainData.data.forEach(row => {
        const pclass = row['Pclass'];
        const survived = parseInt(row[DATA_SCHEMA.targetColumn]);
        if (pclass && !isNaN(survived)) {
            if (!pclassCounts[pclass]) pclassCounts[pclass] = { survived: 0, died: 0 };
            if (survived === 1) pclassCounts[pclass].survived++;
            else pclassCounts[pclass].died++;
        }
    });
    
    const pclassData = [
        { index: 0, value: pclassCounts['1'].survived, label: 'Class 1 Survived' },
        { index: 1, value: pclassCounts['1'].died, label: 'Class 1 Died' },
        { index: 2, value: pclassCounts['2'].survived, label: 'Class 2 Survived' },
        { index: 3, value: pclassCounts['2'].died, label: 'Class 2 Died' },
        { index: 4, value: pclassCounts['3'].survived, label: 'Class 3 Survived' },
        { index: 5, value: pclassCounts['3'].died, label: 'Class 3 Died' }
    ];
    
    tfvis.render.barchart(
        { name: 'Survival by Pclass', tab: 'Data Exploration' },
        pclassData,
        { width: 500, height: 250 }
    );
}

// ============================================================================
// 2. DATA PREPROCESSING
// ============================================================================

function preprocessData() {
    try {
        showStatus('preprocessStatus', 'Предобработка данных...', 'info');
        
        const useFamilySize = document.getElementById('featureFamilySize').checked;
        const useIsAlone = document.getElementById('featureIsAlone').checked;
        
        // Impute missing values
        imputeData(trainData.data);
        imputeData(testData.data);
        
        // Encode features
        preprocessedTrain = encodeFeatures(trainData.data, true, useFamilySize, useIsAlone);
        preprocessedTest = encodeFeatures(testData.data, false, useFamilySize, useIsAlone);
        
        // Display preprocessing summary
        displayPreprocessingSummary(useFamilySize, useIsAlone);
        
        // Enable model building
        document.getElementById('buildModelBtn').disabled = false;
        
        showStatus('preprocessStatus', '✓ Предобработка завершена успешно', 'success');
        
    } catch (error) {
        showStatus('preprocessStatus', `Ошибка предобработки: ${error.message}`, 'error');
        console.error(error);
    }
}

function imputeData(data) {
    // Impute Age with median
    const ageValues = data.map(row => parseFloat(row['Age'])).filter(v => !isNaN(v));
    const ageMedian = getMedian(ageValues);
    
    // Impute Embarked with mode
    const embarkedValues = data.map(row => row['Embarked']).filter(v => v && v !== '');
    const embarkedMode = getMode(embarkedValues);
    
    // Impute Fare with median (for test set)
    const fareValues = data.map(row => parseFloat(row['Fare'])).filter(v => !isNaN(v));
    const fareMedian = getMedian(fareValues);
    
    data.forEach(row => {
        if (!row['Age'] || row['Age'] === '' || isNaN(parseFloat(row['Age']))) {
            row['Age'] = ageMedian.toString();
        }
        if (!row['Embarked'] || row['Embarked'] === '') {
            row['Embarked'] = embarkedMode;
        }
        if (!row['Fare'] || row['Fare'] === '' || isNaN(parseFloat(row['Fare']))) {
            row['Fare'] = fareMedian.toString();
        }
    });
}

function encodeFeatures(data, isTrain, useFamilySize, useIsAlone) {
    const encoded = [];
    const targets = [];
    const ids = [];
    
    // Collect all values for standardization
    const ageValues = data.map(row => parseFloat(row['Age']));
    const fareValues = data.map(row => parseFloat(row['Fare']));
    
    const ageMean = ageValues.reduce((a, b) => a + b, 0) / ageValues.length;
    const ageStd = Math.sqrt(ageValues.reduce((sq, n) => sq + (n - ageMean) ** 2, 0) / ageValues.length);
    
    const fareMean = fareValues.reduce((a, b) => a + b, 0) / fareValues.length;
    const fareStd = Math.sqrt(fareValues.reduce((sq, n) => sq + (n - fareMean) ** 2, 0) / fareValues.length);
    
    data.forEach(row => {
        const features = [];
        
        // One-hot encode Pclass (1, 2, 3)
        features.push(row['Pclass'] === '1' ? 1 : 0);
        features.push(row['Pclass'] === '2' ? 1 : 0);
        features.push(row['Pclass'] === '3' ? 1 : 0);
        
        // One-hot encode Sex
        features.push(row['Sex'] === 'male' ? 1 : 0);
        features.push(row['Sex'] === 'female' ? 1 : 0);
        
        // Standardized Age
        const age = parseFloat(row['Age']);
        features.push((age - ageMean) / (ageStd || 1));
        
        // SibSp and Parch (raw)
        features.push(parseFloat(row['SibSp']) || 0);
        features.push(parseFloat(row['Parch']) || 0);
        
        // Standardized Fare
        const fare = parseFloat(row['Fare']);
        features.push((fare - fareMean) / (fareStd || 1));
        
        // One-hot encode Embarked (C, Q, S)
        features.push(row['Embarked'] === 'C' ? 1 : 0);
        features.push(row['Embarked'] === 'Q' ? 1 : 0);
        features.push(row['Embarked'] === 'S' ? 1 : 0);
        
        // Optional: FamilySize
        if (useFamilySize) {
            const familySize = (parseFloat(row['SibSp']) || 0) + (parseFloat(row['Parch']) || 0) + 1;
            features.push(familySize);
        }
        
        // Optional: IsAlone
        if (useIsAlone) {
            const familySize = (parseFloat(row['SibSp']) || 0) + (parseFloat(row['Parch']) || 0) + 1;
            features.push(familySize === 1 ? 1 : 0);
        }
        
        encoded.push(features);
        
        if (isTrain) {
            targets.push(parseFloat(row[DATA_SCHEMA.targetColumn]));
        }
        
        ids.push(row[DATA_SCHEMA.identifierColumn]);
    });
    
    return {
        features: encoded,
        targets: isTrain ? targets : null,
        ids: ids,
        numFeatures: encoded[0].length
    };
}

function displayPreprocessingSummary(useFamilySize, useIsAlone) {
    const outputDiv = document.getElementById('preprocessOutput');
    outputDiv.style.display = 'block';
    
    let html = '<pre>';
    html += `Обучающих примеров: ${preprocessedTrain.features.length}\n`;
    html += `Тестовых примеров: ${preprocessedTest.features.length}\n`;
    html += `Количество признаков: ${preprocessedTrain.numFeatures}\n\n`;
    
    html += 'Feature Engineering:\n';
    html += `- One-hot encoding: Pclass (3), Sex (2), Embarked (3)\n`;
    html += `- Стандартизированные: Age, Fare\n`;
    html += `- Исходные признаки: SibSp, Parch\n`;
    if (useFamilySize) html += `- Создан: FamilySize = SibSp + Parch + 1\n`;
    if (useIsAlone) html += `- Создан: IsAlone = (FamilySize == 1)\n`;
    
    html += '\n';
    html += `Размерность вектора признаков: [${preprocessedTrain.features.length}, ${preprocessedTrain.numFeatures}]\n`;
    html += `Размерность вектора целей: [${preprocessedTrain.targets.length}]\n`;
    html += '</pre>';
    
    outputDiv.innerHTML = html;
}

// ============================================================================
// 3. MODEL BUILDING
// ============================================================================

function buildModel() {
    try {
        showStatus('modelStatus', 'Построение модели...', 'info');
        
        const inputShape = [preprocessedTrain.numFeatures];
        
        model = tf.sequential({
            layers: [
                tf.layers.dense({ inputShape: inputShape, units: 16, activation: 'relu' }),
                tf.layers.dense({ units: 1, activation: 'sigmoid' })
            ]
        });
        
        model.compile({
            optimizer: 'adam',
            loss: 'binaryCrossentropy',
            metrics: ['accuracy']
        });
        
        // Display model summary
        displayModelSummary();
        
        // Enable training
        document.getElementById('trainBtn').disabled = false;
        
        showStatus('modelStatus', '✓ Модель построена успешно', 'success');
        
    } catch (error) {
        showStatus('modelStatus', `Ошибка построения модели: ${error.message}`, 'error');
        console.error(error);
    }
}

function displayModelSummary() {
    const summaryDiv = document.getElementById('modelSummary');
    summaryDiv.style.display = 'block';
    
    let html = '<pre>';
    html += 'Архитектура модели:\n';
    html += '================================================================\n';
    html += `Layer (type)                 Output Shape              Param #\n`;
    html += '================================================================\n';
    
    model.layers.forEach((layer, i) => {
        const outputShape = layer.outputShape;
        const params = layer.countParams();
        html += `${layer.name} (${layer.getClassName()})`.padEnd(29) + 
                `${JSON.stringify(outputShape)}`.padEnd(26) + 
                `${params}\n`;
    });
    
    html += '================================================================\n';
    html += `Всего параметров: ${model.countParams()}\n`;
    html += `Обучаемых параметров: ${model.countParams()}\n`;
    html += '================================================================\n\n';
    
    html += 'Компиляция:\n';
    html += '- Optimizer: adam\n';
    html += '- Loss: binaryCrossentropy\n';
    html += '- Metrics: accuracy\n';
    html += '</pre>';
    
    summaryDiv.innerHTML = html;
}

// ============================================================================
// 4. MODEL TRAINING
// ============================================================================

async function trainModel() {
    try {
        showStatus('trainingStatus', 'Обучение модели...', 'info');
        
        // Stratified split (80/20)
        const { trainX, trainY, valX, valY } = stratifiedSplit(
            preprocessedTrain.features,
            preprocessedTrain.targets,
            0.8
        );
        
        // Convert to tensors
        const xTrain = tf.tensor2d(trainX);
        const yTrain = tf.tensor2d(trainY, [trainY.length, 1]);
        const xVal = tf.tensor2d(valX);
        const yVal = tf.tensor2d(valY, [valY.length, 1]);
        
        // Training configuration
        const callbacks = tfvis.show.fitCallbacks(
            { name: 'Training Performance', tab: 'Training' },
            ['loss', 'val_loss', 'acc', 'val_acc'],
            { callbacks: ['onEpochEnd'] }
        );
        
        // Early stopping callback
        const earlyStopping = tf.callbacks.earlyStopping({
            monitor: 'val_loss',
            patience: 5,
            restoreBestWeights: true
        });
        
        // Train model
        const history = await model.fit(xTrain, yTrain, {
            epochs: 50,
            batchSize: 32,
            validationData: [xVal, yVal],
            callbacks: [callbacks, earlyStopping],
            shuffle: true
        });
        
        trainResults = {
            history: history.history,
            valX: valX,
            valY: valY
        };
        
        // Cleanup tensors
        xTrain.dispose();
        yTrain.dispose();
        xVal.dispose();
        yVal.dispose();
        
        // Enable evaluation
        document.getElementById('evaluateBtn').disabled = false;
        document.getElementById('predictBtn').disabled = false;
        
        showStatus('trainingStatus', `✓ Обучение завершено (${history.epoch.length} эпох)`, 'success');
        
    } catch (error) {
        showStatus('trainingStatus', `Ошибка обучения: ${error.message}`, 'error');
        console.error(error);
    }
}

function stratifiedSplit(features, targets, trainRatio) {
    // Separate by class
    const class0 = [];
    const class1 = [];
    
    features.forEach((feat, idx) => {
        if (targets[idx] === 0) class0.push({ feat, target: 0 });
        else class1.push({ feat, target: 1 });
    });
    
    // Shuffle each class
    shuffle(class0);
    shuffle(class1);
    
    // Split each class
    const trainCount0 = Math.floor(class0.length * trainRatio);
    const trainCount1 = Math.floor(class1.length * trainRatio);
    
    const trainSet = [...class0.slice(0, trainCount0), ...class1.slice(0, trainCount1)];
    const valSet = [...class0.slice(trainCount0), ...class1.slice(trainCount1)];
    
    // Shuffle combined sets
    shuffle(trainSet);
    shuffle(valSet);
    
    return {
        trainX: trainSet.map(item => item.feat),
        trainY: trainSet.map(item => item.target),
        valX: valSet.map(item => item.feat),
        valY: valSet.map(item => item.target)
    };
}

function shuffle(array) {
    for (let i = array.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [array[i], array[j]] = [array[j], array[i]];
    }
}

// ============================================================================
// 5. EVALUATION METRICS (ROC-AUC)
// ============================================================================

async function evaluateModel() {
    try {
        showStatus('metricsStatus', 'Вычисление ROC-AUC...', 'info');
        
        const xVal = tf.tensor2d(trainResults.valX);
        const probabilities = model.predict(xVal).dataSync();
        xVal.dispose();
        
        // Compute ROC curve
        rocData = computeROC(trainResults.valY, Array.from(probabilities));
        
        // Plot ROC curve
        plotROC(rocData);
        
        // Enable threshold slider
        const slider = document.getElementById('thresholdSlider');
        slider.disabled = false;
        slider.value = 0.5;
        
        // Update metrics with default threshold
        updateMetrics(0.5, trainResults.valY, Array.from(probabilities));
        
        // Enable export
        document.getElementById('exportModelBtn').disabled = false;
        
        showStatus('metricsStatus', `✓ ROC-AUC вычислен: ${rocData.auc.toFixed(4)}`, 'success');
        
    } catch (error) {
        showStatus('metricsStatus', `Ошибка оценки: ${error.message}`, 'error');
        console.error(error);
    }
}

function computeROC(trueLabels, probabilities) {
    // Create pairs and sort by probability descending
    const pairs = trueLabels.map((label, i) => ({ label, prob: probabilities[i] }));
    pairs.sort((a, b) => b.prob - a.prob);
    
    const totalPositives = trueLabels.filter(l => l === 1).length;
    const totalNegatives = trueLabels.length - totalPositives;
    
    const rocPoints = [];
    let tp = 0, fp = 0;
    
    rocPoints.push({ fpr: 0, tpr: 0, threshold: 1.0 });
    
    pairs.forEach(pair => {
        if (pair.label === 1) tp++;
        else fp++;
        
        const tpr = tp / totalPositives;
        const fpr = fp / totalNegatives;
        rocPoints.push({ fpr, tpr, threshold: pair.prob });
    });
    
    // Compute AUC using trapezoidal rule
    let auc = 0;
    for (let i = 1; i < rocPoints.length; i++) {
        const width = rocPoints[i].fpr - rocPoints[i - 1].fpr;
        const height = (rocPoints[i].tpr + rocPoints[i - 1].tpr) / 2;
        auc += width * height;
    }
    
    return { rocPoints, auc };
}

function plotROC(rocData) {
    const data = rocData.rocPoints.map(point => ({
        x: point.fpr,
        y: point.tpr
    }));
    
    tfvis.render.scatterplot(
        { name: 'ROC Curve', tab: 'Metrics' },
        { values: [data], series: ['ROC'] },
        {
            xLabel: 'False Positive Rate',
            yLabel: 'True Positive Rate',
            width: 500,
            height: 400,
            zoomToFit: true
        }
    );
}

function updateMetrics(threshold, trueLabels, probabilities) {
    const predictions = probabilities.map(p => p >= threshold ? 1 : 0);
    
    let tp = 0, tn = 0, fp = 0, fn = 0;
    
    trueLabels.forEach((label, i) => {
        const pred = predictions[i];
        if (label === 1 && pred === 1) tp++;
        else if (label === 0 && pred === 0) tn++;
        else if (label === 0 && pred === 1) fp++;
        else if (label === 1 && pred === 0) fn++;
    });
    
    const precision = tp / (tp + fp) || 0;
    const recall = tp / (tp + fn) || 0;
    const f1 = 2 * (precision * recall) / (precision + recall) || 0;
    const accuracy = (tp + tn) / trueLabels.length;
    
    const outputDiv = document.getElementById('metricsOutput');
    let html = '<h3>Матрица ошибок</h3>';
    html += '<table style="width: auto; margin: 0 auto;">';
    html += '<tr><th></th><th>Предсказано 0</th><th>Предсказано 1</th></tr>';
    html += `<tr><th>Истинно 0</th><td>${tn}</td><td>${fp}</td></tr>`;
    html += `<tr><th>Истинно 1</th><td>${fn}</td><td>${tp}</td></tr>`;
    html += '</table>';
    
    html += '<h3>Метрики производительности</h3>';
    html += '<table>';
    html += `<tr><td><strong>Accuracy</strong></td><td>${accuracy.toFixed(4)}</td></tr>`;
    html += `<tr><td><strong>Precision</strong></td><td>${precision.toFixed(4)}</td></tr>`;
    html += `<tr><td><strong>Recall</strong></td><td>${recall.toFixed(4)}</td></tr>`;
    html += `<tr><td><strong>F1 Score</strong></td><td>${f1.toFixed(4)}</td></tr>`;
    html += `<tr><td><strong>AUC</strong></td><td>${rocData.auc.toFixed(4)}</td></tr>`;
    html += '</table>';
    
    outputDiv.innerHTML = html;
}

// ============================================================================
// 6. INFERENCE ON TEST SET
// ============================================================================

async function predictTestSet() {
    try {
        showStatus('predictionStatus', 'Создание предсказаний на тестовой выборке...', 'info');
        
        const xTest = tf.tensor2d(preprocessedTest.features);
        const probabilities = model.predict(xTest).dataSync();
        xTest.dispose();
        
        const threshold = parseFloat(document.getElementById('thresholdSlider').value);
        
        predictions = {
            ids: preprocessedTest.ids,
            probabilities: Array.from(probabilities),
            labels: Array.from(probabilities).map(p => p >= threshold ? 1 : 0)
        };
        
        // Display preview
        displayPredictionPreview();
        
        // Enable export buttons
        document.getElementById('exportSubmissionBtn').disabled = false;
        document.getElementById('exportProbabilitiesBtn').disabled = false;
        
        showStatus('predictionStatus', `✓ Создано предсказаний: ${predictions.ids.length}`, 'success');
        
    } catch (error) {
        showStatus('predictionStatus', `Ошибка предсказания: ${error.message}`, 'error');
        console.error(error);
    }
}

function displayPredictionPreview() {
    const previewDiv = document.getElementById('predictionPreview');
    
    let html = '<h3>Предпросмотр предсказаний (первые 10 строк)</h3>';
    html += '<table>';
    html += '<tr><th>PassengerId</th><th>Probability</th><th>Survived</th></tr>';
    
    for (let i = 0; i < Math.min(10, predictions.ids.length); i++) {
        html += '<tr>';
        html += `<td>${predictions.ids[i]}</td>`;
        html += `<td>${predictions.probabilities[i].toFixed(4)}</td>`;
        html += `<td>${predictions.labels[i]}</td>`;
        html += '</tr>';
    }
    
    html += '</table>';
    previewDiv.innerHTML = html;
}

// ============================================================================
// 7. EXPORT FUNCTIONS
// ============================================================================

function exportSubmission() {
    let csv = 'PassengerId,Survived\n';
    predictions.ids.forEach((id, i) => {
        csv += `${id},${predictions.labels[i]}\n`;
    });
    
    downloadCSV(csv, 'submission.csv');
    showStatus('exportStatus', '✓ Файл submission.csv загружен', 'success');
}

function exportProbabilities() {
    let csv = 'PassengerId,Probability\n';
    predictions.ids.forEach((id, i) => {
        csv += `${id},${predictions.probabilities[i]}\n`;
    });
    
    downloadCSV(csv, 'probabilities.csv');
    showStatus('exportStatus', '✓ Файл probabilities.csv загружен', 'success');
}

async function exportModel() {
    try {
        await model.save('downloads://titanic-tfjs');
        showStatus('exportStatus', '✓ Модель сохранена (проверьте папку Загрузки)', 'success');
    } catch (error) {
        showStatus('exportStatus', `Ошибка сохранения модели: ${error.message}`, 'error');
    }
}

function downloadCSV(content, filename) {
    const blob = new Blob([content], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
}

// ============================================================================
// EVENT LISTENERS
// ============================================================================

document.addEventListener('DOMContentLoaded', function() {
    console.log('DOM loaded, initializing event listeners...');
    
    document.getElementById('loadDataBtn').addEventListener('click', loadData);
    document.getElementById('preprocessBtn').addEventListener('click', preprocessData);
    document.getElementById('buildModelBtn').addEventListener('click', buildModel);
    document.getElementById('trainBtn').addEventListener('click', trainModel);
    document.getElementById('evaluateBtn').addEventListener('click', evaluateModel);
    document.getElementById('predictBtn').addEventListener('click', predictTestSet);
    document.getElementById('exportSubmissionBtn').addEventListener('click', exportSubmission);
    document.getElementById('exportProbabilitiesBtn').addEventListener('click', exportProbabilities);
    document.getElementById('exportModelBtn').addEventListener('click', exportModel);
    
    // Threshold slider listener
    document.getElementById('thresholdSlider').addEventListener('input', (e) => {
        const threshold = parseFloat(e.target.value);
        document.getElementById('thresholdValue').textContent = threshold.toFixed(2);
        
        if (trainResults && rocData) {
            const xVal = tf.tensor2d(trainResults.valX);
            const probabilities = model.predict(xVal).dataSync();
            xVal.dispose();
            updateMetrics(threshold, trainResults.valY, Array.from(probabilities));
        }
        
        if (predictions) {
            predictions.labels = predictions.probabilities.map(p => p >= threshold ? 1 : 0);
            displayPredictionPreview();
        }
    });
    
    console.log('Titanic Binary Classifier initialized. Ready to load data.');
});
