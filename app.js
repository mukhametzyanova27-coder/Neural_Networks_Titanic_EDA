/**
 * Titanic Survival Predictor using TensorFlow.js
 * Feature: Sigmoid Gate for Feature Importance
 * Fix: PapaParse for CSV comma escape issue
 */

let trainData = null;
let testData = null;
let model = null;
let featureNames = [];

// --- 1. DATA LOADING & INSPECTION ---

document.getElementById('btnLoad').addEventListener('click', async () => {
    const trainFile = document.getElementById('trainFile').files[0];
    const testFile = document.getElementById('testFile').files[0];

    if (!trainFile) return alert("Please select train.csv");

    trainData = await parseCSV(trainFile);
    if (testFile) testData = await parseCSV(testFile);

    displayPreview(trainData.slice(0, 10), 'preview');
    document.getElementById('btnTrain').disabled = false;
    
    // Initial visualization
    const survivedBySex = trainData.map(d => ({
        sex: d.Sex,
        survived: parseInt(d.Survived)
    }));
    tfvis.render.barchart({ name: 'Survival by Sex', tab: 'Inspection' }, 
        [{ index: 'male', value: survivedBySex.filter(d => d.sex === 'male' && d.survived === 1).length },
         { index: 'female', value: survivedBySex.filter(d => d.sex === 'female' && d.survived === 1).length }]);
});

async function parseCSV(file) {
    return new Promise((resolve) => {
        Papa.parse(file, {
            header: true,
            dynamicTyping: true,
            skipEmptyLines: true,
            complete: (results) => resolve(results.data)
        });
    });
}

function displayPreview(data, elementId) {
    let html = '<table><tr>' + Object.keys(data[0]).map(k => `<th>${k}</th>`).join('') + '</tr>';
    data.forEach(row => {
        html += '<tr>' + Object.values(row).map(v => `<td>${v}</td>`).join('') + '</tr>';
    });
    html += '</table>';
    document.getElementById(elementId).innerHTML = html;
}

// --- 2. PREPROCESSING ---

function preprocess(data, isTrain = true) {
    // Schema: Pclass, Sex, Age, SibSp, Parch, Fare, Embarked
    return data.map(d => {
        const processed = {
            survived: isTrain ? (parseInt(d.Survived) || 0) : null,
            features: [
                d.Pclass || 3,
                d.Sex === 'female' ? 1 : 0,
                d.Age || 28, // Simple median imputation
                d.SibSp || 0,
                d.Parch || 0,
                d.Fare || 14.45,
                d.Embarked === 'C' ? 1 : (d.Embarked === 'Q' ? 2 : 0) // One-hot simplified
            ]
        };
        return processed;
    });
}

// --- 3. MODEL CREATION (WITH SIGMOID GATE) ---

function createModel(inputShape) {
    // Using Functional API to implement the Sigmoid Gate (Mask)
    const input = tf.input({shape: [inputShape]});
    
    // THE GATE: Learnable weights -> Sigmoid -> Element-wise multiplication
    // This identifies "Important Factors"
    const gateWeights = tf.layers.dense({
        units: inputShape, 
        activation: 'sigmoid', 
        useBias: false,
        name: 'sigmoid_gate_layer'
    }).apply(input);
    
    const gatedInput = tf.layers.multiply().apply([input, gateWeights]);

    // Shallow Neural Network
    const hidden = tf.layers.dense({units: 16, activation: 'relu'}).apply(gatedInput);
    const output = tf.layers.dense({units: 1, activation: 'sigmoid'}).apply(hidden);

    const model = tf.model({inputs: input, outputs: output});
    model.compile({
        optimizer: tf.train.adam(0.01),
        loss: 'binaryCrossentropy',
        metrics: ['accuracy']
    });
    return model;
}

// --- 4. TRAINING ---

document.getElementById('btnTrain').addEventListener('click', async () => {
    const processed = preprocess(trainData);
    featureNames = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'];
    
    const xData = tf.tensor2d(processed.map(d => d.features));
    const yData = tf.tensor2d(processed.map(d => [d.survived]));

    model = createModel(xData.shape[1]);
    
    const surface = { name: 'Model Training', tab: 'Training' };
    
    await model.fit(xData, yData, {
        epochs: 50,
        batchSize: 32,
        validationSplit: 0.2,
        callbacks: [
            tfvis.show.fitCallbacks(surface, ['loss', 'acc', 'val_loss', 'val_acc']),
            {
                onEpochEnd: async () => {
                    updateMetrics(xData, yData);
                    visualizeImportance();
                }
            }
        ]
    });

    document.getElementById('btnPredict').disabled = false;
    document.getElementById('trainStatus').innerText = "Training Completed!";
});

// --- 5. METRICS & GATE VISUALIZATION ---

async function visualizeImportance() {
    // Extract weights from the first layer (the Gate)
    const gateLayer = model.getLayer('sigmoid_gate_layer');
    const weights = gateLayer.getWeights()[0].dataSync();
    
    const importanceData = Array.from(weights).map((w, i) => ({
        index: featureNames[i],
        value: w
    }));

    tfvis.render.barchart({ name: 'Feature Importance (Sigmoid Gate)', tab: 'Evaluation' }, importanceData);
}

async function updateMetrics(x, y) {
    const threshold = parseFloat(document.getElementById('thresholdSlider').value);
    document.getElementById('thresholdVal').innerText = threshold.toFixed(2);

    const preds = model.predict(x);
    const predValues = preds.dataSync();
    const actualValues = y.dataSync();

    let tp = 0, fp = 0, tn = 0, fn = 0;
    for(let i=0; i < predValues.length; i++) {
        const p = predValues[i] >= threshold ? 1 : 0;
        const a = actualValues[i];
        if (p === 1 && a === 1) tp++;
        else if (p === 1 && a === 0) fp++;
        else if (p === 0 && a === 0) tn++;
        else if (p === 0 && a === 1) fn++;
    }

    const precision = tp / (tp + fp) || 0;
    const recall = tp / (tp + fn) || 0;
    const f1 = 2 * (precision * recall) / (precision + recall) || 0;

    document.getElementById('stats').innerHTML = `
        <div class="metric-box">Precision: ${precision.toFixed(3)}</div>
        <div class="metric-box">Recall: ${recall.toFixed(3)}</div>
        <div class="metric-box">F1: ${f1.toFixed(3)}</div>
    `;

    const matrix = [[tn, fp], [fn, tp]];
    tfvis.render.confusionMatrix({ name: 'Confusion Matrix', tab: 'Evaluation' }, {
        values: matrix,
        tickLabels: ['Dead', 'Survived']
    });
}

document.getElementById('thresholdSlider').addEventListener('input', () => {
    if (model) {
        const processed = preprocess(trainData);
        updateMetrics(tf.tensor2d(processed.map(d => d.features)), tf.tensor2d(processed.map(d => [d.survived])));
    }
});

// --- 6. PREDICTION & DOWNLOAD ---

document.getElementById('btnPredict').addEventListener('click', async () => {
    if (!testData) return alert("Upload test.csv first");

    const processed = preprocess(testData, false);
    const xTest = tf.tensor2d(processed.map(d => d.features));
    const preds = model.predict(xTest).dataSync();
    const threshold = parseFloat(document.getElementById('thresholdSlider').value);

    let csvContent = "PassengerId,Survived\n";
    testData.forEach((row, i) => {
        const survived = preds[i] >= threshold ? 1 : 0;
        csvContent += `${row.PassengerId},${survived}\n`;
    });

    // Download Logic
    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'submission.csv';
    a.click();
    
    // Save model
    await model.save('downloads://titanic-model');
});
