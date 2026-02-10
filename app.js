/**
 * Logic for Titanic Shallow NN with Sigmoid Gate
 * Author: Your Name
 * Task: Feature importance using Hadamard product gate
 */

let trainRows = [];
let testRows = [];
let model = null;
const FEATURE_NAMES = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'];

// 1. ROBUST CSV PARSER (Fixes the "comma in name" issue)
function parseCSV(text) {
    const lines = text.split('\n').filter(line => line.trim() !== '');
    const headers = lines[0].split(',');
    
    return lines.slice(1).map(line => {
        // Regex to handle commas inside quotes (DeepSeek fix)
        const values = line.match(/(".*?"|[^,]+|(?<=,)(?=,)|(?<=^)(?=,))/g) || [];
        const obj = {};
        headers.forEach((h, i) => {
            let val = values[i] ? values[i].replace(/"/g, '').trim() : '';
            obj[h.trim()] = val;
        });
        return obj;
    });
}

// 2. PREPROCESSING
function preprocess(rows, isTrain = true) {
    return rows.map(row => {
        const features = [
            parseFloat(row.Pclass) || 3,
            row.Sex === 'female' ? 1 : 0,
            (parseFloat(row.Age) || 28) / 80, // Normalized
            parseFloat(row.SibSp) || 0,
            parseFloat(row.Parch) || 0,
            (parseFloat(row.Fare) || 15) / 500, // Normalized
            row.Embarked === 'C' ? 0 : (row.Embarked === 'Q' ? 1 : 2)
        ];
        return {
            x: features,
            y: isTrain ? [parseInt(row.Survived) || 0] : null
        };
    });
}

// 3. MODEL WITH SIGMOID GATE (The "Gate" Method)
function buildGateModel() {
    const input = tf.input({shape: [FEATURE_NAMES.length]});
    
    // The Sigmoid Gate Layer
    // We create a weight vector of the same size as input
    const gateWeights = tf.layers.dense({
        units: FEATURE_NAMES.length,
        activation: 'sigmoid',
        name: 'sigmoid_gate',
        useBias: false
    }).apply(input);

    // Hadamard product: Input * Sigmoid(Weights)
    const gatedInput = tf.layers.multiply().apply([input, gateWeights]);

    // Shallow NN layers
    const hidden = tf.layers.dense({units: 16, activation: 'relu'}).apply(gatedInput);
    const output = tf.layers.dense({units: 1, activation: 'sigmoid'}).apply(hidden);

    const m = tf.model({inputs: input, outputs: output});
    m.compile({
        optimizer: tf.train.adam(0.01),
        loss: 'binaryCrossentropy',
        metrics: ['accuracy']
    });
    return m;
}

// 4. UI HANDLERS
document.getElementById('loadBtn').onclick = async () => {
    const trFile = document.getElementById('trainFile').files[0];
    const tsFile = document.getElementById('testFile').files[0];
    
    if (trFile) {
        const text = await trFile.text();
        trainRows = parseCSV(text);
        renderTable(trainRows.slice(0, 10), 'dataPreview');
        document.getElementById('trainBtn').disabled = false;
    }
    if (tsFile) {
        testRows = parseCSV(await tsFile.text());
    }
};

document.getElementById('trainBtn').onclick = async () => {
    const data = preprocess(trainRows);
    const xTrain = tf.tensor2d(data.map(d => d.x));
    const yTrain = tf.tensor2d(data.map(d => d.y));

    model = buildGateModel();
    
    await model.fit(xTrain, yTrain, {
        epochs: 50,
        batchSize: 32,
        validationSplit: 0.2,
        callbacks: tfvis.show.fitCallbacks(
            { name: 'Training Performance', tab: 'Training' },
            ['loss', 'acc', 'val_loss', 'val_acc']
        )
    });

    document.getElementById('predictBtn').disabled = false;
    showImportance();
    updateMetrics();
};

// 5. EVALUATION & IMPORTANCE
function showImportance() {
    const gateLayer = model.getLayer('sigmoid_gate');
    const weights = gateLayer.getWeights()[0].dataSync();
    
    const data = Array.from(weights).map((w, i) => ({
        index: FEATURE_NAMES[i], value: w
    }));

    tfvis.render.barchart({ name: 'Feature Importance (Sigmoid Gate)', tab: 'Evaluation' }, data);
}

function updateMetrics() {
    const threshold = parseFloat(document.getElementById('thresholdSlider').value);
    document.getElementById('thresholdLabel').innerText = threshold.toFixed(2);
    
    const data = preprocess(trainRows);
    const x = tf.tensor2d(data.map(d => d.x));
    const yTrue = data.map(d => d.y[0]);
    
    const yPredProbs = model.predict(x).dataSync();
    const yPred = Array.from(yPredProbs).map(p => p >= threshold ? 1 : 0);

    // Confusion Matrix calculation
    let tp=0, fp=0, tn=0, fn=0;
    yPred.forEach((p, i) => {
        if(p===1 && yTrue[i]===1) tp++;
        else if(p===1 && yTrue[i]===0) fp++;
        else if(p===0 && yTrue[i]===0) tn++;
        else if(p===0 && yTrue[i]===1) fn++;
    });

    const accuracy = (tp+tn)/(tp+tn+fp+fn);
    document.getElementById('statsOutput').innerHTML = `
        <p>Accuracy: ${(accuracy*100).toFixed(2)}% | TP: ${tp} | TN: ${tn} | FP: ${fp} | FN: ${fn}</p>
    `;

    tfvis.render.confusionMatrix(
        { name: 'Confusion Matrix', tab: 'Evaluation' },
        { values: [[tn, fp], [fn, tp]], tickLabels: ['Dead', 'Survived'] }
    );
}

document.getElementById('thresholdSlider').oninput = updateMetrics;

document.getElementById('predictBtn').onclick = () => {
    const data = preprocess(testRows, false);
    const x = tf.tensor2d(data.map(d => d.x));
    const probs = model.predict(x).dataSync();
    const threshold = parseFloat(document.getElementById('thresholdSlider').value);
    
    let csv = "PassengerId,Survived\n";
    testRows.forEach((row, i) => {
        csv += `${row.PassengerId},${probs[i] >= threshold ? 1 : 0}\n`;
    });
    
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url; a.download = 'submission.csv'; a.click();
    document.getElementById('predictStatus').innerText = "Prediction downloaded!";
};

function renderTable(data, containerId) {
    let html = '<table><tr>' + Object.keys(data[0]).map(k => `<th>${k}</th>`).join('') + '</tr>';
    data.forEach(row => {
        html += '<tr>' + Object.values(row).map(v => `<td>${v}</td>`).join('') + '</tr>';
    });
    html += '</table>';
    document.getElementById(containerId).innerHTML = html;
}
