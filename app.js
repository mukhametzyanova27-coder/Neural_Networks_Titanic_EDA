/**
 * Titanic Binary Classifier with Sigmoid Gate
 * Logic: Input -> Sigmoid Gate (Mask) -> Dense Layer -> Output
 */

let trainData = null, testData = null;
let processedData = null;
let model = null;

// Helper: Robust CSV Parser to handle commas in quotes
function parseCSV(text) {
    const lines = text.trim().split('\n');
    const headers = lines[0].split(',');
    return lines.slice(1).map(line => {
        // Regex to split by comma but ignore commas inside double quotes
        const values = line.match(/(".*?"|[^",\s]+)(?=\s*,|\s*$)/g);
        if (!values) return null;
        return headers.reduce((obj, h, i) => {
            obj[h.trim()] = values[i] ? values[i].replace(/"/g, '') : null;
            return obj;
        }, {});
    }).filter(row => row !== null);
}

// 1. Data Loading
document.getElementById('trainFile').addEventListener('change', async (e) => {
    const text = await e.target.files[0].text();
    trainData = parseCSV(text);
    displayPreview(trainData, 'dataPreview');
});

document.getElementById('testFile').addEventListener('change', async (e) => {
    const text = await e.target.files[0].text();
    testData = parseCSV(text);
    document.getElementById('predictBtn').disabled = false;
});

function displayPreview(data, targetId) {
    let html = '<table><tr>' + Object.keys(data[0]).map(k => `<th>${k}</th>`).join('') + '</tr>';
    data.slice(0, 5).forEach(row => {
        html += '<tr>' + Object.values(row).map(v => `<td>${v}</td>`).join('') + '</tr>';
    });
    html += '</table>';
    document.getElementById(targetId).innerHTML = html;
}

// 2. Preprocessing
document.getElementById('prepBtn').addEventListener('click', () => {
    if (!trainData) return alert('Load train.csv first');
    
    const useExtraFeatures = document.getElementById('toggleFamily').checked;
    
    // Simple preprocessing: numeric conversion & imputation
    const clean = (data) => data.map(d => {
        const fare = parseFloat(d.Fare) || 0;
        const age = parseFloat(d.Age) || 29; // Median approx
        const sib = parseInt(d.SibSp) || 0;
        const parch = parseInt(d.Parch) || 0;
        const famSize = sib + parch + 1;

        const feat = {
            Pclass: parseInt(d.Pclass) / 3, // Normalized
            Sex: d.Sex === 'female' ? 1 : 0,
            Age: age / 80,
            SibSp: sib / 8,
            Parch: parch / 6,
            Fare: fare / 500,
            Emb_S: d.Embarked === 'S' ? 1 : 0,
            Emb_C: d.Embarked === 'C' ? 1 : 0,
            Emb_Q: d.Embarked === 'Q' ? 1 : 0
        };

        if (useExtraFeatures) {
            feat.FamilySize = famSize / 10;
            feat.IsAlone = famSize === 1 ? 1 : 0;
        }
        return { x: Object.values(feat), y: parseInt(d.Survived), keys: Object.keys(feat) };
    });

    processedData = clean(trainData);
    document.getElementById('prepStatus').innerText = `Processed ${processedData.length} rows. Features: ${processedData[0].keys.join(', ')}`;
    document.getElementById('trainBtn').disabled = false;
});

// 3. Model Creation (with Sigmoid Gate)
function createModel(inputShape) {
    const inputs = tf.input({shape: [inputShape]});
    
    // Sigmoid Gate: Element-wise mask to learn feature importance
    const gate = tf.layers.dense({
        units: inputShape, 
        activation: 'sigmoid', 
        name: 'gate_layer',
        useBias: false 
    }).apply(inputs);
    
    const gatedInput = tf.layers.multiply().apply([inputs, gate]);
    
    // Shallow Neural Network
    const h1 = tf.layers.dense({units: 16, activation: 'relu'}).apply(gatedInput);
    const outputs = tf.layers.dense({units: 1, activation: 'sigmoid'}).apply(h1);
    
    const m = tf.model({inputs, outputs});
    m.compile({optimizer: 'adam', loss: 'binaryCrossentropy', metrics: ['accuracy']});
    return m;
}

// 4. Training
document.getElementById('trainBtn').addEventListener('click', async () => {
    const X = tf.tensor2d(processedData.map(d => d.x));
    const y = tf.tensor2d(processedData.map(d => d.y), [processedData.length, 1]);

    model = createModel(processedData[0].x.length);
    tfvis.show.modelSummary({name: 'Model Summary'}, model);

    await model.fit(X, y, {
        epochs: 50,
        batchSize: 32,
        validationSplit: 0.2,
        callbacks: [
            tfvis.show.fitCallbacks({ name: 'Training' }, ['loss', 'acc']),
            tf.callbacks.earlyStopping({ monitor: 'val_loss', patience: 5 })
        ]
    });

    document.getElementById('saveModelBtn').disabled = false;
    evaluateModel(X, y);
});

// 5. Evaluation & Gate Visualization
async function evaluateModel(X, y) {
    const probs = model.predict(X);
    const yTrue = y.dataSync();
    const yPredProbs = probs.dataSync();

    // Plot ROC
    const rocData = await tfvis.metrics.roc(yTrue, yPredProbs);
    tfvis.render.linechart({name: 'ROC Curve'}, {values: rocData});

    // Handle Threshold Slider
    const updateMetrics = () => {
        const threshold = parseFloat(document.getElementById('thresholdSlider').value);
        document.getElementById('thresholdLabel').innerText = threshold.toFixed(2);
        
        const yPred = yPredProbs.map(p => p >= threshold ? 1 : 0);
        
        // Manual Confusion Matrix for the UI table
        let tp=0, fp=0, tn=0, fn=0;
        yPred.forEach((p, i) => {
            if(p===1 && yTrue[i]===1) tp++;
            else if(p===1 && yTrue[i]===0) fp++;
            else if(p===0 && yTrue[i]===0) tn++;
            else if(p===0 && yTrue[i]===1) fn++;
        });

        document.getElementById('confusionMatrix').innerHTML = `
            <table>
                <tr><td></td><td>Pred Positive</td><td>Pred Negative</td></tr>
                <tr><td>Actual Pos</td><td>${tp} (TP)</td><td>${fn} (FN)</td></tr>
                <tr><td>Actual Neg</td><td>${fp} (FP)</td><td>${tn} (TN)</td></tr>
            </table>`;
        
        const acc = (tp+tn)/(tp+tn+fp+fn);
        const prec = tp/(tp+fp) || 0;
        const rec = tp/(tp+fn) || 0;
        const f1 = 2*(prec*rec)/(prec+rec) || 0;

        document.getElementById('accVal').innerText = (acc*100).toFixed(2) + '%';
        document.getElementById('precVal').innerText = prec.toFixed(3);
        document.getElementById('recVal').innerText = rec.toFixed(3);
        document.getElementById('f1Val').innerText = f1.toFixed(3);
    };

    document.getElementById('thresholdSlider').oninput = updateMetrics;
    updateMetrics();

    // Feature Importance from Gate
    const gateWeights = model.getLayer('gate_layer').getWeights()[0].dataSync();
    const importanceData = processedData[0].keys.map((key, i) => ({index: key, value: gateWeights[i]}));
    tfvis.render.barchart(document.getElementById('gateWeightsPlot'), importanceData);
}

// 6. Inference & Export
document.getElementById('predictBtn').addEventListener('click', () => {
    if (!testData || !model) return alert('Need model and test data');
    
    // Reuse prep logic for test data (simplified)
    const testClean = testData.map(d => {
        const fare = parseFloat(d.Fare) || 0;
        const age = parseFloat(d.Age) || 29;
        const famSize = (parseInt(d.SibSp)||0) + (parseInt(d.Parch)||0) + 1;
        const feat = [
            (parseInt(d.Pclass)||3)/3, d.Sex==='female'?1:0, age/80, 
            (parseInt(d.SibSp)||0)/8, (parseInt(d.Parch)||0)/6, fare/500,
            d.Embarked==='S'?1:0, d.Embarked==='C'?1:0, d.Embarked==='Q'?1:0
        ];
        if (document.getElementById('toggleFamily').checked) {
            feat.push(famSize/10, famSize===1?1:0);
        }
        return feat;
    });

    const predictions = model.predict(tf.tensor2d(testClean)).dataSync();
    const threshold = parseFloat(document.getElementById('thresholdSlider').value);
    
    let csvContent = "PassengerId,Survived,Probability\n";
    testData.forEach((d, i) => {
        const survived = predictions[i] >= threshold ? 1 : 0;
        csvContent += `${d.PassengerId},${survived},${predictions[i].toFixed(4)}\n`;
    });

    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'submission.csv';
    a.click();
});

document.getElementById('saveModelBtn').addEventListener('click', async () => {
    await model.save('downloads://titanic-model');
});
