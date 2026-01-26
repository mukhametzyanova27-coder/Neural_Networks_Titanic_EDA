// Configuration & Schema
const SCHEMA = {
    target: 'Survived',
    features: ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'],
    id: 'PassengerId'
};

let mergedData = [];
let charts = {};

// Event Listeners
document.getElementById('run-eda-btn').addEventListener('click', handleDataProcessing);
document.getElementById('export-csv').addEventListener('click', exportCSV);
document.getElementById('export-json').addEventListener('click', exportJSON);

async function handleDataProcessing() {
    const trainFile = document.getElementById('train-file').files[0];
    const testFile = document.getElementById('test-file').files[0];

    if (!trainFile || !testFile) {
        alert("Please select both train.csv and test.csv files.");
        return;
    }

    try {
        const trainRaw = await parseFile(trainFile);
        const testRaw = await parseFile(testFile);

        // Add source column and merge
        const train = trainRaw.data.map(row => ({ ...row, source: 'train' }));
        const test = testRaw.data.map(row => ({ ...row, source: 'test' }));
        mergedData = [...train, ...test].filter(row => row[SCHEMA.id]); // Clean empty rows

        showSections();
        renderOverview();
        renderMissingValues();
        renderStats();
        renderCharts();
    } catch (err) {
        alert("Error processing files: " + err.message);
    }
}

function parseFile(file) {
    return new Promise((resolve, reject) => {
        Papa.parse(file, {
            header: true,
            dynamicTyping: true,
            skipEmptyLines: true,
            complete: resolve,
            error: reject
        });
    });
}

function showSections() {
    ['overview-section', 'missing-values', 'stats-summary', 'visualizations', 'export-section'].forEach(id => {
        document.getElementById(id).style.display = 'block';
    });
}

function renderOverview() {
    const div = document.getElementById('overview-stats');
    div.innerHTML = `<p>Total records: <b>${mergedData.length}</b> (Train: ${mergedData.filter(d=>d.source==='train').length}, Test: ${mergedData.filter(d=>d.source==='test').length})</p>`;
    
    const table = document.getElementById('preview-table');
    const headers = Object.keys(mergedData[0]);
    table.innerHTML = `<tr>${headers.map(h => `<th>${h}</th>`).join('')}</tr>` + 
        mergedData.slice(0, 5).map(row => `<tr>${headers.map(h => `<td>${row[h] ?? ''}</td>`).join('')}</tr>`).join('');
}

function renderMissingValues() {
    const cols = Object.keys(mergedData[0]);
    const counts = cols.map(col => {
        const missing = mergedData.filter(row => row[col] === null || row[col] === undefined || row[col] === "").length;
        return (missing / mergedData.length * 100).toFixed(2);
    });

    createChart('missingChart', 'bar', cols, counts, 'Missing Values (%)');
}

function renderStats() {
    const trainOnly = mergedData.filter(d => d.source === 'train');
    const survivalBySex = {
        male: trainOnly.filter(d => d.Sex === 'male' && d.Survived === 1).length / trainOnly.filter(d => d.Sex === 'male').length,
        female: trainOnly.filter(d => d.Sex === 'female' && d.Survived === 1).length / trainOnly.filter(d => d.Sex === 'female').length
    };

    document.getElementById('survival-analysis').innerHTML = `
        <h3>Survival Rate by Gender (Train Data)</h3>
        <p>Female: ${(survivalBySex.female * 100).toFixed(2)}% | Male: ${(survivalBySex.male * 100).toFixed(2)}%</p>
        <p><i>Insight: Gender is a key factor for survival.</i></p>
    `;
}

function renderCharts() {
    // Sex Distribution
    const sexCounts = { 
        male: mergedData.filter(d => d.Sex === 'male').length, 
        female: mergedData.filter(d => d.Sex === 'female').length 
    };
    createChart('sexChart', 'pie', Object.keys(sexCounts), Object.values(sexCounts), 'Gender Distribution');

    // Pclass Distribution
    const pclassCounts = [1, 2, 3].map(p => mergedData.filter(d => d.Pclass === p).length);
    createChart('pclassChart', 'bar', ['1st Class', '2nd Class', '3rd Class'], pclassCounts, 'Passenger Class');

    // Age Histogram (simplified)
    const ages = mergedData.map(d => d.Age).filter(a => a != null);
    const bins = [0, 18, 35, 60, 100];
    const ageCounts = bins.slice(0, -1).map((b, i) => ages.filter(a => a >= b && a < bins[i+1]).length);
    createChart('ageHist', 'bar', ['0-17', '18-34', '35-59', '60+'], ageCounts, 'Age Groups');
}

function createChart(canvasId, type, labels, data, label) {
    if (charts[canvasId]) charts[canvasId].destroy();
    const ctx = document.getElementById(canvasId).getContext('2d');
    charts[canvasId] = new Chart(ctx, {
        type: type,
        data: {
            labels: labels,
            datasets: [{ label: label, data: data, backgroundColor: ['#3498db', '#e74c3c', '#2ecc71', '#f1c40f'] }]
        },
        options: { responsive: true, maintainAspectRatio: false }
    });
}

function exportCSV() {
    const csv = Papa.unparse(mergedData);
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url; a.download = 'merged_titanic.csv'; a.click();
}

function exportJSON() {
    const summary = { total: mergedData.length, timestamp: new Date().toISOString() };
    const blob = new Blob([JSON.stringify(summary)], { type: 'application/json' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url; a.download = 'summary.json'; a.click();
}
