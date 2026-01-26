let mergedData = [];
let charts = {};

document.getElementById('run-eda-btn').addEventListener('click', handleData);
document.getElementById('export-csv').addEventListener('click', exportCSV);

async function handleData() {
    const trainFile = document.getElementById('train-file').files[0];
    const testFile = document.getElementById('test-file').files[0];

    if (!trainFile || !testFile) return alert("Please upload both files");

    const trainRaw = await parseFile(trainFile);
    const testRaw = await parseFile(testFile);

    const train = trainRaw.data.map(d => ({ ...d, source: 'train' }));
    const test = testRaw.data.map(d => ({ ...d, source: 'test' }));
    mergedData = [...train, ...test].filter(d => d.PassengerId);

    document.getElementById('viz-section').style.display = 'block';
    document.getElementById('export-section').style.display = 'block';

    renderSurvivalCharts();
}

function parseFile(file) {
    return new Promise(res => Papa.parse(file, { header: true, dynamicTyping: true, complete: res }));
}

function renderSurvivalCharts() {
    const trainOnly = mergedData.filter(d => d.source === 'train');

    // 1. Survival by Sex
    const sexData = calculateRate(trainOnly, 'Sex');
    createChart('sexSurvChart', 'bar', sexData.labels, sexData.rates, 'Survival Rate by Gender (%)', '#e74c3c');

    // 2. Survival by Pclass
    const pclassData = calculateRate(trainOnly, 'Pclass');
    createChart('pclassSurvChart', 'bar', pclassData.labels, pclassData.rates, 'Survival Rate by Class (%)', '#3498db');

    // 3. Survival by Embarked
    const embData = calculateRate(trainOnly, 'Embarked');
    createChart('embarkedSurvChart', 'bar', embData.labels, embData.rates, 'Survival Rate by Port (%)', '#2ecc71');

    // 4. Age Distribution
    const ages = mergedData.map(d => d.Age).filter(a => a != null);
    const ageBins = ['0-18', '19-35', '36-60', '60+'];
    const ageCounts = [
        ages.filter(a => a <= 18).length,
        ages.filter(a => a > 18 && a <= 35).length,
        ages.filter(a => a > 35 && a <= 60).length,
        ages.filter(a => a > 60).length
    ];
    createChart('ageHist', 'pie', ageBins, ageCounts, 'Age Distribution (All Data)', ['#ff9f43','#54a0ff','#5f27cd','#48dbfb']);
}

// Helper: Calculates % of Survived=1 for each category in a column
function calculateRate(data, column) {
    const categories = [...new Set(data.map(d => d[column]))].filter(c => c !== null);
    const labels = [];
    const rates = [];

    categories.forEach(cat => {
        const group = data.filter(d => d[column] === cat);
        const survivors = group.filter(d => d.Survived === 1).length;
        const rate = (survivors / group.length) * 100;
        labels.push(cat);
        rates.push(rate.toFixed(1));
    });

    return { labels, rates };
}

function createChart(id, type, labels, data, title, color) {
    if (charts[id]) charts[id].destroy();
    const ctx = document.getElementById(id).getContext('2d');
    charts[id] = new Chart(ctx, {
        type: type,
        data: {
            labels: labels,
            datasets: [{ label: title, data: data, backgroundColor: color }]
        },
        options: { 
            responsive: true, 
            maintainAspectRatio: false,
            scales: type === 'bar' ? { y: { beginAtZero: true, max: 100 } } : {}
        }
    });
}

function exportCSV() {
    const csv = Papa.unparse(mergedData);
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url; a.download = 'merged_titanic.csv'; a.click();
}
