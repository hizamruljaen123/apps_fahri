
function convertToRupiah(value) {
    return value.toLocaleString('id-ID', { style: 'currency', currency: 'IDR' });
}


async function dataUji() {
    try {
        const response = await fetch('http://127.0.0.1:5000/data_uji'); // Ganti dengan link API yang sesuai
        const data = await response.json();
        
        // Sort data by name in ascending order
        data.sort((a, b) => a.nama.localeCompare(b.nama));
        
        const tableBody = document.getElementById('data_uji');
        tableBody.innerHTML = ''; // Clear existing data

        data.forEach(item => {
            const row = `
                <tr>
                    <td>${item.nama}</td>
                    <td>${item.alasan_layak_pip}</td>
                    <td>${convertToRupiah(item.jumlah_bantuan)}</td>
                    <td>${item.jumlah_tanggungan}</td>
                    <td>${item.layak_pip}</td>
                    <td>${item.penghasilan}</td>
                    <td>${item.status_bantuan}</td>
                    <td>${item.status_ekonomi}</td>
                    <td>${item.tahun_penerimaan}</td>
                </tr>
            `;
            tableBody.innerHTML += row;
        });
        document.getElementById('data_uji_counter').textContent = `${data.length} Data`;
    } catch (error) {
        console.error('Error fetching data:', error);
    }
}

async function data_latih() {
    try {
        const response = await fetch('http://127.0.0.1:5000/data_latih'); // Ganti dengan link API yang sesuai
        const data = await response.json();
        
        // Sort data by name in ascending order
        data.sort((a, b) => a.nama.localeCompare(b.nama));
        
        const tableBody = document.getElementById('data_latih');
        tableBody.innerHTML = ''; // Clear existing data

        data.forEach(item => {
            const row = `
                <tr>
                    <td>${item.nama}</td>
                    <td>${item.alasan_layak_pip}</td>
                    <td>${convertToRupiah(item.jumlah_bantuan)}</td>
                    <td>${item.jumlah_tanggungan}</td>
                    <td>${item.layak_pip}</td>
                    <td>${item.penghasilan}</td>
                    <td>${item.status_bantuan}</td>
                    <td>${item.status_ekonomi}</td>
                    <td>${item.status_kesesuaian}</td>
                    <td>${item.tahun_penerimaan}</td>
                </tr>
            `;
            tableBody.innerHTML += row;
        });
        document.getElementById('data_latih_counter').textContent = `${data.length} Data`;

    } catch (error) {
        console.error('Error fetching data:', error);
    }
}

document.addEventListener('DOMContentLoaded', hasil_prediksi);

async function hasil_prediksi() {
    try {
        // Fetch data from the API
        const response = await fetch('http://127.0.0.1:5000/predict'); // Ganti dengan link API yang sesuai
        const data = await response.json(); // Parse response as JSON

        // Ensure the data is properly parsed
        let parsedData = data;

        // If the data is a string (not JSON), parse it into a JSON object
        if (typeof parsedData === 'string') {
            parsedData = JSON.parse(parsedData);
        }

        // Sort data by 'nama' field in ascending order
        parsedData.predictions.sort((a, b) => a.nama.localeCompare(b.nama));

        // Get table body for rendering rows
        const tableBody = document.getElementById('data_respons');
        tableBody.innerHTML = ''; // Clear existing table data

        // Categories for counting and displaying results
        const categories = ["Sesuai", "Tidak Sesuai", "Tidak Tepat Sasaran"];
        const adaboostCounts = { "Sesuai": 0, "Tidak Sesuai": 0, "Tidak Tepat Sasaran": 0 };
        const svmCounts = { "Sesuai": 0, "Tidak Sesuai": 0, "Tidak Tepat Sasaran": 0 };

        // Loop through each item in the predictions and populate the table
        parsedData.predictions.forEach(item => {
            const row = `
                <tr>
                    <td>${item.nama ? item.nama : 'N/A'}</td>
                    <td>${item.status_bantuan ? item.status_bantuan : 'N/A'}</td>
                    <td>${item.adaboost_predicted_kesesuaian ? item.adaboost_predicted_kesesuaian : 'N/A'}</td>
                    <td>${item.svm_predicted_kesesuaian ? item.svm_predicted_kesesuaian : 'N/A'}</td>
                </tr>
            `;
            tableBody.innerHTML += row;

            // Count the occurrences for Adaboost and SVM predictions
            if (categories.includes(item.adaboost_predicted_kesesuaian)) {
                adaboostCounts[item.adaboost_predicted_kesesuaian]++;
            }

            if (categories.includes(item.svm_predicted_kesesuaian)) {
                svmCounts[item.svm_predicted_kesesuaian]++;
            }
        });

        // Plotly chart data preparation
        const trace1 = {
            x: categories,
            y: categories.map(cat => adaboostCounts[cat]),
            name: 'Adaboost',
            type: 'bar',
            text: categories.map(cat => adaboostCounts[cat].toString()),
            textposition: 'auto'
        };

        const trace2 = {
            x: categories,
            y: categories.map(cat => svmCounts[cat]),
            name: 'SVM',
            type: 'bar',
            text: categories.map(cat => svmCounts[cat].toString()),
            textposition: 'auto'
        };

        // Plot the chart using Plotly
        const dataPlotly = [trace1, trace2];
        const layout = {
            barmode: 'group',
            title: 'Jumlah Status Kesesuaian Adaboost dan SVM',
            xaxis: { title: 'Status Kesesuaian' },
            yaxis: { title: 'Jumlah' }
        };

        Plotly.newPlot('kesesuaian_chart', dataPlotly, layout);

        // Populate hasil_counter table
        const hasilCounterBody = document.getElementById('hasil_counter');
        hasilCounterBody.innerHTML = ''; // Clear existing data

        const totalAdaboost = adaboostCounts["Sesuai"] + adaboostCounts["Tidak Sesuai"] + adaboostCounts["Tidak Tepat Sasaran"];
        const totalSvm = svmCounts["Sesuai"] + svmCounts["Tidak Sesuai"] + svmCounts["Tidak Tepat Sasaran"];

        const rows = [
            { keterangan: 'Sesuai (SVM Adaboost)', jumlah: adaboostCounts["Sesuai"], persentase: ((adaboostCounts["Sesuai"] / totalAdaboost) * 100 || 0).toFixed(2) },
            { keterangan: 'Tidak Sesuai (SVM Adaboost)', jumlah: adaboostCounts["Tidak Sesuai"], persentase: ((adaboostCounts["Tidak Sesuai"] / totalAdaboost) * 100 || 0).toFixed(2) },
            { keterangan: 'Tidak Tepat Sasaran (SVM Adaboost)', jumlah: adaboostCounts["Tidak Tepat Sasaran"], persentase: ((adaboostCounts["Tidak Tepat Sasaran"] / totalAdaboost) * 100 || 0).toFixed(2) },
            { keterangan: 'Sesuai (SVM)', jumlah: svmCounts["Sesuai"], persentase: ((svmCounts["Sesuai"] / totalSvm) * 100 || 0).toFixed(2) },
            { keterangan: 'Tidak Sesuai (SVM)', jumlah: svmCounts["Tidak Sesuai"], persentase: ((svmCounts["Tidak Sesuai"] / totalSvm) * 100 || 0).toFixed(2) },
            { keterangan: 'Tidak Tepat Sasaran (SVM)', jumlah: svmCounts["Tidak Tepat Sasaran"], persentase: ((svmCounts["Tidak Tepat Sasaran"] / totalSvm) * 100 || 0).toFixed(2) }
        ];

        // Render rows for the counter table
        rows.forEach(row => {
            const rowElement = `
                <tr>
                    <td>${row.keterangan}</td>
                    <td>${row.jumlah}</td>
                    <td>${row.persentase}%</td>
                </tr>
            `;
            hasilCounterBody.innerHTML += rowElement;
        });

    } catch (error) {
        console.error('Error fetching or parsing data:', error);
        alert('Terjadi kesalahan saat mengambil data prediksi. Mohon coba lagi.');
    }
}

document.addEventListener('DOMContentLoaded', () => {
    document.getElementById('dataLatihForm').addEventListener('submit', function(event) {
        event.preventDefault();
        saveDataLatih();
    });
});

async function saveDataLatih() {
    const formData = {
        nama: document.getElementById('nama').value,
        penghasilan: document.getElementById('penghasilan').value,
        status_ekonomi: document.getElementById('status_ekonomi').value,
        jumlah_tanggungan: document.getElementById('jumlah_tanggungan').value,
        layak_pip: document.getElementById('layak_pip').value,
        alasan_layak_pip: document.getElementById('alasan_layak_pip').value,
        tahun_penerimaan: document.getElementById('tahun_penerimaan').value,
        jumlah_bantuan: document.getElementById('jumlah_bantuan').value,
        status_bantuan: document.getElementById('status_bantuan').value,
        status_kesesuaian: document.getElementById('status_kesesuaian').value
    };
    
    try {
        const response = await fetch('http://127.0.0.1:5000/save_data_latih', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(formData)
        });

        if (response.ok) {
            alert('Data berhasil disimpan');
            document.getElementById('dataLatihForm').reset(); // Reset form setelah sukses menyimpan data
        } else {
            alert('Gagal menyimpan data');
        }
    } catch (error) {
        console.error('Error:', error);
        alert('Terjadi kesalahan saat menyimpan data');
    }
}
async function trainModel() {
    try {
        // Update textarea to show loading message
        document.getElementById('evaluationResult').value = "Malatih Model...";

        const response = await fetch('http://127.0.0.1:5000/train', { method: 'GET' });
        const data = await response.json();

        alert('Model di bentuk');
        document.getElementById('evaluationResult').value = "Model training completed.";
    } catch (error) {
        console.error('Error training model:', error);
        alert('Failed to train model');
        document.getElementById('evaluationResult').value = "Failed to train model.";
    }
}

async function evaluateModel() {
    try {
        // Update textarea to show loading message
        document.getElementById('evaluationResult').value = "Loading...";

        const response = await fetch('http://127.0.0.1:5000/evaluate', { method: 'GET' });
        const data = await response.text();

        document.getElementById('evaluationResult').value = data;
    } catch (error) {
        console.error('Error evaluating model:', error);
        alert('Failed to evaluate model');
        document.getElementById('evaluationResult').value = "Failed to evaluate model.";
    }
}
async function saveDataUji() {
    const formData = {
        nama: document.getElementById('data_uji_nama').value,
        penghasilan: document.getElementById('data_uji_penghasilan').value,
        status_ekonomi: document.getElementById('data_uji_status_ekonomi').value,
        jumlah_tanggungan: document.getElementById('data_uji_jumlah_tanggungan').value,
        layak_pip: document.getElementById('data_uji_layak_pip').value,
        alasan_layak_pip: document.getElementById('data_uji_alasan_layak_pip').value,
        tahun_penerimaan: document.getElementById('data_uji_tahun_penerimaan').value,
        jumlah_bantuan: document.getElementById('data_uji_jumlah_bantuan').value,
        status_bantuan: document.getElementById('data_uji_status_bantuan').value
    };

    try {
        const response = await fetch('http://127.0.0.1:5000/save_data_uji', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(formData)
        });

        if (response.ok) {
            alert('Data berhasil disimpan');
            document.getElementById('dataUjiForm').reset(); // Reset form setelah sukses menyimpan data
        } else {
            const errorData = await response.json();
            alert('Gagal menyimpan data: ' + errorData.message);
        }
    } catch (error) {
        console.error('Error:', error);
        alert('Terjadi kesalahan saat menyimpan data');
    }
}
document.getElementById('trainButton').addEventListener('click', trainModel);
document.getElementById('evaluateButton').addEventListener('click', evaluateModel);
data_latih()
dataUji()
hasil_prediksi()