let historicalChart;

// Tải dữ liệu khi trang sẵn sàng
document.addEventListener("DOMContentLoaded", () => {
    loadLatestPredictions();
});

// ------------------ TẢI DỮ LIỆU MỚI NHẤT ------------------

function loadLatestPredictions() {
    fetch("/predictions/latest")
        .then(response => {
            if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
            return response.json();
        })
        .then(data => {
            if (data.status !== "success") throw new Error("Failed to load latest predictions");
            const predictions = data.data;
            populateLatestPredictionsTable(predictions);
            populateTickerDropdown(predictions);
        })
        .catch(error => {
            console.error("Error loading latest predictions:", error);
            const tbody = document.querySelector("#latest-predictions-table tbody");
            tbody.innerHTML = `<tr><td colspan="3">Error: ${error.message}</td></tr>`;
        });
}

// ------------------ BẢNG MỚI NHẤT ------------------

function populateLatestPredictionsTable(predictions) {
    const tbody = document.querySelector("#latest-predictions-table tbody");
    tbody.innerHTML = "";
    predictions.forEach(p => {
        const row = document.createElement("tr");
        row.innerHTML = `
            <td>${p.ticker}</td>
            <td>$${p.predicted_price.toFixed(2)}</td>
            <td>${p.date}</td>
        `;
        tbody.appendChild(row);
    });
}

// ------------------ DROPDOWN CHỌN TICKER ------------------

function populateTickerDropdown(predictions) {
    const select = document.getElementById("ticker-select");
    select.innerHTML = "";
    const tickers = [...new Set(predictions.map(p => p.ticker))];
    tickers.forEach(ticker => {
        const option = document.createElement("option");
        option.value = ticker;
        option.textContent = ticker;
        select.appendChild(option);
    });
}

// ------------------ CẬP NHẬT BIỂU ĐỒ ------------------

function updateChart() {
    const ticker = document.getElementById("ticker-select").value;
    if (!ticker) {
        alert("Please select a ticker");
        return;
    }

    fetch(`/predictions/historical/${ticker}`)
        .then(response => {
            if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
            return response.json();
        })
        .then(data => {
            if (data.status !== "success") throw new Error("Failed to load historical predictions");
            const predictions = data.data;

            if (predictions.length === 0) {
                alert(`No historical data available for ${ticker}`);
                return;
            }

            // Sort theo ngày tăng dần
            predictions.sort((a, b) => new Date(a.date) - new Date(b.date));
            renderHistoricalChart(predictions);
        })
        .catch(error => {
            console.error("Error loading historical predictions:", error);
            alert(`Error: ${error.message}`);
        });
}

// ------------------ VẼ BIỂU ĐỒ ------------------

function renderHistoricalChart(predictions) {
    const ctx = document.getElementById("historical-chart").getContext("2d");

    if (historicalChart) {
        historicalChart.destroy();
    }

    const labels = predictions.map(p => p.date);
    const prices = predictions.map(p => p.predicted_price);

    historicalChart = new Chart(ctx, {
        type: "line",
        data: {
            labels: labels,
            datasets: [{
                label: `Predicted Price for ${predictions[0].ticker}`,
                data: prices,
                borderColor: "rgba(75, 192, 192, 1)",
                borderWidth: 2,
                fill: false,
                tension: 0.1,
                pointRadius: 3,
                pointHoverRadius: 6
            }]
        },
        options: {
            responsive: true,
            plugins: {
                tooltip: {
                    mode: 'index',
                    intersect: false,
                },
                legend: {
                    display: true
                }
            },
            scales: {
                x: {
                    title: {
                        display: true,
                        text: "Date"
                    },
                    ticks: {
                        maxRotation: 45,
                        minRotation: 45
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: "Predicted Price ($)"
                    },
                    beginAtZero: false
                }
            }
        }
    });
}
