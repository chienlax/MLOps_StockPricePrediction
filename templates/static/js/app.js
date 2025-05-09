// let historicalChart;

// // Tải dữ liệu khi trang sẵn sàng
// document.addEventListener("DOMContentLoaded", () => {
//     loadLatestPredictions();
// });

// // ------------------ TẢI DỮ LIỆU MỚI NHẤT ------------------

// function loadLatestPredictions() {
//     fetch("/predictions/latest")
//         .then(response => {
//             if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
//             return response.json();
//         })
//         .then(data => {
//             if (data.status !== "success") throw new Error("Failed to load latest predictions");
//             const predictions = data.data;
//             populateLatestPredictionsTable(predictions);
//             populateTickerDropdown(predictions);
//         })
//         .catch(error => {
//             console.error("Error loading latest predictions:", error);
//             const tbody = document.querySelector("#latest-predictions-table tbody");
//             tbody.innerHTML = `<tr><td colspan="3">Error: ${error.message}</td></tr>`;
//         });
// }

// // ------------------ BẢNG MỚI NHẤT ------------------

// function populateLatestPredictionsTable(predictions) {
//     const tbody = document.querySelector("#latest-predictions-table tbody");
//     tbody.innerHTML = "";
//     predictions.forEach(p => {
//         const row = document.createElement("tr");
//         row.innerHTML = `
//             <td>${p.ticker}</td>
//             <td>$${p.predicted_price.toFixed(2)}</td>
//             <td>${p.date}</td>
//         `;
//         tbody.appendChild(row);
//     });
// }

// // ------------------ DROPDOWN CHỌN TICKER ------------------

// function populateTickerDropdown(predictions) {
//     const select = document.getElementById("ticker-select");
//     select.innerHTML = "";
//     const tickers = [...new Set(predictions.map(p => p.ticker))];
//     tickers.forEach(ticker => {
//         const option = document.createElement("option");
//         option.value = ticker;
//         option.textContent = ticker;
//         select.appendChild(option);
//     });
// }

// // ------------------ CẬP NHẬT BIỂU ĐỒ ------------------

// function updateChart() {
//     const ticker = document.getElementById("ticker-select").value;
//     if (!ticker) {
//         alert("Please select a ticker");
//         return;
//     }

//     fetch(`/predictions/historical/${ticker}`)
//         .then(response => {
//             if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
//             return response.json();
//         })
//         .then(data => {
//             if (data.status !== "success") throw new Error("Failed to load historical predictions");
//             const predictions = data.data;

//             if (predictions.length === 0) {
//                 alert(`No historical data available for ${ticker}`);
//                 return;
//             }

//             // Sort theo ngày tăng dần
//             predictions.sort((a, b) => new Date(a.date) - new Date(b.date));
//             renderHistoricalChart(predictions);
//         })
//         .catch(error => {
//             console.error("Error loading historical predictions:", error);
//             alert(`Error: ${error.message}`);
//         });
// }

// // ------------------ VẼ BIỂU ĐỒ ------------------

// function renderHistoricalChart(predictions) {
//     const ctx = document.getElementById("historical-chart").getContext("2d");

//     if (historicalChart) {
//         historicalChart.destroy();
//     }

//     const labels = predictions.map(p => p.date);
//     const prices = predictions.map(p => p.predicted_price);

//     historicalChart = new Chart(ctx, {
//         type: "line",
//         data: {
//             labels: labels,
//             datasets: [{
//                 label: `Predicted Price for ${predictions[0].ticker}`,
//                 data: prices,
//                 borderColor: "rgba(75, 192, 192, 1)",
//                 borderWidth: 2,
//                 fill: false,
//                 tension: 0.1,
//                 pointRadius: 3,
//                 pointHoverRadius: 6
//             }]
//         },
//         options: {
//             responsive: true,
//             plugins: {
//                 tooltip: {
//                     mode: 'index',
//                     intersect: false,
//                 },
//                 legend: {
//                     display: true
//                 }
//             },
//             scales: {
//                 x: {
//                     title: {
//                         display: true,
//                         text: "Date"
//                     },
//                     ticks: {
//                         maxRotation: 45,
//                         minRotation: 45
//                     }
//                 },
//                 y: {
//                     title: {
//                         display: true,
//                         text: "Predicted Price ($)"
//                     },
//                     beginAtZero: false
//                 }
//             }
//         }
//     });
// }

// templates/static/js/app.js
let historicalChart; 

document.addEventListener("DOMContentLoaded", () => {
    loadLatestPredictionsForTable(); // For the table view
    loadTickersForDropdown();      // For populating the dropdown
});

// Function to load data for the "Latest Predictions" table
function loadLatestPredictionsForTable() {
    fetch("/predictions/latest_for_table") // Still uses this endpoint name
        .then(response => {
            if (!response.ok) throw new Error(`HTTP error! status: ${response.status} for latest_for_table`);
            return response.json();
        })
        .then(data => {
            if (data.status !== "success") throw new Error("Failed to load latest predictions for table from DB");
            populateLatestPredictionsTable(data.data);
        })
        .catch(error => {
            console.error("Error loading latest predictions for table:", error);
            const tbody = document.querySelector("#latest-predictions-table tbody");
            if (tbody) tbody.innerHTML = `<tr><td colspan="3">Error: ${error.message}</td></tr>`;
        });
}

// NEW function to load all distinct tickers for the dropdown
function loadTickersForDropdown() {
    fetch("/tickers") // New endpoint
        .then(response => {
            if (!response.ok) throw new Error(`HTTP error! status: ${response.status} for /tickers`);
            return response.json();
        })
        .then(data => {
            if (data.status !== "success" || !data.tickers) throw new Error("Failed to load tickers for dropdown");
            populateTickerDropdown(data.tickers);
            // Optionally, auto-load chart for the first ticker if list is not empty
            if (data.tickers && data.tickers.length > 0) {
                document.getElementById("ticker-select").value = data.tickers[0];
                updateChart(); // This will now call /chart_data/{ticker}
            }
        })
        .catch(error => {
            console.error("Error loading tickers for dropdown:", error);
            const select = document.getElementById("ticker-select");
            if (select) {
                select.innerHTML = "<option>Error loading tickers</option>";
            }
        });
}


function populateLatestPredictionsTable(predictions) {
    const tbody = document.querySelector("#latest-predictions-table tbody");
    if (!tbody) return;
    tbody.innerHTML = ""; 
    if (!predictions || predictions.length === 0) {
        tbody.innerHTML = `<tr><td colspan="3">No latest predictions available from database.</td></tr>`;
        return;
    }
    predictions.forEach(p => {
        const row = document.createElement("tr");
        // Ensure 'date' and 'predicted_price' keys match what get_latest_prediction_for_all_tickers returns
        row.innerHTML = `
            <td>${p.ticker}</td>
            <td>$${p.predicted_price ? parseFloat(p.predicted_price).toFixed(2) : 'N/A'}</td>
            <td>${p.date}</td> 
        `;
        tbody.appendChild(row);
    });
}

// MODIFIED to take a simple array of ticker strings
function populateTickerDropdown(tickers) {
    const select = document.getElementById("ticker-select");
    if (!select) return;
    select.innerHTML = ""; 

    if (!tickers || tickers.length === 0) {
        const option = document.createElement("option");
        option.textContent = "No tickers available";
        select.appendChild(option);
        return;
    }

    tickers.sort().forEach(ticker => { // Sort tickers alphabetically
        const option = document.createElement("option");
        option.value = ticker;
        option.textContent = ticker;
        select.appendChild(option);
    });
}

// updateChart and renderDualLineChart remain the same as in the previous version,
// as they already call /chart_data/{ticker} which was designed to provide
// both actual and predicted prices from the database.

function updateChart() {
    const ticker = document.getElementById("ticker-select").value;
    if (!ticker || ticker === "No tickers available" || ticker === "Error loading tickers") {
        if (historicalChart) {
            historicalChart.destroy();
            historicalChart = null;
        }
        const chartCtx = document.getElementById("historical-chart").getContext("2d");
        chartCtx.clearRect(0,0, chartCtx.canvas.width, chartCtx.canvas.height);
        // You could also write "Please select a ticker" on the canvas
        return;
    }

    fetch(`/chart_data/${ticker}`)
        .then(response => {
            if (!response.ok) {
                return response.json().then(errData => {
                    throw new Error(`HTTP error! status: ${response.status}. ${errData.detail || 'Failed to fetch chart data.'}`);
                }).catch(() => {
                     throw new Error(`HTTP error! status: ${response.status}. Failed to fetch chart data.`);
                });
            }
            return response.json();
        })
        .then(data => {
            if (data.status !== "success") throw new Error(data.detail || "API returned non-success status for chart data");
            
            if (data.dates.length === 0 && data.actual_prices.every(p => p === null) && data.predicted_prices.every(p => p === null) ) {
                alert(`No chart data available for ${ticker}.`);
                 if (historicalChart) {
                    historicalChart.destroy();
                    historicalChart = null;
                 }
                return;
            }
            renderDualLineChart(data.ticker, data.dates, data.actual_prices, data.predicted_prices);
        })
        .catch(error => {
            console.error(`Error loading chart data for ${ticker}:`, error);
            alert(`Error fetching chart data: ${error.message}`);
            if (historicalChart) {
                historicalChart.destroy();
                historicalChart = null; 
            }
        });
}

function renderDualLineChart(ticker, dates, actualPrices, predictedPrices) {
    const ctx = document.getElementById("historical-chart").getContext("2d");

    if (historicalChart) {
        historicalChart.destroy(); 
    }

    historicalChart = new Chart(ctx, {
        type: "line",
        data: {
            labels: dates,
            datasets: [
                {
                    label: `Actual Price for ${ticker}`,
                    data: actualPrices,
                    borderColor: "rgba(54, 162, 235, 1)", 
                    backgroundColor: "rgba(54, 162, 235, 0.5)",
                    borderWidth: 2,
                    fill: false,
                    tension: 0.1,
                    pointRadius: 2,
                    pointHoverRadius: 5,
                    spanGaps: true, 
                },
                {
                    label: `Predicted Price for ${ticker}`,
                    data: predictedPrices,
                    borderColor: "rgba(255, 99, 132, 1)", 
                    backgroundColor: "rgba(255, 99, 132, 0.5)",
                    borderWidth: 2,
                    fill: false,
                    tension: 0.1,
                    pointRadius: 2,
                    pointHoverRadius: 5,
                    spanGaps: true, 
                }
            ]
        },
        options: { // Your existing chart options are good
            responsive: true,
            maintainAspectRatio: false,
            plugins: { tooltip: { mode: 'index', intersect: false,}, legend: {display: true, position: 'top',}},
            scales: {
                x: { title: {display: true, text: "Date"}, ticks: {maxRotation: 45, minRotation: 45, autoSkip: true, maxTicksLimit: 15 }},
                y: { title: {display: true, text: "Price ($)"}, beginAtZero: false }
            }
        }
    });
}