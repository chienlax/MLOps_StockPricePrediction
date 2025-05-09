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
let historicalContextChart; // Renamed chart variable

document.addEventListener("DOMContentLoaded", () => {
    loadLatestPredictionsForTable(); 
    loadTickersForDropdown();      
});

// loadLatestPredictionsForTable and populateLatestPredictionsTable remain the same
// loadTickersForDropdown and populateTickerDropdown remain the same

function updateChart() {
    const ticker = document.getElementById("ticker-select").value;
    if (!ticker || ticker === "No tickers available" || ticker === "Error loading tickers") {
        if (historicalContextChart) { historicalContextChart.destroy(); historicalContextChart = null; }
        const chartCtx = document.getElementById("historical-chart").getContext("2d");
        chartCtx.clearRect(0,0, chartCtx.canvas.width, chartCtx.canvas.height);
        return;
    }

    // Call the new endpoint for historical context
    fetch(`/historical_context_chart/${ticker}`) 
        .then(response => {
            if (!response.ok) {
                return response.json().then(errData => { // Try to get error detail
                    throw new Error(`HTTP error! status: ${response.status}. ${errData.detail || 'Failed to fetch historical context data.'}`);
                }).catch(() => { // Fallback if error detail parsing fails
                     throw new Error(`HTTP error! status: ${response.status}. Failed to fetch historical context data.`);
                });
            }
            return response.json();
        })
        .then(data => {
            if (data.status !== "success") throw new Error(data.detail || "API returned non-success for historical context");
            
            if (!data.dates || data.dates.length === 0) {
                alert(`No historical context data available to chart for ${ticker}. This might happen if no prediction exists yet for this ticker.`);
                 if (historicalContextChart) { historicalContextChart.destroy(); historicalContextChart = null; }
                return;
            }
            renderHistoricalContextChart(data); // Call new rendering function
        })
        .catch(error => {
            console.error(`Error loading historical context data for ${ticker}:`, error);
            alert(`Error fetching historical context data: ${error.message}`);
            if (historicalContextChart) { historicalContextChart.destroy(); historicalContextChart = null; }
        });
}

// NEW chart rendering function for historical context
function renderHistoricalContextChart(data) {
    const { 
        ticker, 
        dates, // These are historical_dates from API
        actual_prices, // These are historical_actual_prices from API
        prediction_reference_date // The date the prediction (in the table) is for
    } = data;

    const canvas = document.getElementById("historical-chart");
    const ctx = canvas.getContext("2d");

    if (historicalContextChart) {
        historicalContextChart.destroy();
        historicalContextChart = null;
    }

    // If no actual historical prices were found, display a message or empty chart
    if (!actual_prices || actual_prices.length === 0) {
        console.warn(`No actual historical prices received for ${ticker} to plot context chart.`);
        // Optionally clear canvas or show a "No historical data" message within the canvas
        ctx.clearRect(0,0, canvas.width, canvas.height); // Clear previous
        ctx.font = "16px Arial";
        ctx.textAlign = "center";
        ctx.fillText(`No historical price data found for ${ticker} to display context.`, canvas.width/2, canvas.height/2);
        return;
    }
    
    historicalContextChart = new Chart(ctx, {
        type: "line",
        data: {
            labels: dates, // Dates corresponding to historical_actual_prices
            datasets: [
                {
                    label: `Actual Price History for ${ticker} (Context for prediction on ${prediction_reference_date})`,
                    data: actual_prices, 
                    borderColor: "rgba(54, 162, 235, 1)", // Blue
                    borderWidth: 2,
                    fill: false,
                    tension: 0.1,
                    pointRadius: 2,
                    pointHoverRadius: 5,
                    spanGaps: false, // Or true, depending on how you want to handle missing actuals within the history
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: `30-Day Price History for ${ticker} (Leading up to prediction for ${prediction_reference_date})`
                },
                tooltip: { mode: 'index', intersect: false },
                legend: { display: true, position: 'top' }
            },
            scales: {
                x: { 
                    title: { display: true, text: "Date" },
                    ticks: { maxRotation: 45, minRotation: 45, autoSkip: true, maxTicksLimit: 15 } // Adjusted for ~30 data points
                },
                y: { 
                    title: { display: true, text: "Actual Price ($)" }, 
                    beginAtZero: false 
                }
            }
        }
    });
}