document.addEventListener('DOMContentLoaded', function() {
    const tickerSelector = document.getElementById('tickerSelector');
    const predictionsTableBody = document.querySelector('#predictionsTable tbody');
    
    const selectedTickerNameElement = document.getElementById('selectedTickerName');
    const selectedTickerPredictionDateElement = document.getElementById('selectedTickerPredictionDate');
    const selectedTickerPredictedPriceElement = document.getElementById('selectedTickerPredictedPrice');
    const selectedTickerPredictionErrorElement = document.getElementById('selectedTickerPredictionError');
    const chartCanvas = document.getElementById('stockPriceChart');

    let stockChart = null;

    async function fetchTickers() {
        try {
            const response = await fetch('/tickers');
            if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
            const data = await response.json();
            if (data.status === 'success' && data.tickers) {
                populateTickerSelector(data.tickers);
            } else {
                console.error('Failed to load tickers:', data);
                selectedTickerPredictionErrorElement.textContent = 'Failed to load ticker list.';
                selectedTickerPredictionErrorElement.style.display = 'block';
                if (chartCanvas) chartCanvas.style.display = 'none';
            }
        } catch (error) {
            console.error('Error fetching tickers:', error);
            selectedTickerPredictionErrorElement.textContent = 'Error fetching ticker list.';
            selectedTickerPredictionErrorElement.style.display = 'block';
            if (chartCanvas) chartCanvas.style.display = 'none';
        }
    }

    function populateTickerSelector(tickers) {
        tickers.forEach(ticker => {
            const option = document.createElement('option');
            option.value = ticker;
            option.textContent = ticker;
            tickerSelector.appendChild(option);
        });
        
        if (tickers.length > 0) {
            tickerSelector.value = tickers[0];
            fetchAndDisplaySelectedTickerData(tickers[0]); 
        } else {
            selectedTickerNameElement.textContent = 'Ticker: No tickers available';
            selectedTickerPredictionDateElement.textContent = 'Prediction for Date: N/A';
            selectedTickerPredictedPriceElement.textContent = 'Predicted Price: N/A';
            if (chartCanvas) chartCanvas.style.display = 'none';
            renderStockChart("N/A", [], [], null, null, true); 
        }
        
        tickerSelector.addEventListener('change', () => {
            fetchAndDisplaySelectedTickerData(tickerSelector.value);
        });
    }

    async function fetchLatestPredictionsForTable() {
        try {
            const response = await fetch('/predictions/latest_for_table');
            if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
            const data = await response.json();
            predictionsTableBody.innerHTML = ''; 
            if (data.status === 'success' && data.data) {
                data.data.forEach(pred => {
                    const row = predictionsTableBody.insertRow();
                    row.insertCell().textContent = pred.ticker;
                    const price = parseFloat(pred.predicted_price);
                    row.insertCell().textContent = !isNaN(price) ? price.toFixed(2) : 'N/A';
                    row.insertCell().textContent = pred.date; 
                    row.insertCell().textContent = pred.model_mlflow_run_id || 'N/A';
                });
            } else { /* ... error handling ... */ }
        } catch (error) { /* ... error handling ... */ }
    }

    async function fetchAndDisplaySelectedTickerData(ticker) {
        selectedTickerPredictionErrorElement.style.display = 'none';
        selectedTickerPredictionErrorElement.textContent = '';
        selectedTickerNameElement.textContent = `Ticker: ${ticker}`;
        selectedTickerPredictionDateElement.textContent = 'Prediction for Date: Fetching...';
        selectedTickerPredictedPriceElement.textContent = 'Predicted Price: Fetching...';

        if (stockChart) { stockChart.destroy(); stockChart = null; }
        if (chartCanvas) chartCanvas.style.display = 'none';

        try {
            const response = await fetch(`/historical_context_chart/${ticker}`);
            if (!response.ok) {
                const errorData = await response.json().catch(() => ({ detail: `HTTP error! Status: ${response.status}` }));
                throw new Error(errorData.detail || `HTTP error! Status: ${response.status}`);
            }
            const data = await response.json();

            if (data.status === 'success') {
                if (data.prediction_reference_date) {
                    selectedTickerPredictionDateElement.textContent = `Prediction for Date: ${data.prediction_reference_date}`;
                } else {
                    selectedTickerPredictionDateElement.textContent = 'Prediction for Date: N/A';
                }

                const predictedPrice = parseFloat(data.predicted_price_for_ref_date);
                if (!isNaN(predictedPrice)) {
                    selectedTickerPredictedPriceElement.textContent = `Predicted Price: ${predictedPrice.toFixed(2)}`;
                } else {
                    selectedTickerPredictedPriceElement.textContent = 'Predicted Price: N/A';
                    if(data.prediction_reference_date) {
                         selectedTickerPredictionErrorElement.textContent = `No predicted price available for ${ticker} on ${data.prediction_reference_date}.`;
                    } else {
                         selectedTickerPredictionErrorElement.textContent = `No prediction data available for ${ticker}.`;
                    }
                    selectedTickerPredictionErrorElement.style.display = 'block';
                }
                
                // Show canvas only if there's data to plot
                if (chartCanvas && ( (data.historical_dates && data.historical_dates.length > 0) || (data.prediction_reference_date && !isNaN(predictedPrice))) ) {
                    chartCanvas.style.display = 'block';
                } else if (chartCanvas) {
                    chartCanvas.style.display = 'none';
                }


                renderStockChart(
                    ticker, data.historical_dates || [], data.historical_actuals || [],
                    data.prediction_reference_date, data.predicted_price_for_ref_date
                );

            } else { throw new Error(data.detail || `Failed to fetch data for ${ticker}.`); }
        } catch (error) {
            console.error(`Error fetching data for ${ticker}:`, error);
            selectedTickerPredictionDateElement.textContent = 'Prediction for Date: Error';
            selectedTickerPredictedPriceElement.textContent = 'Predicted Price: Error';
            selectedTickerPredictionErrorElement.textContent = `Error loading data: ${error.message}`;
            selectedTickerPredictionErrorElement.style.display = 'block';
            if (chartCanvas) chartCanvas.style.display = 'none';
            renderStockChart(ticker, [], [], null, null, true); 
        }
    }

    function renderStockChart(ticker, historicalDates, historicalActuals, predictionDate, predictedPrice, isError = false) {
        if (!chartCanvas) { console.error("Chart canvas not found!"); return; }
        const ctx = chartCanvas.getContext('2d');
        if (stockChart) { stockChart.destroy(); }
        
        if (isError || (historicalDates.length === 0 && (!predictionDate || predictedPrice === null || typeof predictedPrice === 'undefined'))) {
            console.warn(`Chart for ${ticker} not rendered due to error or insufficient data.`);
            chartCanvas.style.display = 'none'; 
            return;
        }
        chartCanvas.style.display = 'block';

        const datasets = [];
        const validHistoricalData = [];
        if (historicalDates.length > 0 && historicalActuals.length === historicalDates.length) {
            historicalDates.forEach((date, index) => {
                const price = parseFloat(historicalActuals[index]);
                if (!isNaN(price)) { validHistoricalData.push({ x: date, y: price }); }
            });
            if(validHistoricalData.length > 0) {
                 datasets.push({
                    label: `${ticker} - Actual (Last ~14 Trading Days)`, data: validHistoricalData,
                    borderColor: 'rgb(54, 162, 235)', backgroundColor: 'rgba(54, 162, 235, 0.1)',
                    tension: 0.1, pointRadius: 3, borderWidth: 2, type: 'line' });
            }
        }

        const validPredictedPrice = parseFloat(predictedPrice);
        if (predictionDate && !isNaN(validPredictedPrice)) {
            const predictionLineData = [];
            if (validHistoricalData.length > 0) { predictionLineData.push(validHistoricalData[validHistoricalData.length - 1]); }
            predictionLineData.push({ x: predictionDate, y: validPredictedPrice });
            
            datasets.push({
                label: `${ticker} - Predicted`, data: predictionLineData,
                borderColor: 'rgb(255, 99, 132)', borderDash: [5, 5], 
                pointBackgroundColor: 'rgb(255, 99, 132)', pointRadius: 5, pointBorderColor: 'white',
                pointBorderWidth: 1, borderWidth: 2, type: 'line', fill: false });
        }

        if (datasets.length === 0) {
            console.warn(`No valid data to plot for ${ticker}. Hiding chart.`);
            chartCanvas.style.display = 'none'; return;
        }
        
        stockChart = new Chart(ctx, {
            data: { datasets: datasets },
            options: {
                responsive: true, maintainAspectRatio: false,
                plugins: {
                    title: { display: true, text: `Stock Price Trend for ${ticker}` },
                    tooltip: { mode: 'index', intersect: false, callbacks: { label: function(context) { let label = context.dataset.label || ''; if (label) { label += ': '; } if (context.parsed.y !== null) { label += '$' + context.parsed.y.toFixed(2); } return label; }}},
                    legend: { position: 'top', }
                },
                scales: {
                    x: { type: 'time', time: { unit: 'day', tooltipFormat: 'MMM dd, yyyy', displayFormats: { day: 'MMM dd' }}, title: { display: true, text: 'Date' }, ticks: { source: 'auto', maxRotation: 45, autoSkip: true, }},
                    y: { title: { display: true, text: 'Price (USD)' }, beginAtZero: false, ticks: { callback: function(value) { return '$' + parseFloat(value).toFixed(2); }}}
                },
                interaction: { mode: 'nearest', axis: 'x', intersect: false }
            }
        });
    }

    fetchTickers(); 
    fetchLatestPredictionsForTable();
});