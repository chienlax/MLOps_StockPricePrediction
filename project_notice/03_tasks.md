Okay, let's consolidate everything discussed into a comprehensive, step-by-step plan covering the remaining tasks to complete your MLOps Stock Prediction project, including NLP enhancement, deployment, and monitoring.

**Current State:** You have a refactored codebase (`src/`), an automated training pipeline orchestrated by Airflow (`airflow/dags/lstm_refactored_dag.py`), running within Docker (`docker-compose.yml`), and tracking experiments with MLflow (`mlflow-server` service). DVC is set up for data versioning.

**Overall Goal:** Enhance the model with NLP features, deploy predictions via a user-facing dashboard, implement monitoring, and add basic CI/CD & robustness features.

---

**Phase 1: NLP Feature Integration (Enhancing Model Input)**

*   **Goal:** Incorporate news sentiment as an input feature for the LSTM model.

1.  **Setup News Fetching:**
    *   **Action:** Choose news source (API like NewsAPI/Finnhub recommended, or Web Scraping). Obtain API key if needed.
    *   **Action:** Create script `src/data/fetch_news.py`.
    *   **Action:** Implement logic to fetch daily news relevant to your tickers (`config/params.yaml -> data_loading.tickers`). Handle API keys securely (e.g., via `.env` file read by the script or Airflow Connections later).
    *   **Action:** Save raw news data daily (e.g., `data/raw_news/YYYY-MM-DD_news.json`).
    *   **Testing:** Run the script standalone to verify news fetching.

2.  **Setup Sentiment Analysis:**
    *   **Action:** Choose sentiment model (Hugging Face FinBERT recommended, or VADER). Install necessary libraries (`transformers`, `torch`, `vaderSentiment`, etc. - add to `requirements.txt`).
    *   **Action:** Create script `src/features/build_sentiment_features.py`.
    *   **Action:** Implement logic to: Load raw news, apply the chosen sentiment model, aggregate scores daily per ticker (handle missing news), and save to `data/processed/sentiment_features.csv` (Columns: `Date, Ticker, AvgSentiment`).
    *   **Testing:** Run the script standalone on sample news data.

3.  **Integrate Sentiment into Feature Pipeline:**
    *   **Action:** **Modify** `src/features/build_features.py`.
    *   **Action:** After loading processed price/technical data, load `data/processed/sentiment_features.csv`.
    *   **Action:** **Merge** the sentiment data with the price/technical data based on `Date` and `Ticker`. Handle missing sentiment values (e.g., fill with 0).
    *   **Action:** Update the feature list and the final `processed_data` numpy array to include the new sentiment feature.
    *   **Action:** Ensure the `scale_data` function correctly handles the added feature column.
    *   **Testing:** Run the modified `build_features.py` standalone. Check the shape of the output `split_scaled_data.npz` (should have one more feature).

4.  **Update Airflow Data/Feature Tasks:**
    *   **Action:** **Modify** `airflow/dags/lstm_refactored_dag.py`.
    *   **Action:** Add new `BashOperator` tasks (`task_fetch_news`, `task_build_sentiment`) executing the respective scripts.
    *   **Action:** Adjust dependencies: `[task_process_data, task_fetch_news] >> task_build_sentiment >> task_build_features`.
    *   **Testing:** Run the modified DAG up to `task_build_features`. Check logs and verify intermediate files (`sentiment_features.csv`, updated `split_scaled_data.npz`).

**Phase 2: Prediction Generation & Deployment (Serving Predictions)**

*   **Goal:** Generate predictions daily using the best model and display them on a dashboard.

5.  **Implement Prediction Logic:**
    *   **Action:** Create/Refine script `src/models/predict_model.py`.
    *   **Action:** Implement logic to:
        *   Load the desired model from MLflow (e.g., `models:/stock-prediction-lstm/Production` - see step 9).
        *   Load required scalers (`data/processed/scalers.pkl`).
        *   Fetch latest *price* data AND latest *news* data.
        *   Perform *all* necessary preprocessing, technical indicator calculation, sentiment analysis, feature merging, and scaling on this latest data (reuse functions!).
        *   Generate predictions for required horizons.
        *   Save predictions to a structured file (e.g., `data/predictions/latest_predictions.json`).
    *   **Testing:** Run the script standalone, providing a valid model URI.

6.  **Add Prediction Task to DAG:**
    *   **Action:** **Modify** `airflow/dags/lstm_refactored_dag.py`.
    *   **Action:** Add `BashOperator` task `task_generate_predictions` executing `predict_model.py` after `task_train_final_model` (or potentially run independently if training isn't daily). Pass the correct model URI.
    *   **Action:** Add `data/predictions/` to DVC tracking (`dvc add data/predictions`). Commit `.dvc` file.
    *   **Testing:** Run the full DAG. Verify `latest_predictions.json` is created/updated.

7.  **Build API Backend:**
    *   **Action:** Implement FastAPI app in `src/api/main.py`.
    *   **Action:** Create endpoints (`/predictions/latest`, `/predictions/historical/<ticker>`) to read `latest_predictions.json` and potentially processed/raw data.
    *   **Testing:** Run API locally (`uvicorn src.api.main:app --reload`) and test endpoints.

8.  **Build Dashboard Frontend:**
    *   **Action:** Implement `templates/index.html` + JavaScript (using Fetch, Plotly.js/Chart.js) OR use Streamlit/Dash (`dashboard_app.py`).
    *   **Action:** Fetch data from the API and display interactive charts/predictions.
    *   **Testing:** Open the HTML file or run the Streamlit/Dash app locally.

9.  **Containerize & Deploy Frontend:**
    *   **Action:** Create `Dockerfile.frontend` or similar.
    *   **Action:** Add `frontend` service to `docker-compose.yml`. Configure image, ports, command, and **mount `data` volume (read-only)**.
    *   **Testing:** Run `docker compose up -d --build`. Access the frontend service via its exposed port (e.g., `http://localhost:8000`). Verify it fetches and displays predictions correctly.

**Phase 3: Monitoring & Model Management (Improving Robustness)**

*   **Goal:** Track model performance, detect data issues, and manage model versions systematically.

10. **Integrate MLflow Model Registry:**
    *   **Action:** **Modify** `src/models/train_model.py`: Use `registered_model_name` in `mlflow.pytorch.log_model`. Use `MlflowClient` to transition new versions to "Staging".
    *   **Action:** **Modify** `src/models/predict_model.py` (and API): Load models using registry URIs (`models:/stock-prediction-lstm/Production` or `Staging`).
    *   **Action:** Define promotion process: Manually via MLflow UI or build an evaluation/promotion task in Airflow.

11. **Implement Performance Monitoring:**
    *   **Action:** Create script `src/monitoring/log_performance.py` to fetch metrics from the latest MLflow training run and append to `data/monitoring/performance_log.csv`.
    *   **Action:** Add `task_log_performance` to the DAG after `task_train_final_model`.
    *   **Action:** Track `data/monitoring/` with DVC.
    *   **(Optional):** Add performance trend plot to the dashboard.

12. **Implement Data Drift Detection:**
    *   **Action:** Create script `src/monitoring/check_data_drift.py` using `evidently.ai` or similar. Compare latest processed data vs. a reference dataset.
    *   **Action:** Add `task_check_data_drift` to the DAG (e.g., after `task_build_features`).
    *   **Action:** Log drift report/metric to MLflow or trigger alerts.

**Phase 4: CI/CD & Final Touches (Automation & Quality)**

*   **Goal:** Automate testing and improve code quality.

13. **Implement Unit Tests:**
    *   **Action:** Write `pytest` tests in `tests/` for core functions (data processing, feature engineering utilities).

14. **Setup Continuous Integration (CI):**
    *   **Action:** Create GitHub Actions workflow (`.github/workflows/ci.yaml`).
    *   **Action:** Configure jobs for linting (`flake8/black`) and running unit tests (`pytest`) on pushes/PRs.

15. **(Optional) Automate DVC/Git in DAGs:**
    *   **Action:** Add `BashOperator` tasks for `dvc add`, `git add/commit/push` if needed, ensuring secure credential handling.

16. **Documentation:**
    *   **Action:** Update `README.md` comprehensively: Explain the project, architecture, setup instructions (`docker compose up`), how to access UI/API, DAG details, and monitoring aspects.
    *   **Action:** Document key design decisions in `docs/architecture.md`.

**Execution Order:**

*   Complete Phase 1 (NLP) first, as it changes the model's input features. Test thoroughly.
*   Implement Phase 2 (Deployment) to make the (now NLP-enhanced) model's predictions visible.
*   Implement Phase 3 (Monitoring) to add stability and insight.
*   Implement Phase 4 (CI/CD) for better development practices.


Okay, let's create that comprehensive, step-by-step plan, integrating all the phases (NLP, Deployment via Hybrid Cloud, Monitoring, CI/CD) and respecting your constraint of not using a credit card for GCP (leveraging the Always Free tier where possible).

**Project Goal:** Complete the Automated Stock Market Price Prediction project with NLP enhancements, deploy a publicly accessible dashboard using GCP Always Free tier + local execution, and implement monitoring and basic CI.

**Core Architecture (Hybrid):**

*   **Local Machine (via Docker Compose):** Runs Airflow, MLflow Server, Postgres DB, executes the entire data processing, feature engineering (including NLP), model training, optimization, and prediction generation pipeline.
*   **GCP Cloud (Always Free Tier):**
    *   **Artifact Registry:** Stores the Docker image for the frontend/API service.
    *   **Cloud Run:** Hosts the containerized frontend/API service, serving the dashboard.
    *   **Cloud Storage (GCS):** Stores the `latest_predictions.json` file, acting as the bridge between local execution and the cloud-hosted frontend.

---

**Comprehensive Step-by-Step Instructions:**

**Phase 1: Finalize Local Pipeline with NLP & Prediction Output**

*   **Goal:** Ensure the complete pipeline, including NLP and prediction saving, runs correctly locally within Docker Compose.

1.  **(NLP) Implement News Fetching:**
    *   Choose news source (API/Scraping).
    *   Create `src/data/fetch_news.py`.
    *   Implement fetching logic, saving daily raw news (e.g., `data/raw_news/YYYY-MM-DD_news.json`). Handle API keys via `.env`.
    *   Test standalone: `docker compose exec airflow-scheduler python /opt/airflow/src/data/fetch_news.py --config /opt/airflow/config/params.yaml`

2.  **(NLP) Implement Sentiment Analysis:**
    *   Choose model (FinBERT/VADER). Add dependencies to `requirements.txt`.
    *   Create `src/features/build_sentiment_features.py`.
    *   Implement logic to load raw news, analyze sentiment, aggregate daily scores per ticker, save to `data/processed/sentiment_features.csv`.
    *   Test standalone: `docker compose exec airflow-scheduler python /opt/airflow/src/features/build_sentiment_features.py --config /opt/airflow/config/params.yaml`

3.  **(NLP) Integrate Sentiment into Feature Building:**
    *   Modify `src/features/build_features.py`.
    *   Add logic to load `sentiment_features.csv` and merge it with price/technical data *before* sequence creation and scaling.
    *   Ensure scaling handles the new sentiment column.
    *   Rebuild Docker image if `requirements.txt` changed: `docker compose build airflow-scheduler` (or just `docker compose build`).
    *   Test standalone: `docker compose exec airflow-scheduler python /opt/airflow/src/features/build_features.py --config /opt/airflow/config/params.yaml`. Verify output shapes.

4.  **(Prediction) Implement Prediction Script:**
    *   Refine `src/models/predict_model.py`.
    *   Ensure it loads the target model (initially, maybe latest run; later from Registry - Step 14).
    *   Ensure it performs *all* feature engineering steps, **including fetching latest news and running sentiment analysis** on that fresh data before merging and scaling. Reuse functions!
    *   Ensure it saves predictions to `data/predictions/latest_predictions.json`.
    *   Test standalone (requires a trained model artifact in MLflow): `docker compose exec airflow-scheduler python /opt/airflow/src/models/predict_model.py --config /opt/airflow/config/params.yaml --model-uri runs:/<your_mlflow_run_id>/model`

5.  **(Airflow) Update Main DAG:**
    *   Modify `airflow/dags/lstm_refactored_dag.py`.
    *   Add `BashOperator` tasks: `task_fetch_news`, `task_build_sentiment`, `task_generate_predictions`.
    *   Adjust dependencies: `[task_process_data, task_fetch_news] >> task_build_sentiment >> task_build_features >> task_optimize_hyperparams >> task_train_final_model >> task_generate_predictions`.
    *   Test the *entire DAG* locally via Airflow UI (`http://localhost:8080`). Verify all tasks succeed and `latest_predictions.json` is generated.

6.  **(DVC) Track New Artifacts:**
    *   Run locally on your host machine (ensure DVC is installed locally `pip install dvc`):
        ```bash
        dvc add data/raw_news data/processed/sentiment_features.csv data/predictions
        git add data/raw_news.dvc data/processed/sentiment_features.csv.dvc data/predictions.dvc data/.gitignore
        git commit -m "feat: Add NLP features and prediction output tracking"
        ```

**Phase 2: GCP Setup (Cloud Foundation)**

*   **Goal:** Prepare the necessary GCP services within the Always Free tier limits.

7.  **Setup GCP Project:**
    *   Create/Use a Google Account.
    *   Go to Google Cloud Console ([https://console.cloud.google.com/](https://console.cloud.google.com/)).
    *   Create a new GCP Project. Note down the **Project ID**. Try proceeding without adding billing/credit card info.
    *   Install Google Cloud SDK (`gcloud` CLI) locally following Google's instructions.
    *   Authenticate gcloud: Run `gcloud auth login` in your local terminal and follow browser steps.
    *   Set your project: `gcloud config set project YOUR_PROJECT_ID`.

8.  **Enable APIs:**
    *   Enable necessary APIs (might prompt for billing setup, proceed cautiously):
        ```bash
        gcloud services enable artifactregistry.googleapis.com
        gcloud services enable run.googleapis.com
        gcloud services enable storage.googleapis.com
        gcloud services enable cloudbuild.googleapis.com # Optional for later CI/CD builds
        ```

9.  **Create GCS Bucket:**
    *   Choose a **globally unique** bucket name (e.g., `yourname-stockpred-data`).
    *   Create the bucket (standard storage class, choose a region):
        ```bash
        gsutil mb -p YOUR_PROJECT_ID -c standard -l YOUR_CHOSEN_REGION gs://your-unique-bucket-name
        ```
    *   Note down `gs://your-unique-bucket-name`.

10. **Create Artifact Registry Repository:**
    *   Choose a repository name (e.g., `stockpred-repo`).
    *   Create the Docker repository in your chosen region:
        ```bash
        gcloud artifacts repositories create stockpred-repo \
            --repository-format=docker \
            --location=YOUR_CHOSEN_REGION \
            --description="Docker images for Stock Prediction app"
        ```
    *   Note the full repository path (e.g., `YOUR_CHOSEN_REGION-docker.pkg.dev/YOUR_PROJECT_ID/stockpred-repo`).

**Phase 3: Build & Push Frontend Docker Image**

*   **Goal:** Create the deployable Docker image for the dashboard/API and push it to GCP Artifact Registry.

11. **Prepare Frontend Dockerfile & Requirements:**
    *   Create `requirements_api.txt` listing dependencies for your FastAPI app (e.g., `fastapi`, `uvicorn[standard]`, `pandas`, `plotly`, `google-cloud-storage`).
    *   Create `Dockerfile.frontend` (or adapt your main `Dockerfile` if simple enough):
        ```dockerfile
        # Dockerfile.frontend
        FROM python:3.9-slim
        WORKDIR /app
        COPY requirements_api.txt .
        RUN pip install --no-cache-dir -r requirements_api.txt
        COPY ./src/api /app/src/api
        COPY ./templates /app/templates # If using templates
        # Ensure API code reads bucket name/prediction path from ENV VARS
        ENV GCS_BUCKET_NAME=your-unique-bucket-name
        ENV PREDICTION_FILE_PATH=latest_predictions.json
        EXPOSE 8000
        # Command to run API, ensuring it binds to 0.0.0.0
        CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
        ```
    *   **Modify API Code (`src/api/main.py`):** Update it to read the GCS bucket name and prediction file path from environment variables and use the `google-cloud-storage` library to download the `latest_predictions.json` from GCS.

12. **Build, Tag, and Push Frontend Image:**
    *   Authenticate Docker with Artifact Registry:
        ```bash
        gcloud auth configure-docker YOUR_CHOSEN_REGION-docker.pkg.dev
        ```
    *   Define image name: `export IMAGE_NAME=YOUR_CHOSEN_REGION-docker.pkg.dev/YOUR_PROJECT_ID/stockpred-repo/frontend:latest` (use `set` instead of `export` in Windows CMD).
    *   Build the image:
        ```bash
        docker build -f Dockerfile.frontend -t $IMAGE_NAME .
        ```
        (Replace `$IMAGE_NAME` with `%IMAGE_NAME%` in Windows CMD if using `set`).
    *   Push the image:
        ```bash
        docker push $IMAGE_NAME
        ```

**Phase 4: Deploy Frontend to Cloud Run**

*   **Goal:** Make the dashboard publicly accessible using the container image.

13. **Deploy to Cloud Run:**
    *   Deploy using `gcloud`:
        ```bash
        gcloud run deploy stockpred-frontend \
            --image=$IMAGE_NAME \
            --platform=managed \
            --region=YOUR_CHOSEN_REGION \
            --allow-unauthenticated \
            --set-env-vars=GCS_BUCKET_NAME=your-unique-bucket-name,PREDICTION_FILE_PATH=latest_predictions.json
            # Add --port=8000 if your container uses a different port
        ```
    *   This command deploys the image, allows public access, sets environment variables for the API, and assigns a public HTTPS URL. Note down the **Service URL**.

14. **Initial Test:** Access the Service URL in your browser. Expect it to load but potentially show no prediction data yet, as the GCS file doesn't exist or is empty.

**Phase 5: Implement Data Bridge (Local to Cloud)**

*   **Goal:** Get the `latest_predictions.json` generated locally by Airflow into the GCS bucket for Cloud Run to access.

15. **Choose Upload Method:**
    *   **Option A (Manual Upload - Start here):** After your local Airflow DAG run completes `task_generate_predictions`:
        ```bash
        gsutil cp data/predictions/latest_predictions.json gs://your-unique-bucket-name/latest_predictions.json
        ```
        *Refresh your Cloud Run dashboard URL - predictions should now appear.* Document this manual step in your `README.md`.
    *   **Option B (Automate with Airflow Task - Better):**
        *   Add `apache-airflow-providers-google` to `requirements.txt`. Rebuild main Docker image (`docker compose build`).
        *   Configure a GCP Connection in Airflow UI (`Admin -> Connections -> Add Record`). Choose "Google Cloud", provide a Conn Id (e.g., `google_cloud_default`). Authenticate using a Service Account Key JSON file (download from GCP IAM, mount securely into Airflow container or store path in `.env` - security risk locally!).
        *   Import `GCSTaskUploadOperator` in your DAG.
        *   Add a new task at the end:
            ```python
            from airflow.providers.google.cloud.transfers.local_to_gcs import LocalFilesystemToGCSOperator

            task_upload_predictions = LocalFilesystemToGCSOperator(
                task_id='upload_predictions_to_gcs',
                src='/opt/airflow/data/predictions/latest_predictions.json', # Source inside container
                dst='latest_predictions.json', # Destination path within bucket
                bucket='your-unique-bucket-name',
                gcp_conn_id='google_cloud_default',
            )
            # Set dependency: task_generate_predictions >> task_upload_predictions
            ```
        *   Rerun the DAG and check if the file appears in GCS.

**Phase 6: Monitoring & Model Management (Local Pipeline)**

*   **Goal:** Add robustness to the locally running pipeline.

16. **Implement Model Registry:**
    *   Modify `src/models/train_model.py`: Use `registered_model_name` in `mlflow.pytorch.log_model` and transition to "Staging".
    *   Modify `src/models/predict_model.py`: Load model using registry URI (e.g., `models:/stock-prediction-lstm/Staging`). Update `task_generate_predictions` in DAG if needed.
    *   Define promotion process (Manual via MLflow UI for now).

17. **Implement Performance Logging:**
    *   Create `src/monitoring/log_performance.py`.
    *   Add `task_log_performance` to DAG after `task_train_final_model`.
    *   Track `data/monitoring/performance_log.csv` with DVC.

18. **Implement Data Drift Check:**
    *   Create `src/monitoring/check_data_drift.py`.
    *   Add `task_check_data_drift` to DAG after `task_build_features`. Log results/artifacts to MLflow.

**Phase 7: CI/CD & Final Touches**

*   **Goal:** Automate quality checks and finalize documentation.

19. **Implement Unit Tests:**
    *   Write tests in `tests/` for helper functions.

20. **Setup Basic CI:**
    *   Create `.github/workflows/ci.yaml` for linting and running unit tests.

21. **Final Documentation:**
    *   Update `README.md` with full setup (local Docker, GCP setup, deployment), usage instructions, architecture overview, and manual steps (like GCS upload if using Option 5A).

This comprehensive plan guides you through integrating NLP, deploying the user-facing part to the cloud for free, bridging the data gap, and adding essential monitoring and quality checks, all while working within your constraints. Remember to test each step!