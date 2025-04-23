**1. Initial Problem: `404 Not Found` for `/predictions/latest` (API Request)**

*   **Symptom:** Your dashboard loaded (`GET /` was 200 OK), static files (`.css`, `.js`) loaded, but the JavaScript `fetch` request to get prediction data failed with a 404 error in the browser console and Docker logs.
*   **Initial Check:** We verified that your FastAPI code (`src/api/main.py`) *did* have a route defined with `@app.get("/predictions/latest")`.
*   **Root Cause:** The JavaScript code (`app.js`) was using a relative path (`fetch("../predictions/latest")`). In the context of a web page served from the root (`/`), this relative path is ambiguous or incorrect. The browser wasn't actually requesting the correct `/predictions/latest` path from the server root.
*   **Solution:** Changed the `fetch` URL in `app.js` to use an **absolute path**: `fetch("/predictions/latest")`.

**2. Intermediate Problem: `ModuleNotFoundError: No module named 'src'` (Uvicorn Startup)**

*   **Symptom:** (This might have happened earlier or after fixing #1) The `frontend` container failed to start properly. Docker logs showed a `ModuleNotFoundError` traceback originating from Uvicorn when trying to load the app specified as `src.api.main:app`.
*   **Root Cause:** Although the `./src` directory was mounted into the container (e.g., at `/opt/airflow/src`), the Python interpreter running Uvicorn didn't know to look in `/opt/airflow` for top-level modules like `src`. The container's default Python search path (`sys.path`) didn't include the base directory where `src` was located.
*   **Solution:** Explicitly added the base directory to Python's search path by setting the `PYTHONPATH` environment variable for the `frontend` service in `docker-compose.yml`: `environment: - PYTHONPATH=/opt/airflow`.

**3. Final Problem: Still `404 Not Found` for `/predictions/latest` (Internal File Not Found)**

*   **Symptom:** Even after fixing the fetch path (#1) and the module import (#2), the `/predictions/latest` request *still* returned a 404. However, Uvicorn was running correctly, and the route definition was confirmed to exist.
*   **Investigation:** We examined the code *inside* the `@app.get("/predictions/latest")` function in `main.py`. It correctly checked if the prediction file existed (`if not PREDICTION_FILE_PATH.exists(): raise HTTPException(status_code=404, ...)`).
*   **Root Cause:** The `PREDICTION_FILE_PATH` was defined using a **relative path** within `main.py` (e.g., `Path("data/predictions/latest_predictions.json")`). This path was interpreted relative to the *container's working directory* (where Uvicorn was started, often `/app` or `/`), not relative to the volume mount point (`/opt/airflow/data`). The file *did* exist inside the container at `/opt/airflow/data/predictions/latest_predictions.json` (due to the volume mount), but the code was looking in the wrong place (e.g., `/app/data/predictions/...`). The `.exists()` check failed, causing the function itself to raise the 404 error.
*   **Solution:** Changed the definition of `PREDICTION_FILE_PATH` (and `HISTORICAL_DIR`) in `main.py` to use the **absolute path** reflecting the volume mount target inside the container: `PREDICTION_FILE_PATH = Path("/opt/airflow/data/predictions/latest_predictions.json")`. We also ensured static/template paths were similarly absolute or correctly relative.

**In Summary:**

The debugging process involved correcting how the frontend requested data (absolute vs. relative URL), ensuring the backend Python environment could find the source code modules (`PYTHONPATH`), and finally making sure the backend code looked for data files using the correct absolute paths *within the container's filesystem*, reflecting the volume mounts defined in `docker-compose.yml`. This highlights the importance of understanding paths and execution context within Docker containers.