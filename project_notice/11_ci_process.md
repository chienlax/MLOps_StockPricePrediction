**CI Goal:** Automate code validation, testing, Airflow DAG checks, Docker image building, and pushing versioned images to Docker Hub on pushes to `dev` and `main` branches.

**Prerequisites (Recap - Ensure these are in place):**

1.  **GitHub Repository:** Your project code is hosted on GitHub.
2.  **Branching:**
    *   `dev` branch exists and is your primary development integration branch.
    *   `main` branch exists and will represent your "production" deployable state.
3.  **Docker Hub Account & Repositories:**
    *   Account created.
    *   Public repositories (or private with appropriate plan):
        *   `yourdockerhubusername/stockpred-app` (for the main app/Airflow image)
        *   `yourdockerhubusername/stockpred-frontend` (for the FastAPI frontend image)
4.  **Docker Hub Access Token:** Generated and securely noted.
5.  **GitHub Secrets in your Repository (`Settings` -> `Secrets and variables` -> `Actions`):**
    *   `DOCKERHUB_USERNAME`: Your Docker Hub username.
    *   `DOCKERHUB_TOKEN`: The Docker Hub access token you generated.
6.  **Project Files:**
    *   `Dockerfile`: For building your main application image (Airflow, Python `src`, DAGs, etc.).
    *   `Dockerfile.frontend`: For building your FastAPI frontend image.
    *   `requirements.txt`: Lists all Python dependencies, including `pytest`, `flake8`, `black`, `isort`, `apache-airflow`, and any specific providers like `psycopg2-binary` (or `psycopg2` if you handle build dependencies in Dockerfile).
    *   `tests/` directory: Contains your Pytest unit tests.
    *   `airflow/dags/` directory: Contains your Airflow DAG Python files.
    *   `src/` directory: Contains your application's Python modules.

---

**Step-by-Step CI Implementation Guide for the CI Team**

**Step 1: Create the GitHub Actions Workflow File**

1.  In the root of your local Git repository, ensure the directory path `.github/workflows/` exists. If not, create it:
    ```bash
    mkdir -p .github/workflows
    ```
2.  Inside this directory, create a new YAML file named `ci-pipeline.yml` (or `main.yml`, `ci.yml` - the name doesn't strictly matter, but `.yml` or `.yaml` extension is required).

**Step 2: Define Workflow Triggers, Environment, and Initial Setup in `ci-pipeline.yml`**

*   Open `ci-pipeline.yml` and add the following initial content:

    ```yaml
    name: Application CI Pipeline

    # 1. DEFINE WORKFLOW TRIGGERS
    on:
      push: # Run on pushes to these branches
        branches:
          - main
          - dev
      pull_request: # Run on pull requests targeting these branches
        branches:
          - main
          - dev

    jobs:
      # ============================================
      # == CI JOB: Validate, Build, & Push Images ==
      # ============================================
      validate_build_and_push: # Job ID
        name: Validate, Build & Push Images # Display name for the job in GitHub UI
        runs-on: ubuntu-latest # Use the latest Ubuntu runner provided by GitHub
        
        env: # Environment variables available to all steps in this job
          PYTHON_VERSION: '3.11' # Match Python version in your Dockerfiles
          # For CI testing, you might want to set Airflow/MLflow to use SQLite
          # to avoid external service dependencies for unit tests, if your code supports it.
          # Example (these would be used if tests run 'airflow' or 'mlflow' commands):
          # AIRFLOW__CORE__SQL_ALCHEMY_CONN: sqlite:////tmp/ci_airflow.db
          # MLFLOW_TRACKING_URI: sqlite:////tmp/ci_mlflow.db

        steps: # Sequence of tasks to execute
        # 2. CHECKOUT CODE
        - name: Checkout repository code
          uses: actions/checkout@v4 # GitHub Action to checkout your code

        # 3. SETUP PYTHON ENVIRONMENT
        - name: Set up Python ${{ env.PYTHON_VERSION }}
          uses: actions/setup-python@v5
          with:
            python-version: ${{ env.PYTHON_VERSION }}
            cache: 'pip' # Cache pip dependencies to speed up subsequent workflow runs

        # 4. INSTALL PYTHON DEPENDENCIES
        - name: Install Python project dependencies
          run: |
            python -m pip install --upgrade pip
            pip install -r requirements.txt
            # Ensure development and testing tools are installed if not in requirements.txt
            # These are often included in a requirements-dev.txt or installed directly here.
            pip install pytest flake8 black isort psycopg2-binary apache-airflow
    ```

**Step 3: Add Code Quality and Linting Steps to `ci-pipeline.yml`**

*   Continue adding steps under the `steps:` section of the `validate_build_and_push` job:

    ```yaml
        # ... (previous steps: checkout, setup python, install dependencies) ...

        # 5. LINTING AND FORMATTING CHECKS
        - name: Lint code with Flake8
          run: |
            # Check for major errors first (syntax errors, undefined names)
            flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
            # Then run a broader check, treating errors as warnings for reporting
            flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics

        - name: Check code formatting with Black
          run: black --check .

        - name: Check import sorting with isort
          run: isort --check-only .
    ```

**Step 4: Add Unit Testing Step to `ci-pipeline.yml`**

*   Continue adding steps:

    ```yaml
        # ... (previous steps: linting and formatting) ...

        # 6. RUN UNIT TESTS
        - name: Run Pytest unit tests
          run: |
            # Set PYTHONPATH to ensure tests can import modules from 'src' and project root
            export PYTHONPATH=$(pwd):$(pwd)/src:$PYTHONPATH
            pytest tests/
          # env: # If your tests need specific environment variables
            # YOUR_TEST_SPECIFIC_ENV_VAR: "some_value"
    ```

**Step 5: Add Airflow DAG Validation Step to `ci-pipeline.yml`**

*   Continue adding steps:

    ```yaml
        # ... (previous step: run unit tests) ...

        # 7. VALIDATE AIRFLOW DAGS
        - name: Validate Airflow DAG definitions
          run: |
            # Create a temporary AIRFLOW_HOME for this validation step
            export AIRFLOW_HOME=$(pwd)/.ci_airflow_home 
            mkdir -p $AIRFLOW_HOME

            # Set PYTHONPATH so DAGs can import custom operators/hooks from 'src/utils', etc.
            export PYTHONPATH=$(pwd):$(pwd)/src:$PYTHONPATH
            
            echo "Listing all DAGs and checking for import errors with airflow dags list --report..."
            # This command initializes Airflow's internal DB (SQLite by default if not configured)
            # and tries to parse all DAG files. It's a good overall check.
            airflow dags list --report

            echo "Individually testing DAG file importability and basic syntax..."
            # This loop provides more granular feedback if a specific DAG file has issues.
            # 'python -m ast "$dag_file"' checks for Python syntax errors.
            # 'airflow dags list-import-errors --subdir "$dag_file"' checks for Airflow-specific import issues.
            find ./airflow/dags -name "*.py" -print0 | while IFS= read -r -d $'\0' dag_file; do
              echo "--- Validating DAG file: $dag_file ---"
              if python -m ast "$dag_file"; then
                echo "Python syntax OK for $dag_file"
                # Note: list-import-errors might still report issues even if ast passes,
                # especially with complex imports or Airflow specific constructs.
                airflow dags list-import-errors --subdir "$dag_file"
              else
                echo "Python syntax ERROR in $dag_file"
                # Optionally, fail the step here: exit 1
              fi
              echo "--- Finished validating $dag_file ---"
            done
    ```
    *   `export AIRFLOW_HOME=$(pwd)/.ci_airflow_home`: This is important. Airflow commands often need `AIRFLOW_HOME` to be set. Creating a temporary one within the runner workspace prevents interference with any system-level Airflow configurations.
    *   The `find` loop tries to give more specific feedback per DAG file.

**Step 6: Add Docker Image Build and Push Steps to `ci-pipeline.yml`**

*   This is the core of building your artifacts. Continue adding steps:

    ```yaml
        # ... (previous step: validate airflow dags) ...

        # 8. DETERMINE DOCKER IMAGE TAGS BASED ON BRANCH
        - name: Determine Docker image tag suffix for floating tags
          id: docker_meta_tags # Give this step an ID to reference its outputs
          run: |
            # Default to a generic branch tag if not main or dev
            # (though subsequent 'if' conditions will prevent pushing these)
            TAG_SUFFIX="branch-${GITHUB_REF_NAME//\//-}" # Replace / with - for valid tag
            if [ "${{ github.ref }}" == "refs/heads/main" ]; then
              TAG_SUFFIX="latest"
            elif [ "${{ github.ref }}" == "refs/heads/dev" ]; then
              TAG_SUFFIX="dev"
            fi
            echo "Calculated tag suffix: $TAG_SUFFIX"
            echo "tag_suffix=$TAG_SUFFIX" >> $GITHUB_OUTPUT

        # 9. LOGIN TO DOCKER HUB
        - name: Log in to Docker Hub
          # This step only runs on 'push' events to 'main' or 'dev', not on PRs
          if: github.event_name == 'push' && (github.ref == 'refs/heads/main' || github.ref == 'refs/heads/dev')
          uses: docker/login-action@v3
          with:
            username: ${{ secrets.DOCKERHUB_USERNAME }}
            password: ${{ secrets.DOCKERHUB_TOKEN }}

        # 10. SET UP DOCKER BUILDX (for multi-platform builds and better caching)
        - name: Set up Docker Buildx
          # Only run if we intend to build and push
          if: github.event_name == 'push' && (github.ref == 'refs/heads/main' || github.ref == 'refs/heads/dev')
          uses: docker/setup-buildx-action@v3

        # 11. BUILD AND PUSH MAIN APPLICATION DOCKER IMAGE
        - name: Build and push Main App (Airflow) Docker image
          # Only build and push on actual PUSH events to 'main' or 'dev' branches
          if: github.event_name == 'push' && (github.ref == 'refs/heads/main' || github.ref == 'refs/heads/dev')
          uses: docker/build-push-action@v5
          with:
            context: . # Root of the repository is the build context
            file: ./Dockerfile # Path to your main Dockerfile
            push: true # Enable pushing to registry
            tags: | # Multiple tags for the image
              ${{ secrets.DOCKERHUB_USERNAME }}/stockpred-app:${{ github.sha }}
              ${{ secrets.DOCKERHUB_USERNAME }}/stockpred-app:${{ steps.docker_meta_tags.outputs.tag_suffix }}
            cache-from: type=gha # Use GitHub Actions cache for Docker layers
            cache-to: type=gha,mode=max # Save layers to GHA cache

        # 12. BUILD AND PUSH FRONTEND DOCKER IMAGE
        - name: Build and push Frontend Docker image
          # Only build and push on actual PUSH events to 'main' or 'dev' branches
          if: github.event_name == 'push' && (github.ref == 'refs/heads/main' || github.ref == 'refs/heads/dev')
          uses: docker/build-push-action@v5
          with:
            context: .
            file: ./Dockerfile.frontend # Path to your frontend Dockerfile
            push: true
            tags: |
              ${{ secrets.DOCKERHUB_USERNAME }}/stockpred-frontend:${{ github.sha }}
              ${{ secrets.DOCKERHUB_USERNAME }}/stockpred-frontend:${{ steps.docker_meta_tags.outputs.tag_suffix }}
            cache-from: type=gha
            cache-to: type=gha,mode=max
    ```
    *   **`Determine Docker image tag suffix` step:**
        *   This step now uses `GITHUB_REF_NAME` which gives a cleaner branch name.
        *   The output `tag_suffix` is made available via `steps.docker_meta_tags.outputs.tag_suffix`.
    *   **Conditional Execution (`if` condition):** The `docker login`, `setup-buildx`, and `build-push-action` steps are now conditional. They will **only run** if the event that triggered the workflow is a `push` (not a `pull_request` build for example) AND the push is to the `main` or `dev` branch. This prevents pushing images from every PR build.
    *   **Tags:**
        *   `<your-repo>:${{ github.sha }}`: A unique tag based on the commit SHA.
        *   `<your-repo>:${{ steps.docker_meta_tags.outputs.tag_suffix }}`: A "floating" tag (`latest` for `main`, `dev` for `dev`).
    *   **`cache-from` and `cache-to`:** Enables Docker layer caching within GitHub Actions, making subsequent builds much faster if base layers haven't changed.

**Step 7: Commit, Push, and Test the CI Pipeline**

1.  **Save `ci-pipeline.yml`.**
2.  **Add it to Git, commit, and push to your `dev` branch:**
    ```bash
    git add .github/workflows/ci-pipeline.yml
    git commit -m "feat(ci): Implement comprehensive CI pipeline for validation, build, and push"
    git push origin dev
    ```
3.  **Monitor GitHub Actions:**
    *   Go to the "Actions" tab of your repository on GitHub.
    *   You should see your "Application CI Pipeline" workflow run for the push to `dev`.
    *   Carefully examine each step's output. Debug any failures.
        *   Common issues: Typos in YAML, incorrect paths, missing Python packages for linting/testing/Airflow CLI, Dockerfile errors, Docker Hub authentication issues (double-check `DOCKERHUB_USERNAME` and `DOCKERHUB_TOKEN` secrets).
4.  **Verify Docker Hub:**
    *   If the `dev` branch push workflow succeeds, go to your Docker Hub.
    *   You should see new images in `yourdockerhubusername/stockpred-app` and `yourdockerhubusername/stockpred-frontend` tagged with the commit SHA and `:dev`.
5.  **Test Pull Request Workflow:**
    *   Create a new feature branch from `dev`: `git checkout -b feature/ci-test`
    *   Make a small, non-breaking change (e.g., add a comment to a Python file).
    *   Commit and push this feature branch: `git push origin feature/ci-test`
    *   Create a Pull Request from `feature/ci-test` to `dev`.
    *   Go to the "Actions" tab or the PR checks. The CI pipeline should run.
    *   **Verify:** Linting, testing, and DAG validation steps should run. The Docker login, setup-buildx, and build-push steps should be **SKIPPED** (because `github.event_name` is `pull_request`, not `push`, or the branch isn't `main`/`dev` for the `push` event if you tested by pushing to a feature branch directly). This is the correct behavior for PRs.
6.  **Merge PR to `dev` (Simulate Development Cycle):**
    *   If the PR checks pass, merge the `feature/ci-test` PR into `dev`.
    *   This merge will trigger a `push` event on `dev`.
    *   The CI pipeline will run again. This time, it **should** log in to Docker Hub and push new images tagged with the new commit SHA and `:dev`.
7.  **Test `main` branch Workflow (Simulate Release):**
    *   Create a Pull Request from `dev` to `main`.
    *   CI checks on the PR run (no Docker pushes).
    *   Merge the PR into `main`.
    *   This merge triggers a `push` event on `main`.
    *   The CI pipeline runs. It **should** log in to Docker Hub and push new images tagged with the commit SHA and `:latest`.

