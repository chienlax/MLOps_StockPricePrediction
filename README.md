# MLOps Project: Stock Price Prediction

Deadline: 18/04/2025

For Docker setup and Airflow configuration, please refer to the [docker_setup.md](project_notice/01_docker_setup.md) file.

For a detailed guide on setting up and running the project, go to [instruction.md](project_notice/02_instruction.md) file.

For tasks and their descriptions, check the [tasks.md](project_notice/03_tasks.md) file.


Run Dockerfile.cuda for cuda

```bash
sudo docker build -f Dockerfile.cuda -t stockpred-cuda-image .
```

```bash
sudo docker run --gpus all -p 8000:8000 stockpred-cuda-image
```