## How To Run

1. Copy `.env.example` into `.env`

2. Run `airflow-init` docker compose to setup database
```bash
docker compose up airflow-init
```

3. Run containers using docker compose, may take a while, should not show any errors though
```bash
docker compose up --build -d
```

4. Stop docker containers, use `--volumes` flag to remove database & object storage (minio) data, use `--rmi local` to remove locally built images
```bash
docker compose down
# or
docker compose down --volumes
# or
docker compose down --volumes --rmi local
```

## References

- Spark Docker Compose Configuration [https://medium.com/@SaphE/testing-apache-spark-locally-docker-compose-and-kubernetes-deployment-94d35a54f222]
- Apache Airflow Docker Tutorial [https://airflow.apache.org/docs/apache-airflow/stable/howto/docker-compose/index.html]
- MLFlow Docker Compose Example [https://github.com/sachua/mlflow-docker-compose]
