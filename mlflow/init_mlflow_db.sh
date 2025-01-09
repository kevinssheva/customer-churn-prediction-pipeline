#!/bin/bash

# Set environment variables
POSTGRES_USER=airflow
POSTGRES_PASSWORD=airflow
POSTGRES_DB=airflow

NEW_DB=mlflow
NEW_USER=mlflow
NEW_PASSWORD=mlflow

# Wait for PostgreSQL to be ready
echo "Waiting for PostgreSQL to start..."
until pg_isready -h postgres -U "$POSTGRES_USER"; do
  sleep 1
done
echo "PostgreSQL is ready!"

# Create new database and user
echo "Creating database '$NEW_DB' and user '$NEW_USER'..."
PGPASSWORD="airflow" psql -h postgres -U "$POSTGRES_USER" <<EOSQL
  -- Create the database if it does not exist
  SELECT 'CREATE DATABASE $NEW_DB'
  WHERE NOT EXISTS (
    SELECT FROM pg_database WHERE datname = '$NEW_DB'
  )\gexec

  -- Create the user if it does not exist
  SELECT 'CREATE USER $NEW_USER WITH PASSWORD ''$NEW_PASSWORD'''
  WHERE NOT EXISTS (
    SELECT FROM pg_roles WHERE rolname = '$NEW_USER'
  )\gexec

  -- Grant privileges
  GRANT ALL PRIVILEGES ON DATABASE $NEW_DB TO $NEW_USER;
EOSQL

echo "Database '$NEW_DB' and user '$NEW_USER' initialized successfully!"
