# Stock Market Data Pipeline

This project is a data pipeline that collects stock market data using the YFinance library, streams it to Kafka, processes the data using Python, stores it in BigQuery, and creates reports in Power BI. Additionally, it includes logging functionality to monitor the pipeline.

## Prerequisites

Before you begin, ensure you have met the following requirements:

- Python +3.9 installed
- Access to a Kafka cluster
- Access to Google BigQuery
- Access to Power BI
- yfinance library for collecting stock data

## Setup

### 1. Kafka Cluster Setup

To stream data to and from the Kafka cluster, you need to set up a Kafka broker and ZooKeeper. You can use Docker to create a local Kafka cluster for development and testing. Follow these steps to set up the Kafka cluster using Docker:

1. **Install Docker**: If you haven't already installed Docker, you can download it from [Docker's official website](https://docs.docker.com/get-docker/).

2. **Docker Compose Configuration**: run the `docker-compose.yml` file within the repo. using "docker-compose up -d"

### 2. Collecting Data with yfinance

The project utilizes the yfinance library to collect stock market data. This data is then streamed to the Kafka cluster for further processing.

### 3. Data Streaming

Python Kafka producers are employed to publish collected stock data to a Kafka topic. This ensures that the data is available for processing and analysis in real-time.

### 4. Data Processing

Data processing is a critical step in the pipeline. A Python application is built to consume data from the Kafka topic. This component is highly flexible and can accommodate various data processing tasks, such as data transformation and enrichment.

### 5. Storing Data in BigQuery

Processed data is stored in Google BigQuery, a highly scalable and performant data warehousing solution. The project provides guidance on setting up BigQuery, creating datasets, and defining tables for efficient data storage.


## Acknowledgments

The project may have been inspired by or may use libraries, tools, or resources from the open-source community. Any acknowledgments and credits for such contributions should be included in this section.

