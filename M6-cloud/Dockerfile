FROM python:3.10

# Install necessary packages
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*


# Create directories for SageMaker
RUN mkdir -p /opt/ml/code /opt/ml/model

# Copy the application code
COPY ./app /opt/ml/code
COPY requirements.txt /opt/ml

# Install pip packages
RUN pip install --no-cache-dir -r /opt/ml/requirements.txt

# Set the working directory
WORKDIR /opt/ml/code

# Expose the port the app runs on
EXPOSE 8080

# Set the entry point to start the Flask application
ENTRYPOINT ["python", "serve"]
