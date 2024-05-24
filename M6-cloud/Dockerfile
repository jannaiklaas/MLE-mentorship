FROM python:3.10

# Install necessary packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    curl \
    nginx \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /opt

# Install pip packages
RUN pip install --no-cache-dir -r /opt/requirements.txt

ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/program:${PATH}"

# Set the working directory
COPY app /opt/program
COPY models /opt/program/models
WORKDIR /opt/program

# Expose the port the app runs on
EXPOSE 8080

# Set the entry point to start the Flask application
ENTRYPOINT ["python", "serve"]
