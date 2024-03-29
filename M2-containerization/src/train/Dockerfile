FROM ubuntu:22.04

# Set environment variables to avoid user interaction during installation
ENV DEBIAN_FRONTEND=noninteractive

# Install software-properties-common to add repositories
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    jq \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set ARGs for user ID and group ID
ARG USER_ID
ARG GROUP_ID

# Create a group and user
RUN groupadd --gid $GROUP_ID user || groupmod -n user $(getent group $GROUP_ID | cut -d: -f1) && \
    useradd --create-home --no-log-init --shell /bin/bash --uid $USER_ID --gid $GROUP_ID user

# Switch to the non-root user
USER user

# Set the working directory
WORKDIR /app

# Copy the project files into the container
COPY --chown=user:user ./data/raw /app/data/raw
COPY --chown=user:user ./data/train /app/data/train
COPY --chown=user:user ./src/train /app/src/train
COPY --chown=user:user ./src/preprocess.py /app/src
COPY --chown=user:user ./src/utils.py /app/src
COPY --chown=user:user settings.json ./
COPY --chown=user:user requirements.txt ./
COPY --chown=user:user entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# Set permissions for the entire /app directory
# Give read, write, and execute permissions to the owner and group for all directories
RUN find /app -type d -exec chmod 775 {} \;

# Give read and write permissions to the owner and group for all files
RUN find /app -type f -exec chmod 664 {} \;

# Retrieve environment variables from settings.json
RUN jq -r '. | to_entries | .[] | "\(.key)=\(.value)"' settings.json > .env \
    && echo "source .env" >> ~/.bashrc

# Install Python dependencies
RUN pip install --user -r requirements.txt

# Set the entrypoint to the training script
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["python3", "/app/src/train/train.py"]