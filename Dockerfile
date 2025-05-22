# Use a slim Python base image
FROM python:3.11-slim

# Set working directory in container
WORKDIR /app

# Copy all project files to /app
COPY . .

# Install system dependencies required to build psutil and other packages
RUN apt-get update && apt-get install -y gcc python3-dev build-essential

# Install Python packages from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expose FastAPI port
EXPOSE 8000

# Command to run the FastAPI app
CMD ["uvicorn", "lab3_fastapi:app", "--host", "0.0.0.0", "--port", "8000"]