# Use an official Python runtime as a parent image
FROM python:3.11.2-slim

# Set the working directory
WORKDIR /app

# Copy application files
COPY requirements.txt .

# Install dependencies
RUN pip3 install -r requirements.txt

# Run Gunicorn WSGI server with 4 worker processes
CMD ["uvicorn", "main:app", "--host=0.0.0.0", "--port=8000"]