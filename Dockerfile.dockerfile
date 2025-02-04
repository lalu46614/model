# Use official Python image
FROM python:3.9

# Set the working directory
WORKDIR /app

# Copy all files into the container
COPY . .

# Install dependencies
RUN pip install --no-cache-dir fastapi uvicorn torch torchvision ultralytics opencv-python

# Expose the port FastAPI runs on
EXPOSE 8080

# Command to run FastAPI
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
