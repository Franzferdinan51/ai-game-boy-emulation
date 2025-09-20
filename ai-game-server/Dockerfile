# Use an official Python runtime as the base image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Install the required packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Expose the port the app runs on
EXPOSE 5000

# Set environment variables (these should be overridden at runtime)
ENV GEMINI_API_KEY=""
ENV OPENROUTER_API_KEY=""
ENV NVIDIA_API_KEY=""

# Run the application
CMD ["python", "src/main.py"]