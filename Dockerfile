# Use a lightweight Python base image
FROM python:3.10-slim

# Set working directory inside the container
WORKDIR /app

# Copy only the requirements file first for layer caching
COPY requirement.txt .

# Install Python dependencies (no cache to keep image small)
RUN pip install --no-cache-dir -r requirement.txt

# Copy the rest of your project files into the container
COPY . .

# Expose the FastAPI port
EXPOSE 8000

# Run the FastAPI app using uvicorn
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
