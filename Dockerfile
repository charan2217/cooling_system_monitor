FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY *.py ./

# Copy sample CSV data
COPY sample_obd_data.csv ./

# Create output directory
RUN mkdir -p /app/output

# Default command: process sample CSV and emit log and JSON snapshot
CMD ["python", "main.py", "--input-csv", "sample_obd_data.csv", "--output-dir", "/app/output"]
