FROM python:3.9-slim

WORKDIR /app

# Install dependencies for numpy and pandas
RUN apt-get update && apt-get install -y \
    build-essential \
    libssl-dev \
    gcc \
    python3-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install dependencies 
RUN pip install --no-cache-dir -r requirements.txt

# Create uploads directory
RUN mkdir -p uploads

# Copy the rest of the application
COPY . .

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8080

EXPOSE 8080

# Run gunicorn
CMD gunicorn --workers=2 --bind 0.0.0.0:$PORT app:app 