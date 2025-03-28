# src/Dockerfile
FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y poppler-utils

# Copy the requirements file and install dependencies
# COPY src/requirements.txt .
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --upgrade faiss-cpu
RUN pip install --upgrade pillow
RUN pip install --upgrade pdf2image

# Copy the application code
COPY . .

# Expose the port Streamlit will run on
EXPOSE 8501

# Command to run the application
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]