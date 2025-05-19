FROM python:3.10-slim

WORKDIR /app

# Install system deps (optional for performance)
RUN apt-get update && apt-get install -y build-essential && apt-get clean

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "src/tune.py"]

