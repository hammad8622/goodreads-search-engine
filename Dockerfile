FROM python:3.11-slim

WORKDIR /app

# install deps first so they get cached
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy app + dataset
COPY app.py .
COPY books.csv .

EXPOSE 7860

CMD ["python", "app.py"]
