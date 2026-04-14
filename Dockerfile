FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY pyproject.toml README.md ./
COPY src ./src
COPY data ./data

RUN pip install --no-cache-dir -e .

CMD ["bioml-localization", "train", "--data", "data/raw/protein_localization_sample.csv", "--out", "artifacts"]
