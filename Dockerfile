FROM python:3.11-slim
WORKDIR /work
ENV PYTHONPATH=/work
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY tests/ ./tests/
CMD ["python", "scripts/run_study.py", "--mode", "fast"]
