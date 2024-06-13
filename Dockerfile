FROM python:3.10-slim
WORKDIR /app
COPY bin bin
RUN pip3 install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip3 install --no-cache-dir Flask pandas numpy scipy transformers
COPY app.py .
EXPOSE 5000
ENTRYPOINT ["python3", "app.py"]