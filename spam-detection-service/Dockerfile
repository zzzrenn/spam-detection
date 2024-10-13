FROM python:3.10.11-slim
LABEL description="SMS spam detection service"
WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install -r requirements.txt
COPY . /app/
EXPOSE 6000
CMD ["python",  "app.py"]