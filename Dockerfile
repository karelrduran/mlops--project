FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt requirements.txt
COPY . .


RUN pip install --upgrade pip
RUN pip install -r requirements.txt

RUN chmod +x start.sh

EXPOSE 5000

ENTRYPOINT ["./start.sh"]

#COPY mlflow-service/mlflow-server.service /etc/systemd/system/
#
#CMD ["sudo", "systemctl", "start", "mlflow-server.service"]
#
#
#CMD ["python", "mlflow-optuna.py"]
