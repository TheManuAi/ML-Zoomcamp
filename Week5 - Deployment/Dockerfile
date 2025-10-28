FROM python:3.13-slim

RUN pip install pipenv

WORKDIR /app

COPY ["Pipfile", "Pipfile.lock", "./"]

RUN pipenv install --system --deploy

COPY ["predict.py", "model_C=1.0.bin", "./"]

EXPOSE 1234

ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:1234", "predict:app"]