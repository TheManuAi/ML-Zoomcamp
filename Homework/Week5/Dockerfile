FROM agrigorev/zoomcamp-model:2025

COPY pyproject.toml uv.lock ./

RUN pip install uv

RUN uv sync --frozen

COPY serve_q6.py .

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "serve_q6:app", "--host", "0.0.0.0", "--port", "8000"]