FROM python:3.11-slim

WORKDIR /app

RUN pip install uv

COPY . .

RUN uv sync --frozen --no-dev

CMD ["uv", "run", "python", "docker_entrypoint.py"]
