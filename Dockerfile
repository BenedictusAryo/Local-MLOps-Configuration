ARG PYTHON_VERSION=3.12
FROM python:$PYTHON_VERSION-slim-bookworm
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app
COPY . /app
# Install litserve and requirements
RUN uv sync --frozen

RUN uv pip install uvloop
EXPOSE 8000
CMD ["python", "/app/.\src\api\api_model.py"]
