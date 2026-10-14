# Mock-only image: runs the test suite and the MOCK demo with no GPU and no weights.
# For a real study run the harness on the host (or mount your config and point base_url at a
# host-reachable llama-server, for example http://host.docker.internal:8080/v1).
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-cache-dir -e ".[dev]"

COPY configs ./configs
COPY docs ./docs
COPY scripts ./scripts
COPY suites ./suites
COPY tests ./tests
COPY results ./results

CMD ["refusalscope", "mock-demo", "--out", "/tmp/mock-demo"]
