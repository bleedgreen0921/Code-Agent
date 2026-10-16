FROM python:3.12.14-slim-bookworm@sha256:392307d22300de8b5986851a12d9176dfc0fc073e65bf6523ebd7dcbeb23564e
COPY numpy-*.whl /tmp/wheels/
RUN pip install --no-index --no-cache-dir /tmp/wheels/numpy-*.whl && rm -rf /tmp/wheels
ENV PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
WORKDIR /tmp
ENTRYPOINT ["python", "-B", "/checks/evaluator.py"]
