FROM python:3.10-slim-bookworm
RUN apt-get update && apt-get install -y --no-install-recommends g++ \
    && apt-get clean && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt \
    && python -m pip install --no-cache-dir 'torch==2.7.0+cu128' --extra-index-url https://download.pytorch.org/whl/cu128
COPY bspnet bspnet
COPY setup.py verify_weights.py ./
RUN python setup.py build_ext --inplace
COPY tests tests
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 NUMBA_CACHE_DIR=/tmp/numba
ENTRYPOINT ["python", "-m", "bspnet.run"]
