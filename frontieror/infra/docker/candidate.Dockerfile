FROM python:3.13-slim@sha256:6771159cd4fa5d9bba1258caf0b82e6b73458c694d178ad97c5e925c2d0e1a91

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ ca-certificates && \
    rm -rf /var/lib/apt/lists/*

COPY frontieror/infra/docker/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY scripts/utils/solution_logger.py /opt/bench/solution_logger.py
COPY frontieror/infra/wls_proxy.py /opt/bench/restricted_egress_proxy.py
ENV PYTHONPATH=/opt/bench
ENV GRB_LICENSE_FILE=/opt/gurobi/gurobi.lic

WORKDIR /workspace
