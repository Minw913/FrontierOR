FROM python:3.13-slim

# System deps for compiled packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ && \
    rm -rf /var/lib/apt/lists/*

# Python packages that LLM-generated code may import
RUN pip install --no-cache-dir \
    numpy scipy networkx \
    gurobipy pyomo mip cvxpy \
    ortools pandas scikit-learn

# Pre-install solution_logger utility
COPY scripts/utils/solution_logger.py /opt/bench/solution_logger.py
ENV PYTHONPATH="/opt/bench:${PYTHONPATH}"

# Gurobi license: mount at runtime via -v /path/to/gurobi.lic:/opt/gurobi/gurobi.lic
ENV GRB_LICENSE_FILE=/opt/gurobi/gurobi.lic

WORKDIR /workspace
