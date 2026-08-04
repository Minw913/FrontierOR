FROM python:3.11-slim-bookworm@sha256:b18992999dbe963a45a8a4da40ac2b1975be1a776d939d098c647482bcad5cba

COPY frontieror/infra/agent/model_proxy.py /opt/frontieror/secure_model_proxy.py
RUN chmod 0555 /opt/frontieror/secure_model_proxy.py

CMD ["python3", "/opt/frontieror/secure_model_proxy.py"]
