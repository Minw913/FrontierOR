FROM node:24-bookworm-slim@sha256:cd84903a12dbd26b46f1f3b8144a2568c41c5d37ddd0c7a80a34c7a19786b35f

ARG CODEX_VERSION=0.145.0
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates git python3 ripgrep \
    && rm -rf /var/lib/apt/lists/* \
    && npm install --global "@openai/codex@${CODEX_VERSION}"

COPY frontieror/infra/agent/codex_entrypoint.py /opt/frontieror/secure_codex_entrypoint.py
COPY frontieror/infra/agent/egress_proxy.py /opt/frontieror/secure_egress_proxy.py
COPY frontieror/infra/agent/submit.py /opt/frontieror/secure_submit.py
RUN mkdir -p /home/agent/.codex /codex-home \
    && printf '#!/bin/sh\nexec python3 /opt/frontieror/secure_submit.py "$@"\n' > /usr/local/bin/coral \
    && chmod 0555 /usr/local/bin/coral /opt/frontieror/*.py \
    && chmod 0755 /home/agent /home/agent/.codex

WORKDIR /workspace
