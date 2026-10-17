FROM python:3.12.14-slim-bookworm@sha256:392307d22300de8b5986851a12d9176dfc0fc073e65bf6523ebd7dcbeb23564e
# Debian snapshot fixes the package universe as well as the Python base image.
RUN rm -f /etc/apt/sources.list.d/debian.sources && \
    printf 'deb [check-valid-until=no] http://snapshot.debian.org/archive/debian/20261001T000000Z bookworm main\n' > /etc/apt/sources.list && \
    apt-get update && apt-get install -y --no-install-recommends bash ripgrep git && \
    rm -rf /var/lib/apt/lists/*
COPY requirements.lock /opt/agent/requirements.lock
RUN pip install --no-cache-dir -r /opt/agent/requirements.lock
COPY execution/git_tools.py execution/helper.py execution/entrypoint.py execution/background.py /opt/agent/
COPY execution/checks /opt/agent/checks
RUN chmod 0555 /opt/agent/checks && chmod 0444 /opt/agent/checks/*.py
LABEL io.agent-framework.checks="python-repair-v1"
LABEL io.agent-framework.helper="workspace-v2" io.agent-framework.background="background-v1"
ENV PATH="/usr/local/bin:/usr/bin:/bin" LANG="C.UTF-8" HOME="/tmp" TMPDIR="/tmp"
WORKDIR /workspace
ENTRYPOINT ["/usr/local/bin/python", "-I", "/opt/agent/entrypoint.py"]
