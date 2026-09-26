# The Shot Tracker web app as one container, for a server on the internet
# (see "PUT IT ON YOUR PHONE.txt").  At home, start.py does the same job.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # Large frames go straight back to the system when freed, so memory does
    # not creep up from one analysis to the next (a 4K clip peaks at ~1.6 GB).
    MALLOC_MMAP_THRESHOLD_=1048576 \
    SHOTTRACKER_DATA=/data \
    SHOTTRACKER_MAX_UPLOAD_MB=3000 \
    PYTHONPATH=/app/src:/app/server \
    PORT=8080

WORKDIR /app
COPY deploy/requirements.txt deploy/requirements.txt
RUN pip install -r deploy/requirements.txt
COPY src src
COPY server server
COPY web web

# Sessions and clips live in /data: a volume the host keeps between restarts.
RUN mkdir -p /data
EXPOSE 8080
# --proxy-headers: behind the host's HTTPS proxy, so the login cookie is
# marked secure.
CMD ["sh", "-c", "exec uvicorn app:app --host 0.0.0.0 --port ${PORT} --proxy-headers --forwarded-allow-ips='*'"]
