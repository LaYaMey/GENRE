# How to use:
#   docker build --tag genre:latest .
#   docker run --rm -it genre:latest /bin/bash
#   docker run --rm -it -v $(pwd)/tests:/GENRE/genre/tests genre:latest /bin/bash
#   pytest genre/tests
FROM python:3.8

WORKDIR /GENRE/

# Install system dependencies
RUN apt-get update && \
    apt-get install -y git axel && \
    apt-get clean

# Install Python dependencies with specific versions
RUN pip install --no-cache-dir \
    numpy==1.23 \
    torch==2.5.1 \
    fastapi \
    uvicorn[standard] \
    requests

# Clone and install patched fairseq
RUN git clone -b fixing_prefix_allowed_tokens_fn --single-branch https://github.com/nicola-decao/fairseq.git
RUN pip install -e ./fairseq

# Copy GENRE code
COPY setup.py .
COPY genre ./genre
RUN pip install -e .

# Make model directory available inside container
VOLUME ["/models"]

# Expose the FastAPI port
EXPOSE 5000

# Start FastAPI app when container runs
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "5000"]
