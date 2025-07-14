# How to use:
#   docker build --tag genre:latest .

FROM python:3.10

# Clone and install patched fairseq
RUN git clone -b fixing_prefix_allowed_tokens_fn --single-branch https://github.com/nicola-decao/fairseq.git

# Set PYTHONPATH to make examples/ available for import
ENV PYTHONPATH="/fairseq:$PYTHONPATH"

RUN pip install -e ./fairseq

WORKDIR /GENRE

# Install Python dependencies with specific versions
RUN pip install --no-cache-dir \
    "numpy<1.24" \
    "torch<2.5.1" \
    fastapi \
    "uvicorn[standard]" \
    requests \
    beautifulsoup4

# Copy GENRE code
COPY setup.py .
COPY README.md .
COPY genre ./genre
RUN pip install -e .

# Make model directory available inside container
VOLUME ["/models"]
VOLUME ["/data"]

# Expose the FastAPI port
EXPOSE 5000

# Start FastAPI app when container runs
CMD ["uvicorn", "genre.server:app", "--host", "0.0.0.0", "--port", "5000"]
