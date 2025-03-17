# Stage 1: Builder - Install dependencies
FROM public.ecr.aws/lambda/python:3.10 AS builder

WORKDIR /build

# Copy requirements.txt
COPY requirements.txt .

# Install dependencies into a custom directory
RUN pip install --no-cache-dir -r requirements.txt --upgrade -t /build/lib

# Stage 2: Runtime - Create the final image
FROM public.ecr.aws/lambda/python:3.10

WORKDIR ${LAMBDA_TASK_ROOT}

# Copy only the installed libraries from the builder stage
COPY --from=builder /build/lib ./lib

# Copy application code
COPY config ./config
COPY utils ./utils
COPY main.py .
COPY rag.py .

# Copy data folders (optional, consider S3 offloading)
COPY chroma_db_docs ./chroma_db_docs
COPY chroma_db_query ./chroma_db_query

# Set PYTHONPATH to include the copied libraries
ENV PYTHONPATH=${LAMBDA_TASK_ROOT}/lib

# Expose port for local testing
EXPOSE 8000

# Set environment variable
ENV IS_USING_IMAGE_RUNTIME=True

# Set Lambda handler
CMD ["main.handler"]