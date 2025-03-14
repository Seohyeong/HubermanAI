FROM public.ecr.aws/lambda/python:3.10

# Copy requirements.txt
COPY requirements.txt ${LAMBDA_TASK_ROOT}

# Required to make SQLlite3 work for Chroma.
RUN pip install pysqlite3-binary

# Install the specified packages
RUN pip install -r requirements.txt --upgrade

# For local testing.
EXPOSE 8000

# Set IS_USING_IMAGE_RUNTIME Environment Variable
ENV IS_USING_IMAGE_RUNTIME=True

# Copy files from api folder, excluding data
COPY api/__init__.py ${LAMBDA_TASK_ROOT}/api/
COPY api/utils ${LAMBDA_TASK_ROOT}/api/utils
COPY api/logger_config.py ${LAMBDA_TASK_ROOT}/api/
COPY api/main.py ${LAMBDA_TASK_ROOT}/api/
COPY api/model_config.py ${LAMBDA_TASK_ROOT}/api/
COPY api/rag.py ${LAMBDA_TASK_ROOT}/api/

# Copy app folder
COPY app ${LAMBDA_TASK_ROOT}/app

# Copy data folder
COPY chroma_db_docs ${LAMBDA_TASK_ROOT}/chroma_db_docs
COPY chroma_db_query ${chroma_db_docs}/chroma_db_query