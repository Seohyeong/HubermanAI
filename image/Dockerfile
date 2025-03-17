FROM public.ecr.aws/lambda/python:3.10

# Copy requirements.txt
COPY requirements.txt ${LAMBDA_TASK_ROOT}

# Required to make SQLlite3 work for Chroma.
RUN pip install pysqlite3-binary

# Install the specified packages
RUN pip install --no-cache-dir -r requirements.txt --upgrade

# For local testing.
EXPOSE 8000

# Set IS_USING_IMAGE_RUNTIME Environment Variable
ENV IS_USING_IMAGE_RUNTIME=True

# copy code
COPY config ${LAMBDA_TASK_ROOT}/config
COPY utils ${LAMBDA_TASK_ROOT}/utils
COPY main.py ${LAMBDA_TASK_ROOT}/
COPY rag.py ${LAMBDA_TASK_ROOT}/

# Copy data folder
COPY chroma_db_docs ${LAMBDA_TASK_ROOT}/chroma_db_docs
COPY chroma_db_query ${LAMBDA_TASK_ROOT}/chroma_db_query

CMD ["main.handler"]