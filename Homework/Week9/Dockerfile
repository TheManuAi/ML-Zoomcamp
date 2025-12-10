FROM agrigorev/model-2025-hairstyle:v1

# Install required Python packages
RUN pip install --no-cache-dir \
    numpy \
    pillow \
    onnxruntime

# Copy lambda function
COPY lambda_function.py .

# Set the handler
CMD ["lambda_function.lambda_handler"]
