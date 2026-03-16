# Base Image
FROM python:3.11-slim

# Thiết lập thư mục làm việc
WORKDIR /app

# Sao chép requirements và cài đặt
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Sao chép mã nguồn và các thư mục cần thiết
COPY src/ /app/src/
COPY utils/ /app/utils/
COPY schema/ /app/schema/
COPY config.yaml /app/config.yaml

# Sao chép model mới nhất (Giả sử thư mục models chứa model đã train)
# Trong môi trường AWS thực tế, mô hình thường được kéo từ S3 vào /opt/ml/model/
COPY models/ /app/models/

# Thiết lập biến môi trường
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Chạy lệnh (Ví dụ: Chạy Flask/Gunicorn cho SageMaker Endpoint)
# EXPOSE 8080
# CMD ["gunicorn", "--bind", "0.0.0.0:8080", "src.api.serve:app"]
