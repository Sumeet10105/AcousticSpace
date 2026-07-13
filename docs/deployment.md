# Deployment Guide

## Overview

This guide covers deploying AcousticSpace models to production environments.

## Prerequisites

- Docker and Docker Compose installed
- NVIDIA GPU with CUDA support (optional)
- 4GB+ RAM
- 10GB+ disk space

## Local Deployment

### Development Server
```bash
# Install dependencies
pip install -r requirements.txt

# Run API server
python -m uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

### Production Server
```bash
# Install dependencies
pip install -r requirements.txt

# Run with Gunicorn
gunicorn -w 4 -k uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000 \
  --timeout 120 \
  src.api.main:app
```

## Docker Deployment

### Build Image
```bash
docker build -t acousticspace:latest .
```

### Run Container
```bash
docker run -d \
  --name acousticspace-api \
  -p 8000:8000 \
  -e CUDA_VISIBLE_DEVICES=0 \
  -v $(pwd)/saved_models:/app/saved_models \
  -v $(pwd)/outputs:/app/outputs \
  acousticspace:latest
```

### Docker Compose
```bash
docker-compose up -d
```

### Health Check
```bash
curl http://localhost:8000/health
```

## Cloud Deployment

### AWS
```bash
# Push to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin \
  123456789.dkr.ecr.us-east-1.amazonaws.com

docker tag acousticspace:latest \
  123456789.dkr.ecr.us-east-1.amazonaws.com/acousticspace:latest

docker push 123456789.dkr.ecr.us-east-1.amazonaws.com/acousticspace:latest

# Deploy on ECS or EC2
```

### Google Cloud
```bash
# Push to GCR
docker tag acousticspace:latest \
  gcr.io/project-id/acousticspace:latest

docker push gcr.io/project-id/acousticspace:latest

# Deploy on Cloud Run
gcloud run deploy acousticspace \
  --image gcr.io/project-id/acousticspace:latest \
  --platform managed
```

## Configuration

### Environment Variables
```bash
# .env file
CUDA_VISIBLE_DEVICES=0
API_HOST=0.0.0.0
API_PORT=8000
MODEL_PATH=./saved_models/best_model.pt
LOG_LEVEL=INFO
```

### Model Loading
```bash
# Copy model to container volume
docker cp saved_models/best_model.pt acousticspace-api:/app/saved_models/
```

## Monitoring

### Logging
```bash
docker logs -f acousticspace-api
```

### Metrics
```bash
# Access Prometheus metrics
curl http://localhost:8000/metrics
```

### Health Monitoring
```bash
# Regular health checks
while true; do
  curl http://localhost:8000/health
  sleep 60
done
```

## Performance Tuning

### GPU Optimization
- Enable mixed precision training
- Use TensorRT for inference optimization
- Implement batch prediction

### CPU Optimization
- Use ONNX runtime
- Quantize model weights
- Reduce model size with distillation

### Load Balancing
```bash
# Use Nginx for load balancing
upstream acousticspace_api {
  server api1:8000;
  server api2:8000;
  server api3:8000;
}

server {
  listen 80;
  location / {
    proxy_pass http://acousticspace_api;
  }
}
```

## Security

### API Authentication
- Implement JWT tokens
- Rate limiting (Slowhammer)
- CORS configuration

### Model Security
- Encrypt model files
- Version control with hashing
- Monitor unauthorized access

### Data Security
- HTTPS only
- Input validation
- Sanitize file uploads

## Troubleshooting

### OOM (Out of Memory)
- Reduce batch size
- Use model quantization
- Enable gradient checkpointing

### Slow Inference
- Check GPU utilization: `nvidia-smi`
- Enable batch processing
- Use model caching

### Failed Requests
- Check logs: `docker logs`
- Verify model path
- Test with sample audio

## Backup and Recovery

### Model Checkpoints
```bash
# Backup models
tar -czf models_backup.tar.gz saved_models/

# Upload to S3
aws s3 cp models_backup.tar.gz s3://bucket/backups/
```

### Recovery
```bash
# Restore from backup
aws s3 cp s3://bucket/backups/models_backup.tar.gz .
tar -xzf models_backup.tar.gz
```

## Rollback

```bash
# Deploy previous version
docker run -d \
  --name acousticspace-api-v1 \
  -p 8001:8000 \
  acousticspace:v1.0

# Switch traffic
docker stop acousticspace-api
docker rename acousticspace-api-v1 acousticspace-api
```
