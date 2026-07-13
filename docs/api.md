# API Documentation

## Overview

The AcousticSpace API provides endpoints for audio spoofing detection. It's built with FastAPI and can be deployed using Docker.

## Base URL
```
http://localhost:8000
```

## Endpoints

### Health Check
**GET** `/health`

Check if the API is running and healthy.

**Response:**
```json
{
  "status": "healthy",
  "version": "0.1.0"
}
```

### Predict
**POST** `/predict`

Detect if an audio file is real or spoofed.

**Request:**
- Content-Type: `multipart/form-data`
- file: Audio file (WAV, MP3, FLAC)

**Response:**
```json
{
  "prediction": "real",
  "confidence": 0.95,
  "score": 0.87,
  "timestamp": "2024-01-15T10:30:00Z"
}
```

**Response Codes:**
- 200: Success
- 400: Bad request
- 422: Invalid audio format
- 500: Server error

### List Models
**GET** `/models`

List available models.

**Response:**
```json
{
  "models": [
    {
      "name": "ast_best",
      "version": "1.0",
      "accuracy": 0.95
    }
  ]
}
```

## Running the API

### Locally
```bash
uvicorn src.api.main:app --reload
```

### Docker
```bash
docker-compose up
```

## Authentication

Currently no authentication is required. This should be added for production deployments.

## Rate Limiting

Rate limiting should be implemented for production deployments.

## Error Handling

All errors return a JSON response with error details and appropriate HTTP status codes.

## Documentation

Interactive API documentation is available at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`
