# Docker Setup Guide for SpeechEmotion-LSTM

This guide explains how to run the SpeechEmotion-LSTM application using Docker.

## 📋 Prerequisites

- Docker (version 20.10 or higher)
- Docker Compose (version 2.0 or higher)

## 🏗️ Architecture

The application consists of two services:

1. **Backend** (`src/api/`): FastAPI server with WebSocket support
   - Handles audio processing, MFCC extraction, and emotion prediction
   - Runs on port 8000
   
2. **Frontend** (`src/site/`): React + Vite application
   - User interface for audio upload and visualization
   - Communicates with backend via WebSockets
   - Production: runs on port 3000 (nginx)
   - Development: runs on port 5173 (Vite dev server)

## 🚀 Quick Start

### Production Mode

Build and run both services in production mode:

```bash
# Build and start containers
docker-compose up --build

# Or run in detached mode
docker-compose up -d --build
```

Access the application:
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- WebSocket: ws://localhost:8000/ws

### Development Mode (with Hot Reload)

For development with automatic code reloading:

```bash
# Build and start in development mode
docker-compose -f docker-compose.dev.yml up --build

# Or run in detached mode
docker-compose -f docker-compose.dev.yml up -d --build
```

Access the application:
- Frontend (Vite dev): http://localhost:5173
- Backend API: http://localhost:8000

## 🛠️ Docker Files Explained

### `Dockerfile.backend`
- Multi-stage Python container for the FastAPI backend
- Installs system dependencies (libsndfile, ffmpeg) for audio processing
- Uses uvicorn with hot-reload for development

### `Dockerfile.frontend` (Production)
- Multi-stage build: builds React app, then serves with nginx
- Optimized for production with gzip compression
- Smaller final image size

### `Dockerfile.frontend.dev` (Development)
- Single-stage container running Vite dev server
- Supports hot module replacement (HMR)
- Larger but better for development

### `docker-compose.yml` (Production)
- Orchestrates both services
- Frontend served via nginx on port 3000
- Configured for production deployment

### `docker-compose.dev.yml` (Development)
- Development version with volume mounts for hot reload
- Vite dev server on port 5173
- Source code changes reflected immediately

## 📝 Common Commands

### Start services
```bash
# Production
docker-compose up

# Development
docker-compose -f docker-compose.dev.yml up
```

### Stop services
```bash
# Production
docker-compose down

# Development
docker-compose -f docker-compose.dev.yml down
```

### View logs
```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f backend
docker-compose logs -f frontend
```

### Rebuild containers (after dependency changes)
```bash
# Production
docker-compose up --build

# Development
docker-compose -f docker-compose.dev.yml up --build
```

### Access container shell
```bash
# Backend
docker exec -it speechemotion-backend bash

# Frontend (production)
docker exec -it speechemotion-frontend sh

# Frontend (development)
docker exec -it speechemotion-frontend-dev sh
```

## 🔧 Configuration

### Environment Variables

Copy the example environment file:
```bash
cp .env.example .env
```

Edit `.env` to customize:
- `BACKEND_PORT`: Backend API port (default: 8000)
- `FRONTEND_PORT`: Frontend port in production (default: 3000)
- `FRONTEND_DEV_PORT`: Frontend dev server port (default: 5173)
- `VITE_WS_URL`: WebSocket URL for frontend (default: ws://localhost:8000/ws)

### WebSocket Configuration

The frontend connects to the backend WebSocket using the `VITE_WS_URL` environment variable.

**For local development (outside Docker):**
```bash
VITE_WS_URL=ws://localhost:8000/ws
```

**For Docker development:**
```bash
VITE_WS_URL=ws://localhost:8000/ws
```

**For production deployment:**
Update the URL to match your domain:
```bash
VITE_WS_URL=wss://yourdomain.com/ws
```

## 📦 Volume Mounts

### Development Mode
The following directories are mounted for hot reload:
- `./src/api` → `/app/src/api` (Backend source)
- `./src/site` → `/app` (Frontend source)
- `./training` → `/app/training` (Model checkpoints)
- `./data` → `/app/data` (Audio data)

### Production Mode
Only necessary data is mounted:
- `./training` → `/app/training` (Model checkpoints - read-only)
- `./data` → `/app/data` (Audio data - if needed)

## 🐛 Troubleshooting

### WebSocket Connection Fails

1. Ensure backend is running:
   ```bash
   docker-compose logs backend
   ```

2. Check if port 8000 is accessible:
   ```bash
   curl http://localhost:8000
   ```

3. Verify WebSocket URL in browser console

### Frontend Can't Connect to Backend

1. Check docker network:
   ```bash
   docker network inspect speechemotion-lstm_speechemotion-network
   ```

2. Ensure both containers are on the same network:
   ```bash
   docker-compose ps
   ```

### Hot Reload Not Working (Development)

1. Ensure you're using the dev compose file:
   ```bash
   docker-compose -f docker-compose.dev.yml up
   ```

2. Check volume mounts:
   ```bash
   docker inspect speechemotion-backend-dev
   ```

### Model File Not Found

1. Ensure `model.pth` exists in `src/api/` or `training/checkpoints/`
2. Check volume mounts include the model path
3. Update `MODEL_PATH` in `.env` if needed

### Build Fails - Dependency Issues

1. Clear Docker cache:
   ```bash
   docker-compose down -v
   docker system prune -a
   ```

2. Rebuild from scratch:
   ```bash
   docker-compose build --no-cache
   ```

### Port Already in Use

If ports 8000, 3000, or 5173 are in use:

1. Find the process:
   ```bash
   lsof -i :8000  # or :3000, :5173
   ```

2. Stop the process or change ports in `docker-compose.yml`

## 🌐 Production Deployment

### With Domain Name

1. Update `VITE_WS_URL` in `.env`:
   ```bash
   VITE_WS_URL=wss://yourdomain.com/ws
   ```

2. Configure SSL/TLS (recommended: use a reverse proxy like Caddy or Traefik)

3. Update nginx configuration if needed

### Using Reverse Proxy

If using nginx/Caddy/Traefik in front of containers:

1. Uncomment WebSocket proxy section in `nginx.conf`
2. Configure your reverse proxy to handle WebSocket upgrades
3. Update internal network configuration

## 📚 Additional Resources

- [Docker Documentation](https://docs.docker.com/)
- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [FastAPI Deployment](https://fastapi.tiangolo.com/deployment/)
- [Vite Docker Guide](https://vitejs.dev/guide/build.html)

## 🤝 Contributing

When adding new dependencies:

1. **Backend**: Update `requirements.txt` and rebuild:
   ```bash
   docker-compose build backend
   ```

2. **Frontend**: Update `package.json` and rebuild:
   ```bash
   docker-compose build frontend
   ```

## 📄 License

See the LICENSE file in the root directory.
