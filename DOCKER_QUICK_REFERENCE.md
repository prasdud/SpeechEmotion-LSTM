# 🚀 Quick Reference - Docker Commands

## Start Application

### Interactive Menu (Easiest)
```bash
./docker-start.sh
```

### Production Mode
```bash
# Start
docker-compose up -d

# Start with rebuild
docker-compose up --build -d

# Access
Frontend: http://localhost:3000
Backend:  http://localhost:8000
```

### Development Mode (Hot Reload)
```bash
# Start
docker-compose -f docker-compose.dev.yml up -d

# Start with rebuild
docker-compose -f docker-compose.dev.yml up --build -d

# Access
Frontend: http://localhost:5173
Backend:  http://localhost:8000
```

## Stop Application

```bash
# Production
docker-compose down

# Development
docker-compose -f docker-compose.dev.yml down

# Stop and remove volumes
docker-compose down -v
```

## View Logs

```bash
# All services
docker-compose logs -f

# Backend only
docker-compose logs -f backend

# Frontend only
docker-compose logs -f frontend

# Last 100 lines
docker-compose logs --tail=100
```

## Rebuild After Changes

```bash
# Rebuild backend only
docker-compose build backend

# Rebuild frontend only
docker-compose build frontend

# Rebuild everything
docker-compose build

# Rebuild without cache
docker-compose build --no-cache
```

## Container Management

```bash
# List running containers
docker-compose ps

# Execute command in backend
docker exec -it speechemotion-backend bash

# Execute command in frontend
docker exec -it speechemotion-frontend sh

# Restart specific service
docker-compose restart backend
docker-compose restart frontend
```

## Troubleshooting

```bash
# Check container status
docker-compose ps

# View resource usage
docker stats

# Clean up everything
docker-compose down -v
docker system prune -a

# Check networks
docker network ls
docker network inspect speechemotion-lstm_speechemotion-network
```

## Development Tips

```bash
# Watch logs while coding
docker-compose -f docker-compose.dev.yml logs -f

# Restart after dependency changes
docker-compose -f docker-compose.dev.yml up --build

# Check backend health
curl http://localhost:8000

# Test WebSocket
wscat -c ws://localhost:8000/ws
```

## Port Mapping

| Service | Container Port | Host Port | Mode |
|---------|---------------|-----------|------|
| Backend | 8000 | 8000 | Both |
| Frontend (prod) | 80 | 3000 | Production |
| Frontend (dev) | 5173 | 5173 | Development |

## Environment Variables

Edit `.env` file:
```bash
VITE_WS_URL=ws://localhost:8000/ws  # WebSocket URL
BACKEND_PORT=8000                    # Backend port
FRONTEND_PORT=3000                   # Frontend production port
FRONTEND_DEV_PORT=5173               # Frontend dev port
```

## Common Issues & Fixes

**Port already in use:**
```bash
# Find process using port
lsof -i :8000
# Kill process
kill -9 <PID>
```

**WebSocket not connecting:**
```bash
# Check backend logs
docker-compose logs backend
# Verify WebSocket URL in browser console
```

**Hot reload not working:**
```bash
# Ensure using dev compose file
docker-compose -f docker-compose.dev.yml up
# Check volume mounts
docker inspect speechemotion-backend-dev
```

**Build cache issues:**
```bash
# Clear everything and rebuild
docker-compose down -v
docker system prune -a
docker-compose build --no-cache
docker-compose up
```
