#!/bin/bash

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}🎭 SpeechEmotion-LSTM Docker Setup${NC}"
echo ""

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ Docker is not installed. Please install Docker first.${NC}"
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo -e "${RED}❌ Docker Compose is not installed. Please install Docker Compose first.${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Docker and Docker Compose are installed${NC}"
echo ""

# Menu
echo "Select mode:"
echo "1) Production mode (nginx + optimized build)"
echo "2) Development mode (hot reload)"
echo "3) Stop all containers"
echo "4) View logs"
echo "5) Clean up (remove containers and volumes)"
echo ""
read -p "Enter choice [1-5]: " choice

case $choice in
    1)
        echo -e "${YELLOW}🚀 Starting in PRODUCTION mode...${NC}"
        docker-compose up --build
        ;;
    2)
        echo -e "${YELLOW}🔧 Starting in DEVELOPMENT mode...${NC}"
        docker-compose -f docker-compose.dev.yml up --build
        ;;
    3)
        echo -e "${YELLOW}🛑 Stopping all containers...${NC}"
        docker-compose down
        docker-compose -f docker-compose.dev.yml down
        echo -e "${GREEN}✅ Containers stopped${NC}"
        ;;
    4)
        echo "Select service to view logs:"
        echo "1) Backend"
        echo "2) Frontend"
        echo "3) All services"
        read -p "Enter choice [1-3]: " log_choice
        
        case $log_choice in
            1)
                docker-compose logs -f backend
                ;;
            2)
                docker-compose logs -f frontend
                ;;
            3)
                docker-compose logs -f
                ;;
            *)
                echo -e "${RED}Invalid choice${NC}"
                ;;
        esac
        ;;
    5)
        echo -e "${YELLOW}🧹 Cleaning up containers and volumes...${NC}"
        docker-compose down -v
        docker-compose -f docker-compose.dev.yml down -v
        read -p "Remove Docker images too? (y/n): " remove_images
        if [ "$remove_images" = "y" ]; then
            docker rmi speechemotion-lstm-backend speechemotion-lstm-frontend 2>/dev/null
            echo -e "${GREEN}✅ Images removed${NC}"
        fi
        echo -e "${GREEN}✅ Cleanup complete${NC}"
        ;;
    *)
        echo -e "${RED}Invalid choice${NC}"
        exit 1
        ;;
esac
