# Makefile for Remem Memory Agent
.PHONY: help install install-dev clean test lint format type-check build run run-dev docker-build docker-run docker-dev docker-stop docker-clean setup-mcp

# Default target
help: ## Show this help message
	@echo "Remem Memory Agent - Available commands:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# Python environment setup
install: ## Install production dependencies
	pip install -r requirements.txt

install-dev: ## Install development dependencies
	pip install -r requirements.txt
	pip install -e ".[dev]"

# Development commands
clean: ## Clean up cache files and build artifacts
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +

test: ## Run tests
	python -m pytest tests/ -v

lint: ## Run linting checks
	flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
	flake8 . --count --exit-zero --max-complexity=10 --max-line-length=100 --statistics

format: ## Format code with black and isort
	black .
	isort .

type-check: ## Run type checking with mypy
	mypy memory/ api/ clients/ --ignore-missing-imports

check: lint type-check ## Run all code quality checks

# Application commands
run: ## Run the web API server
	python web_app.py

run-cli: ## Run the CLI interface
	python main.py

run-mcp: ## Run the MCP server
	python mcp_server.py

run-dev: ## Run in development mode with auto-reload
	uvicorn web_app:app --host 0.0.0.0 --port 5001 --reload

# Docker commands
docker-build: ## Build Docker image
	docker build -t remem:latest .

docker-build-dev: ## Build Docker image for development
	docker build --target development -t remem:dev .

docker-run: ## Run with Docker Compose (production)
	docker-compose up -d

docker-dev: ## Run with Docker Compose (development)
	docker-compose --profile dev up -d

docker-stop: ## Stop Docker containers
	docker-compose down

docker-clean: ## Stop containers and remove volumes
	docker-compose down -v
	docker system prune -f

docker-logs: ## Show Docker logs
	docker-compose logs -f

# Redis commands
redis-start: ## Start Redis with Docker
	docker run -d --name remem-redis -p 6379:6379 redis:8

redis-stop: ## Stop Redis container
	docker stop remem-redis && docker rm remem-redis

redis-cli: ## Connect to Redis CLI
	docker exec -it remem-redis redis-cli

# Setup commands
setup: ## Initial project setup
	cp .env.example .env
	@echo "Please edit .env file with your configuration"

setup-mcp: ## Setup MCP server for Claude Desktop
	python scripts/setup_mcp_server.py

# Build and package
build: clean ## Build the package
	python -m build

# Development workflow
dev-setup: install-dev setup ## Complete development setup
	@echo "Development environment ready!"
	@echo "1. Edit .env file with your API keys"
	@echo "2. Start Redis: make redis-start"
	@echo "3. Run the app: make run-dev"

# Production deployment
deploy: docker-build docker-run ## Build and deploy with Docker

# Testing and CI
ci: clean install-dev check test ## Run CI pipeline locally

# Documentation
docs: ## Generate documentation (placeholder)
	@echo "Documentation generation not implemented yet"

# Database/Redis management
db-reset: ## Reset Redis database (WARNING: deletes all data)
	@echo "This will delete ALL data in Redis. Are you sure? [y/N]" && read ans && [ $${ans:-N} = y ]
	docker exec remem-redis redis-cli FLUSHALL

# Utility commands
logs: ## Show application logs
	tail -f logs/*.log 2>/dev/null || echo "No log files found"

ps: ## Show running processes
	docker-compose ps

status: ## Show service status
	@echo "=== Docker Services ==="
	docker-compose ps
	@echo ""
	@echo "=== Redis Status ==="
	docker exec remem-redis redis-cli ping 2>/dev/null || echo "Redis not running"
	@echo ""
	@echo "=== API Health ==="
	curl -s http://localhost:5001/api/health 2>/dev/null | python -m json.tool || echo "API not responding"
