# Redis Setup for Memory Agent

The Memory Agent requires Redis with the RedisSearch module for vector similarity search.

## Option 1: Redis Stack (Recommended)

Redis Stack includes RedisSearch and other modules out of the box.

### Using Docker (Easiest)

```bash
# Run Redis Stack with Docker
docker run -d --name redis-stack -p 6379:6379 -p 8001:8001 redis/redis-stack:latest

# Check if it's running
docker ps
```

### Using Homebrew (macOS)

```bash
# Install Redis Stack
brew tap redis-stack/redis-stack
brew install redis-stack

# Start Redis Stack
redis-stack-server

# Or run in background
brew services start redis-stack
```

### Using Package Manager (Linux)

```bash
# Ubuntu/Debian
curl -fsSL https://packages.redis.io/gpg | sudo gpg --dearmor -o /usr/share/keyrings/redis-archive-keyring.gpg
echo "deb [signed-by=/usr/share/keyrings/redis-archive-keyring.gpg] https://packages.redis.io/deb $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/redis.list
sudo apt-get update
sudo apt-get install redis-stack-server

# Start the service
sudo systemctl start redis-stack-server
```

## Option 2: Regular Redis + RedisSearch Module

If you already have Redis installed, you can add the RedisSearch module:

```bash
# Download and compile RedisSearch
git clone https://github.com/RediSearch/RediSearch.git
cd RediSearch
make setup
make build

# Start Redis with the module
redis-server --loadmodule ./bin/linux-x64-release/search/redisearch.so
```

## Testing the Setup

Once Redis is running, test the connection:

```bash
# Test basic Redis connection
redis-cli ping

# Test RedisSearch module
redis-cli MODULE LIST
```

You should see `search` in the module list.

## Configuration

The Memory Agent connects to Redis using these default settings:
- Host: `localhost`
- Port: `6379`
- Database: `0`

You can modify these in the `MemoryAgent` constructor if needed.

## Troubleshooting

### Connection Issues
- Make sure Redis is running: `redis-cli ping`
- Check if the port is accessible: `telnet localhost 6379`
- Verify RedisSearch is loaded: `redis-cli MODULE LIST`

### Permission Issues
- On macOS/Linux, you might need to use `sudo` for system-wide installation
- For Docker, make sure Docker daemon is running

### Memory Issues
- Redis Stack requires more memory than regular Redis
- Adjust Docker memory limits if using containers
- Monitor memory usage with `redis-cli INFO memory`
