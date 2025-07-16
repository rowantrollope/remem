#!/usr/bin/env python3

import sys
import traceback

# Add the current directory to the path
sys.path.insert(0, '.')

def test_performance_optimizer():
    try:
        print("Importing performance optimizer...")
        from optimizations.performance_optimizer import get_performance_optimizer, init_performance_optimizer
        
        print("Initializing performance optimizer...")
        optimizer = init_performance_optimizer(
            redis_host="localhost",
            redis_port=6379,
            redis_db=0,
            cache_enabled=True,
            use_semantic_cache=True
        )
        
        print(f"Optimizer created: {optimizer}")
        print(f"Cache enabled: {optimizer.cache_enabled}")
        print(f"Use semantic cache: {optimizer.use_semantic_cache}")
        
        print("Getting performance metrics...")
        metrics = optimizer.get_performance_metrics()
        print(f"Metrics: {metrics}")
        
        print("✅ Performance optimizer test successful!")
        return True
        
    except Exception as e:
        print(f"❌ Error in performance optimizer test: {e}")
        print(f"Exception type: {type(e)}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_performance_optimizer() 