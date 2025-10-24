# -*- coding: utf-8 -*-
"""
MediaPipe Cache Control Example
Demonstrates how to control MediaPipe model cache location
"""

import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from video_filter import VideoQualityFilter


def setup_mediapipe_cache(cache_dir: str = "./mediapipe_cache"):
    """
    Set MediaPipe cache directory
    
    Args:
        cache_dir: Cache directory path
    """
    # Create cache directory
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    
    # Set environment variable
    os.environ['MEDIAPIPE_CACHE_DIR'] = str(cache_path.absolute())
    
    print(f"MediaPipe cache directory set to: {cache_path.absolute()}")
    return str(cache_path.absolute())


def main():
    """Main function"""
    print("=== MediaPipe Cache Control Example ===\n")
    
    # Method 1: Set via environment variable
    print("1. Set cache directory via environment variable")
    cache_dir = setup_mediapipe_cache("./custom_mediapipe_cache")
    
    # Method 2: Set via config file
    print("\n2. Set cache directory via config file")
    config_path = "config_with_cache.yaml"
    
    if os.path.exists(config_path):
        print(f"Using config file: {config_path}")
        filter = VideoQualityFilter(config_path)
    else:
        print("Config file not found, using default config")
        filter = VideoQualityFilter()
    
    # Verify cache directory
    print(f"\nCurrent MediaPipe cache directory: {os.environ.get('MEDIAPIPE_CACHE_DIR', 'Not set')}")
    
    # Check cache directory contents
    cache_path = Path(os.environ.get('MEDIAPIPE_CACHE_DIR', './mediapipe_cache'))
    if cache_path.exists():
        print(f"Cache directory contents:")
        for item in cache_path.iterdir():
            print(f"  - {item.name} ({'Directory' if item.is_dir() else 'File'})")
    else:
        print("Cache directory does not exist")
    
    print("\n=== Cache Control Complete ===")


if __name__ == "__main__":
    main()
