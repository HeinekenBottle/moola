#!/usr/bin/env python3
"""Test script for Moola ML API deployment.

Validates that the production API is working correctly by:
1. Health check
2. Single prediction test
3. Batch prediction test
4. Latency benchmarking
5. Prometheus metrics validation

Usage:
    # Test local deployment
    python scripts/test_deployment.py

    # Test remote deployment
    python scripts/test_deployment.py --host http://api.example.com:8000
"""

import argparse
import time
from typing import List

import numpy as np
import requests
from loguru import logger


def generate_random_window() -> List[List[float]]:
    """Generate random OHLC window for testing.

    Returns:
        List of 105 OHLC bars
    """
    window = []
    base_price = 4500.0

    for i in range(105):
        # Simulate realistic OHLC with constraints: H >= max(O,C), L <= min(O,C)
        open_price = base_price + np.random.randn() * 10
        close_price = base_price + np.random.randn() * 10
        high_price = max(open_price, close_price) + abs(np.random.randn()) * 5
        low_price = min(open_price, close_price) - abs(np.random.randn()) * 5

        window.append([
            round(open_price, 2),
            round(high_price, 2),
            round(low_price, 2),
            round(close_price, 2)
        ])

        # Update base price for next bar
        base_price = close_price

    return window


def test_health(base_url: str) -> bool:
    """Test health check endpoint.

    Args:
        base_url: Base API URL

    Returns:
        True if healthy, False otherwise
    """
    logger.info("Testing health check...")

    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        response.raise_for_status()

        data = response.json()
        logger.success(f"✓ Health check passed: {data}")

        if not data.get("model_loaded", False):
            logger.warning("⚠ Model not loaded - API degraded")
            return False

        return True

    except Exception as e:
        logger.error(f"✗ Health check failed: {e}")
        return False


def test_single_prediction(base_url: str) -> bool:
    """Test single prediction endpoint.

    Args:
        base_url: Base API URL

    Returns:
        True if prediction succeeds, False otherwise
    """
    logger.info("Testing single prediction...")

    try:
        # Generate test window
        window = generate_random_window()

        # Send prediction request
        start_time = time.time()
        response = requests.post(
            f"{base_url}/predict",
            json={"features": window},
            timeout=10
        )
        latency = time.time() - start_time

        response.raise_for_status()
        data = response.json()

        logger.success(f"✓ Prediction successful:")
        logger.info(f"  Pattern: {data['prediction']}")
        logger.info(f"  Confidence: {data['confidence']:.3f}")
        logger.info(f"  Latency: {latency*1000:.1f}ms")

        # Validate response format
        assert "prediction" in data
        assert "confidence" in data
        assert "probabilities" in data
        assert 0 <= data["confidence"] <= 1

        return True

    except Exception as e:
        logger.error(f"✗ Single prediction failed: {e}")
        return False


def test_batch_prediction(base_url: str, batch_size: int = 10) -> bool:
    """Test batch prediction endpoint.

    Args:
        base_url: Base API URL
        batch_size: Number of windows to predict

    Returns:
        True if batch prediction succeeds, False otherwise
    """
    logger.info(f"Testing batch prediction (size={batch_size})...")

    try:
        # Generate batch of windows
        windows = [generate_random_window() for _ in range(batch_size)]

        # Send batch prediction request
        start_time = time.time()
        response = requests.post(
            f"{base_url}/predict/batch",
            json={"windows": windows},
            timeout=30
        )
        latency = time.time() - start_time

        response.raise_for_status()
        data = response.json()

        logger.success(f"✓ Batch prediction successful:")
        logger.info(f"  Batch size: {data['batch_size']}")
        logger.info(f"  Total latency: {latency*1000:.1f}ms")
        logger.info(f"  Avg latency/prediction: {latency*1000/batch_size:.1f}ms")

        # Validate response
        assert data["batch_size"] == batch_size
        assert len(data["predictions"]) == batch_size

        return True

    except Exception as e:
        logger.error(f"✗ Batch prediction failed: {e}")
        return False


def benchmark_latency(base_url: str, n_requests: int = 100) -> bool:
    """Benchmark prediction latency.

    Args:
        base_url: Base API URL
        n_requests: Number of requests to send

    Returns:
        True if benchmark succeeds, False otherwise
    """
    logger.info(f"Benchmarking latency ({n_requests} requests)...")

    try:
        latencies = []

        for i in range(n_requests):
            window = generate_random_window()

            start_time = time.time()
            response = requests.post(
                f"{base_url}/predict",
                json={"features": window},
                timeout=10
            )
            latency = time.time() - start_time

            response.raise_for_status()
            latencies.append(latency)

            if (i + 1) % 25 == 0:
                logger.debug(f"  Progress: {i+1}/{n_requests}")

        # Calculate statistics
        latencies_ms = [l * 1000 for l in latencies]
        p50 = np.percentile(latencies_ms, 50)
        p95 = np.percentile(latencies_ms, 95)
        p99 = np.percentile(latencies_ms, 99)
        avg = np.mean(latencies_ms)

        logger.success(f"✓ Latency benchmark complete:")
        logger.info(f"  Average: {avg:.1f}ms")
        logger.info(f"  p50: {p50:.1f}ms")
        logger.info(f"  p95: {p95:.1f}ms")
        logger.info(f"  p99: {p99:.1f}ms")

        # Check against targets
        if p95 > 100:
            logger.warning(f"⚠ p95 latency ({p95:.1f}ms) exceeds target (100ms)")

        return True

    except Exception as e:
        logger.error(f"✗ Latency benchmark failed: {e}")
        return False


def test_metrics(base_url: str) -> bool:
    """Test Prometheus metrics endpoint.

    Args:
        base_url: Base API URL

    Returns:
        True if metrics endpoint accessible, False otherwise
    """
    logger.info("Testing Prometheus metrics...")

    try:
        response = requests.get(f"{base_url}/metrics", timeout=5)
        response.raise_for_status()

        metrics_text = response.text

        # Check for expected metrics
        expected_metrics = [
            "predictions_total",
            "prediction_latency_seconds",
            "prediction_confidence",
            "prediction_errors_total"
        ]

        found_metrics = []
        for metric in expected_metrics:
            if metric in metrics_text:
                found_metrics.append(metric)

        logger.success(f"✓ Metrics endpoint accessible:")
        logger.info(f"  Found metrics: {len(found_metrics)}/{len(expected_metrics)}")

        for metric in found_metrics:
            logger.debug(f"    - {metric}")

        return len(found_metrics) == len(expected_metrics)

    except Exception as e:
        logger.error(f"✗ Metrics test failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Test Moola ML API deployment")
    parser.add_argument("--host", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--skip-benchmark", action="store_true", help="Skip latency benchmark")
    parser.add_argument("--benchmark-requests", type=int, default=100, help="Number of benchmark requests")

    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("MOOLA ML API DEPLOYMENT TEST")
    logger.info("=" * 80)
    logger.info(f"Target: {args.host}\n")

    # Run tests
    results = {}

    results["health"] = test_health(args.host)
    results["single_prediction"] = test_single_prediction(args.host)
    results["batch_prediction"] = test_batch_prediction(args.host)
    results["metrics"] = test_metrics(args.host)

    if not args.skip_benchmark:
        results["latency_benchmark"] = benchmark_latency(args.host, args.benchmark_requests)

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("TEST SUMMARY")
    logger.info("=" * 80)

    passed = sum(results.values())
    total = len(results)

    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        logger.info(f"{status:8} | {test_name}")

    logger.info("=" * 80)
    logger.info(f"Result: {passed}/{total} tests passed")

    if passed == total:
        logger.success("🎉 All tests passed! Deployment is healthy.")
        return 0
    else:
        logger.error(f"❌ {total - passed} test(s) failed. Check deployment.")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
