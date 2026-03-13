import unittest
import json
import os
from datetime import datetime

RESULTS_DIR = os.getenv("RESULTS_DIR", "results")
RESULTS_PATH = os.path.join(RESULTS_DIR, "results.json")

def main():
    # Explicitly load tests from tests/test_grader (works for .py or .so)
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromName("tests.test_grader")

    # Ensure results directory exists
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Run the test suite
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Collect structured results (Gradescope-style JSON)
    results_json = {
        "date": datetime.now().isoformat(),
        "num_tests": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "skipped": len(result.skipped),
        "successful": result.wasSuccessful(),
        "details": [
            {
                "name": str(test),
                "status": "failed",
                "message": msg
            }
            for test, msg in result.failures + result.errors
        ],
    }

    # Write results to JSON file
    with open(RESULTS_PATH, "w") as f:
        json.dump(results_json, f, indent=2)

    print(f"\nResults written to {RESULTS_PATH}")

if __name__ == "__main__":
    main()