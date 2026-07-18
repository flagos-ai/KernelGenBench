#!/usr/bin/env python3

# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Verifier Client for Verifier Server

Submits test requests to the verifier server, retrieves results and saves them to files.
The client reads local kernel file content and sends the code to the server for execution.

Supports:
    - CUDA (NVIDIA GPUs)

Usage:
    # Submit test and save result
    python -m sandbox.server.verifier_client path/to/kernel.py --output-file result.json

    # Specify server address
    python -m sandbox.server.verifier_client path/to/kernel.py --server http://localhost:8888 --output-file result.json

    # Check server status
    python -m sandbox.server.verifier_client --status

    # Health check
    python -m sandbox.server.verifier_client --health
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import requests


DEFAULT_SERVER = "http://localhost:8888"


class VerifierClient:
    """Client for interacting with the Verifier Server."""

    def __init__(self, server: str = DEFAULT_SERVER):
        """Initialize the client with a server address.

        Args:
            server: The server URL (e.g., "http://localhost:8888")
        """
        self.server = server.rstrip("/")

    def submit_test(
        self,
        operator_name: str,
        kernel_code: str,
        test_module: str = "",
        test_set: str = "KernelGenBench",
        timeout: int = 300
    ) -> dict:
        """Submit a test request to the server.

        Args:
            operator_name: Name of the operator to test
            kernel_code: Source code of the kernel
            test_module: Optional test module name (takes priority over test_set)
            test_set: Test set to use: KernelGenBench (default: KernelGenBench)
            timeout: Timeout in seconds (default: 300)

        Returns:
            dict: Test result from the server
        """
        payload = {
            "operator_name": operator_name,
            "kernel_code": kernel_code,
            "test_module": test_module,
            "test_set": test_set,
            "timeout": timeout,
        }

        response = requests.post(
            f"{self.server}/test",
            json=payload,
            timeout=timeout + 120
        )
        response.raise_for_status()
        return response.json()

    def get_status(self) -> dict:
        """Get server and device status.

        Returns:
            dict: Server status including device information
        """
        response = requests.get(f"{self.server}/status", timeout=10)
        response.raise_for_status()
        return response.json()

    def health_check(self) -> dict:
        """Perform a health check on the server.

        Returns:
            dict: Health status including device type
        """
        response = requests.get(f"{self.server}/health", timeout=10)
        response.raise_for_status()
        return response.json()

    def submit_file(
        self,
        kernel_file: str,
        test_module: str = "",
        test_set: str = "KernelGenBench",
        timeout: int = 300,
        output_file: Optional[str] = None
    ) -> dict:
        """Submit a kernel file for testing.

        Args:
            kernel_file: Path to the kernel file
            test_module: Optional test module name (takes priority over test_set)
            test_set: Test set to use: KernelGenBench (default: KernelGenBench)
            timeout: Timeout in seconds (default: 300)
            output_file: Optional path to save the result

        Returns:
            dict: Test result from the server
        """
        kernel_path = Path(kernel_file)
        if not kernel_path.exists():
            raise FileNotFoundError(f"File not found: {kernel_file}")

        with open(kernel_path, 'r', encoding='utf-8') as f:
            kernel_code = f.read()

        operator_name = extract_operator_name(kernel_file)

        result = self.submit_test(
            operator_name=operator_name,
            kernel_code=kernel_code,
            test_module=test_module,
            test_set=test_set,
            timeout=timeout,
        )

        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

        return result


def extract_operator_name(filename: str) -> str:
    """Extract operator name from filename.

    Examples:
        - aten_add.py -> add
        - aten::add.py -> add
        - my_kernel.py -> my_kernel
    """
    stem = Path(filename).stem
    if stem.startswith("aten_"):
        return stem[5:]  # aten_add -> add
    elif "::" in stem:
        return stem.split("::")[-1]  # aten::add -> add
    else:
        return stem


def main():
    parser = argparse.ArgumentParser(
        description="Verifier Client for Verifier Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Submit a kernel file for testing
    python -m sandbox.server.verifier_client kernel.py

    # Save result to file
    python -m sandbox.server.verifier_client kernel.py --output-file result.json

    # Use a different server
    python -m sandbox.server.verifier_client kernel.py --server http://10.0.0.1:8888

    # Check server status
    python -m sandbox.server.verifier_client --status

    # Health check
    python -m sandbox.server.verifier_client --health
"""
    )
    parser.add_argument(
        "kernel_file",
        nargs="?",
        type=str,
        help="Path to the kernel file"
    )
    parser.add_argument(
        "--server",
        type=str,
        default=DEFAULT_SERVER,
        help=f"Server address (default: {DEFAULT_SERVER})"
    )
    parser.add_argument(
        "--output-file",
        type=str,
        help="Path to save the result"
    )
    parser.add_argument(
        "--test-module",
        type=str,
        default="",
        help="Test module name (takes priority over --test-set)"
    )
    parser.add_argument(
        "--test-set",
        type=str,
        choices=["KernelGenBench"],
        default="KernelGenBench",
        help="Test set to use (default: KernelGenBench)"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Timeout in seconds (default: 300)"
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Get server status"
    )
    parser.add_argument(
        "--health",
        action="store_true",
        help="Perform health check"
    )

    args = parser.parse_args()

    client = VerifierClient(server=args.server)

    try:
        # Health check
        if args.health:
            result = client.health_check()
            print(json.dumps(result, indent=2, ensure_ascii=False))
            return

        # Get status
        if args.status:
            result = client.get_status()
            print(json.dumps(result, indent=2, ensure_ascii=False))
            return

        # Submit test
        if not args.kernel_file:
            parser.print_help()
            sys.exit(1)

        # Check file exists
        kernel_path = Path(args.kernel_file)
        if not kernel_path.exists():
            print(f"Error: File not found: {args.kernel_file}", file=sys.stderr)
            sys.exit(1)

        # Read file content
        with open(kernel_path, 'r', encoding='utf-8') as f:
            kernel_code = f.read()

        # Extract operator name
        operator_name = extract_operator_name(args.kernel_file)

        print(f"Submitting test: {kernel_path}")
        print(f"Operator name: {operator_name}")
        print(f"Code length: {len(kernel_code)} characters")

        result = client.submit_test(
            operator_name=operator_name,
            kernel_code=kernel_code,
            test_module=args.test_module,
            test_set=args.test_set,
            timeout=args.timeout,
        )

        # Print result
        print(json.dumps(result, indent=2, ensure_ascii=False))

        # Save to file
        if args.output_file:
            with open(args.output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"\nResult saved to: {args.output_file}")

        # Exit code based on success
        sys.exit(0 if result.get("success") else 1)

    except requests.exceptions.ConnectionError:
        print(f"Error: Cannot connect to server {args.server}", file=sys.stderr)
        print("Make sure the server is running:", file=sys.stderr)
        print("  python -m sandbox.server.verifier_server", file=sys.stderr)
        sys.exit(1)
    except requests.exceptions.HTTPError as e:
        print(f"HTTP Error: {e}", file=sys.stderr)
        if e.response is not None:
            try:
                print(json.dumps(e.response.json(), indent=2, ensure_ascii=False), file=sys.stderr)
            except Exception:
                print(e.response.text, file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
