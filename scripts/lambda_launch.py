#!/usr/bin/env python3
"""Lambda Cloud API: poll available GPU instances and launch one.

Usage:
    # Set your API key (get from https://cloud.lambdalabs.com/api-keys)
    export LAMBDA_API_KEY="your-key-here"

    # List available instance types
    python scripts/lambda_launch.py --list

    # Launch the most available GPU with Ubuntu 22.04
    python scripts/lambda_launch.py --launch

    # Launch a specific instance type
    python scripts/lambda_launch.py --launch --type gpu_1x_a100_sxm4

    # Specify SSH key name (must already exist in Lambda dashboard)
    python scripts/lambda_launch.py --launch --ssh-key "my-key"
"""

import argparse
import json
import os
import sys
from urllib.request import Request, urlopen
from urllib.error import HTTPError

API_BASE = "https://cloud.lambdalabs.com/api/v1"


def api_call(endpoint: str, method: str = "GET", data: dict | None = None) -> dict:
    """Make an authenticated Lambda API call."""
    api_key = os.environ.get("LAMBDA_API_KEY")
    if not api_key:
        print("ERROR: Set LAMBDA_API_KEY environment variable")
        print("Get your key at: https://cloud.lambdalabs.com/api-keys")
        sys.exit(1)

    url = f"{API_BASE}/{endpoint}"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    body = json.dumps(data).encode() if data else None
    req = Request(url, data=body, headers=headers, method=method)

    try:
        with urlopen(req) as resp:
            return json.loads(resp.read().decode())
    except HTTPError as e:
        error_body = e.read().decode()
        print(f"API Error {e.code}: {error_body}")
        sys.exit(1)


def list_instances() -> list[dict]:
    """List currently running instances."""
    result = api_call("instances")
    return result.get("data", [])


def list_available_types() -> list[dict]:
    """Get instance types with availability, sorted by GPU count desc."""
    result = api_call("instance-types")
    types_data = result.get("data", {})

    available = []
    for type_name, info in types_data.items():
        specs = info.get("instance_type", {})
        regions = info.get("regions_with_capacity_available", [])
        available.append({
            "name": type_name,
            "description": specs.get("description", ""),
            "gpu_count": specs.get("specs", {}).get("gpus", 0),
            "gpu_description": specs.get("specs", {}).get("gpu_description", ""),
            "vcpus": specs.get("specs", {}).get("vcpus", 0),
            "ram_gb": specs.get("specs", {}).get("memory_gib", 0),
            "storage_gb": specs.get("specs", {}).get("storage_gib", 0),
            "price_cents_hr": specs.get("price_cents_per_hour", "?"),
            "regions": [r.get("name", r.get("description", "?")) for r in regions],
            "available": len(regions) > 0,
        })

    # Sort: available first, then by GPU count descending
    available.sort(key=lambda x: (-int(x["available"]), -x["gpu_count"]))
    return available


def list_ssh_keys() -> list[dict]:
    """List SSH keys registered in Lambda."""
    result = api_call("ssh-keys")
    return result.get("data", [])


def launch_instance(
    instance_type: str,
    region: str,
    ssh_key_names: list[str],
    name: str = "striatica-gpu",
) -> dict:
    """Launch an instance."""
    payload = {
        "region_name": region,
        "instance_type_name": instance_type,
        "ssh_key_names": ssh_key_names,
        "name": name,
    }
    result = api_call("instance-operations/launch", method="POST", data=payload)
    return result


def print_current_instances():
    """Print currently running instances."""
    instances = list_instances()
    if not instances:
        print("\n  No running instances.\n")
        return

    print(f"\n  Running instances ({len(instances)}):")
    print(f"  {'ID':<40} {'Type':<25} {'Region':<15} {'IP':<16} {'Status'}")
    print(f"  {'-'*40} {'-'*25} {'-'*15} {'-'*16} {'-'*10}")
    for inst in instances:
        print(
            f"  {inst.get('id', '?'):<40} "
            f"{inst.get('instance_type', {}).get('name', '?'):<25} "
            f"{inst.get('region', {}).get('name', '?'):<15} "
            f"{inst.get('ip', '?'):<16} "
            f"{inst.get('status', '?')}"
        )
    print()


def print_available_types(types: list[dict]):
    """Print available instance types."""
    print(f"\n  Instance Types ({len(types)} total):")
    print(f"  {'Type':<30} {'GPUs':<6} {'GPU':<22} {'RAM':<8} {'$/hr':<8} {'Available'}")
    print(f"  {'-'*30} {'-'*6} {'-'*22} {'-'*8} {'-'*8} {'-'*20}")
    for t in types:
        price = f"${t['price_cents_hr'] / 100:.2f}" if isinstance(t["price_cents_hr"], (int, float)) else "?"
        regions_str = ", ".join(t["regions"][:3]) if t["regions"] else "NONE"
        avail_marker = f"YES ({regions_str})" if t["available"] else "no"
        print(
            f"  {t['name']:<30} "
            f"{t['gpu_count']:<6} "
            f"{t['gpu_description']:<22} "
            f"{t['ram_gb']:<8} "
            f"{price:<8} "
            f"{avail_marker}"
        )
    print()


def main():
    parser = argparse.ArgumentParser(description="Lambda Cloud instance launcher for striatica")
    parser.add_argument("--list", action="store_true", help="List available types and running instances")
    parser.add_argument("--launch", action="store_true", help="Launch most available instance")
    parser.add_argument("--type", type=str, default=None, help="Specific instance type to launch")
    parser.add_argument("--ssh-key", type=str, default=None, help="SSH key name (from Lambda dashboard)")
    parser.add_argument("--name", type=str, default="striatica-gpu", help="Instance name")
    parser.add_argument("--region", type=str, default=None, help="Specific region (auto-selects if omitted)")
    args = parser.parse_args()

    if not args.list and not args.launch:
        args.list = True  # Default to listing

    if args.list:
        print("\n=== Current Instances ===")
        print_current_instances()

        print("=== Available Instance Types ===")
        types = list_available_types()
        print_available_types(types)

    if args.launch:
        # Get SSH keys
        ssh_keys = list_ssh_keys()
        if not ssh_keys:
            print("ERROR: No SSH keys registered in Lambda. Add one at https://cloud.lambdalabs.com/ssh-keys")
            sys.exit(1)

        if args.ssh_key:
            key_names = [args.ssh_key]
        else:
            key_names = [ssh_keys[0]["name"]]
            print(f"  Using SSH key: {key_names[0]} (first registered key)")

        # Pick instance type
        types = list_available_types()
        available_types = [t for t in types if t["available"]]

        if not available_types:
            print("ERROR: No instance types currently available. Try again later.")
            print("  Tip: Lambda capacity changes frequently. A100s often open up at off-peak hours (late night US time).")
            sys.exit(1)

        if args.type:
            chosen = next((t for t in available_types if t["name"] == args.type), None)
            if not chosen:
                print(f"ERROR: '{args.type}' is not available right now.")
                print("  Available types:", ", ".join(t["name"] for t in available_types))
                sys.exit(1)
        else:
            # Pick the most powerful available (highest GPU count, then most RAM)
            chosen = max(available_types, key=lambda t: (t["gpu_count"], t["ram_gb"]))
            print(f"  Auto-selected: {chosen['name']} ({chosen['gpu_count']}x {chosen['gpu_description']}, {chosen['ram_gb']}GB RAM)")

        # Pick region
        if args.region:
            region = args.region
        else:
            region = chosen["regions"][0]  # First available region
            print(f"  Region: {region}")

        # Confirm
        price = f"${chosen['price_cents_hr'] / 100:.2f}/hr" if isinstance(chosen["price_cents_hr"], (int, float)) else "?"
        print(f"\n  Launching: {chosen['name']}")
        print(f"  Price: {price}")
        print(f"  Region: {region}")
        print(f"  SSH keys: {key_names}")
        print(f"  Name: {args.name}")
        print()

        result = launch_instance(
            instance_type=chosen["name"],
            region=region,
            ssh_key_names=key_names,
            name=args.name,
        )

        instance_ids = result.get("data", {}).get("instance_ids", [])
        if instance_ids:
            print(f"  LAUNCHED! Instance ID: {instance_ids[0]}")
            print(f"  Check status: python scripts/lambda_launch.py --list")
            print(f"  Or visit: https://cloud.lambdalabs.com/instances")
        else:
            print(f"  Launch response: {json.dumps(result, indent=2)}")


if __name__ == "__main__":
    main()
