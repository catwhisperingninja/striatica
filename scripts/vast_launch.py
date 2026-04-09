#!/usr/bin/env python3
"""Vast.ai API: search available GPU offers and launch an instance.

Usage:
    # Set your API key (or it reads from img/correction/v2_plans/.env)
    export VAST_API_KEY="your-key-here"

    # List available offers (>=32GB VRAM, sorted cheapest first)
    python scripts/vast_launch.py --list

    # Launch the cheapest available offer
    python scripts/vast_launch.py --launch

    # Launch a specific offer by ID
    python scripts/vast_launch.py --launch --offer 12345678

    # Show current running instances
    python scripts/vast_launch.py --instances
"""

import argparse
import json
import os
import sys
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError

API_BASE = "https://console.vast.ai/api/v0"

# Default image — Ubuntu with CUDA, no desktop overhead
DEFAULT_IMAGE = "nvidia/cuda:12.4.0-devel-ubuntu22.04"

# Minimum requirements for striatica pipeline
MIN_GPU_RAM_GB = 32
MIN_DISK_GB = 50
DEFAULT_DISK_GB = 80


def load_api_key() -> str:
    """Load VAST_API_KEY from env or project .env file."""
    key = os.environ.get("VAST_API_KEY")
    if key:
        return key

    # Try project .env files
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    env_paths = [
        project_root / "img" / "correction" / "v2_plans" / ".env",
        project_root / ".env",
        Path.home() / ".vast_api_key",
    ]

    for env_path in env_paths:
        if env_path.exists():
            content = env_path.read_text()
            for line in content.splitlines():
                line = line.strip()
                if line.startswith("VAST_API_KEY="):
                    key = line.split("=", 1)[1].strip().strip('"').strip("'")
                    if key:
                        return key

    print("ERROR: Set VAST_API_KEY environment variable")
    print("  Or add VAST_API_KEY=... to img/correction/v2_plans/.env")
    print("  Get your key at: https://cloud.vast.ai/api/")
    sys.exit(1)


def load_ssh_key_path() -> str | None:
    """Load VAST_SSH_KEY path from env or project .env file."""
    key = os.environ.get("VAST_SSH_KEY")
    if key:
        return key

    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    env_path = project_root / "img" / "correction" / "v2_plans" / ".env"

    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line.startswith("VAST_SSH_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None


def api_call(
    endpoint: str,
    method: str = "GET",
    data: dict | None = None,
) -> dict:
    """Make an authenticated Vast.ai API call."""
    api_key = load_api_key()
    url = f"{API_BASE}/{endpoint}"

    # Guard: only allow HTTPS to the Vast.ai API (CWE-939)
    if not url.startswith("https://console.vast.ai/"):
        print(f"ERROR: Refusing non-Vast.ai URL: {url}")
        sys.exit(1)

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    body = json.dumps(data).encode() if data else None
    req = Request(url, data=body, headers=headers, method=method)

    try:
        with urlopen(req) as resp:  # nosemgrep: python.lang.security.audit.dynamic-urllib-use-detected
            return json.loads(resp.read().decode())
    except HTTPError as e:
        error_body = e.read().decode()
        print(f"API Error {e.code}: {error_body}")
        sys.exit(1)


def search_offers(
    min_gpu_ram_gb: int = MIN_GPU_RAM_GB,
    gpu_name: str | None = None,
) -> list[dict]:
    """Search for available GPU offers, sorted cheapest first."""
    query: dict = {
        "verified": {"eq": True},
        "rentable": {"eq": True},
        "rented": {"eq": False},
        "gpu_ram": {"gte": min_gpu_ram_gb * 1024},  # API uses MB
        "type": "on-demand",
        "order": [["dph_total", "asc"]],
        "limit": 50,
    }

    if gpu_name:
        query["gpu_name"] = {"eq": gpu_name}

    result = api_call("bundles/", method="POST", data=query)
    return result.get("offers", [])


def list_instances() -> list[dict]:
    """List currently running instances."""
    result = api_call("instances/")
    return result.get("instances", result) if isinstance(result, dict) else result


def get_ssh_keys() -> list[dict]:
    """List SSH keys registered with Vast.ai."""
    result = api_call("ssh-keys/")
    return result.get("ssh_keys", result) if isinstance(result, dict) else result


def create_instance(
    offer_id: int,
    image: str = DEFAULT_IMAGE,
    disk_gb: float = DEFAULT_DISK_GB,
    onstart: str = "",
    label: str = "striatica-gpu",
) -> dict:
    """Accept an offer and create an instance."""
    payload: dict = {
        "image": image,
        "disk": disk_gb,
        "label": label,
        "runtype": "ssh",
        "python_utf8": True,
        "lang_utf8": True,
    }

    if onstart:
        payload["onstart"] = onstart

    return api_call(f"asks/{offer_id}/", method="PUT", data=payload)


def list_volumes() -> list[dict]:
    """List storage volumes."""
    result = api_call("volumes/")
    return result.get("volumes", result) if isinstance(result, dict) else result


def create_volume(machine_id: int, size_gb: int) -> dict:
    """Create a storage volume on a specific machine."""
    return api_call("volumes/", method="PUT", data={
        "id": machine_id,
        "size": size_gb,
    })


def copy_data(src: str, dst: str) -> dict:
    """Initiate a copy between instances/volumes.

    Format: <instance_id>:/path/ or V.<volume_id>:/path/
    API: PUT /commands/copy_direct/
    """
    # Parse src and dst — format is "ID:/path" or "V.ID:/path"
    def parse_target(target: str) -> tuple[str | None, str]:
        if ":" in target:
            id_part, path = target.split(":", 1)
            return id_part, path
        return None, target

    src_id, src_path = parse_target(src)
    dst_id, dst_path = parse_target(dst)

    payload: dict = {
        "src_path": src_path,
        "dst_path": dst_path,
    }
    if src_id:
        payload["src_id"] = src_id
    if dst_id:
        payload["dst_id"] = dst_id

    return api_call("commands/copy_direct/", method="PUT", data=payload)


def create_template(
    name: str,
    image: str = DEFAULT_IMAGE,
    disk_gb: float = DEFAULT_DISK_GB,
    onstart: str = "",
    env: str = "",
) -> dict:
    """Create a reusable instance template."""
    payload: dict = {
        "name": name,
        "image": image,
        "tag": "latest",
        "runtype": "ssh",
        "ssh_direct": True,
        "disk": disk_gb,
    }
    if onstart:
        payload["onstart"] = onstart
    if env:
        payload["env"] = env
    return api_call("template/", method="POST", data=payload)


def print_volumes(volumes: list[dict]):
    """Print storage volumes."""
    if not volumes:
        print("\n  No volumes.\n")
        return

    print(f"\n  Volumes ({len(volumes)}):")
    print(f"  {'ID':<12} {'Machine':<12} {'Size GB':<10} {'Used GB':<10} {'Status'}")
    print(f"  {'-'*12} {'-'*12} {'-'*10} {'-'*10} {'-'*12}")
    for v in volumes:
        print(
            f"  {v.get('id', '?'):<12} "
            f"{v.get('machine_id', '?'):<12} "
            f"{v.get('size', '?'):<10} "
            f"{v.get('disk_usage', '?'):<10} "
            f"{v.get('status', '?')}"
        )
    print()


def print_offers(offers: list[dict]):
    """Print available offers."""
    if not offers:
        print("\n  No offers matching criteria.\n")
        return

    print(f"\n  Available Offers ({len(offers)}):")
    print(f"  {'ID':<12} {'GPUs':<6} {'GPU':<22} {'VRAM':<8} {'RAM':<8} {'Disk':<8} {'$/hr':<8} {'DL Mbps':<10} {'Reliability'}")
    print(f"  {'-'*12} {'-'*6} {'-'*22} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*10} {'-'*11}")
    for o in offers:
        gpu_ram = o.get("gpu_ram", 0) / 1024  # MB -> GB
        print(
            f"  {o.get('id', '?'):<12} "
            f"{o.get('num_gpus', '?'):<6} "
            f"{o.get('gpu_name', '?'):<22} "
            f"{gpu_ram:<8.0f} "
            f"{o.get('cpu_ram', 0) / 1024:<8.0f} "
            f"{o.get('disk_space', 0):<8.0f} "
            f"${o.get('dph_total', 0):<7.2f} "
            f"{o.get('inet_down', 0):<10.0f} "
            f"{o.get('reliability2', 0):.2f}"
        )
    print()


def print_instances(instances: list[dict]):
    """Print running instances."""
    if not instances:
        print("\n  No running instances.\n")
        return

    print(f"\n  Running Instances ({len(instances)}):")
    print(f"  {'ID':<12} {'GPU':<22} {'Status':<12} {'$/hr':<8} {'SSH'}")
    print(f"  {'-'*12} {'-'*22} {'-'*12} {'-'*8} {'-'*40}")
    for inst in instances:
        ssh_host = inst.get("ssh_host", "")
        ssh_port = inst.get("ssh_port", "")
        ssh_str = f"ssh -p {ssh_port} root@{ssh_host}" if ssh_host else "pending..."
        gpu_name = inst.get("gpu_name", "?")
        num_gpus = inst.get("num_gpus", 1)
        print(
            f"  {inst.get('id', '?'):<12} "
            f"{f'{num_gpus}x {gpu_name}':<22} "
            f"{(inst.get('actual_status') or inst.get('status_msg') or '?'):<12} "
            f"${inst.get('dph_total', 0):<7.2f} "
            f"{ssh_str}"
        )
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Vast.ai instance launcher for striatica"
    )
    parser.add_argument(
        "--list", action="store_true", help="List available GPU offers"
    )
    parser.add_argument(
        "--instances", action="store_true", help="Show running instances"
    )
    parser.add_argument(
        "--launch", action="store_true", help="Launch cheapest available offer"
    )
    parser.add_argument(
        "--offer", type=int, default=None, help="Specific offer ID to accept"
    )
    parser.add_argument(
        "--gpu", type=str, default=None,
        help="Filter by GPU name (e.g., A100_SXM4, H100_SXM, RTX_4090)",
    )
    parser.add_argument(
        "--image", type=str, default=DEFAULT_IMAGE,
        help=f"Docker image (default: {DEFAULT_IMAGE})",
    )
    parser.add_argument(
        "--disk", type=float, default=DEFAULT_DISK_GB,
        help=f"Disk size in GB (default: {DEFAULT_DISK_GB})",
    )
    parser.add_argument(
        "--label", type=str, default="striatica-gpu",
        help="Instance label",
    )
    parser.add_argument(
        "--min-vram", type=int, default=MIN_GPU_RAM_GB,
        help=f"Minimum GPU VRAM in GB (default: {MIN_GPU_RAM_GB})",
    )
    # Volume commands
    parser.add_argument(
        "--volumes", action="store_true", help="List storage volumes"
    )
    parser.add_argument(
        "--create-volume", action="store_true",
        help="Create a volume (requires --machine-id and --size)",
    )
    parser.add_argument(
        "--machine-id", type=int, default=None, help="Machine ID for volume creation"
    )
    parser.add_argument(
        "--size", type=int, default=30, help="Volume size in GB (default: 30)"
    )
    # Copy command
    parser.add_argument(
        "--copy", nargs=2, metavar=("SRC", "DST"),
        help="Copy data: C.<id>:/path/ V.<id>:/path/",
    )
    # Template command
    parser.add_argument(
        "--create-template", type=str, default=None, metavar="NAME",
        help="Create a reusable template with given name",
    )
    parser.add_argument(
        "--onstart", type=str, default="",
        help="Commands to run on instance start",
    )
    args = parser.parse_args()

    has_action = any([
        args.list, args.launch, args.instances, args.volumes,
        args.create_volume, args.copy, args.create_template,
    ])
    if not has_action:
        args.list = True
        args.instances = True

    if args.instances:
        print("\n=== Current Instances ===")
        instances = list_instances()
        if isinstance(instances, list):
            print_instances(instances)
        else:
            print(f"  Response: {json.dumps(instances, indent=2)[:500]}")

    if args.list:
        print("=== Available Offers ===")
        print(f"  Filters: VRAM >= {args.min_vram}GB, on-demand, verified")
        if args.gpu:
            print(f"  GPU filter: {args.gpu}")
        offers = search_offers(min_gpu_ram_gb=args.min_vram, gpu_name=args.gpu)
        print_offers(offers)

    if args.volumes:
        print("\n=== Storage Volumes ===")
        volumes = list_volumes()
        if isinstance(volumes, list):
            print_volumes(volumes)
        else:
            print(f"  Response: {json.dumps(volumes, indent=2)[:500]}")

    if args.create_volume:
        if not args.machine_id:
            print("ERROR: --create-volume requires --machine-id")
            print("  Use --instances to find machine IDs")
            sys.exit(1)
        print(f"\n  Creating {args.size}GB volume on machine {args.machine_id}...")
        result = create_volume(args.machine_id, args.size)
        print(f"  Result: {json.dumps(result, indent=2)}")

    if args.copy:
        src, dst = args.copy
        print(f"\n  Copying: {src} -> {dst}")
        result = copy_data(src, dst)
        print(f"  Result: {json.dumps(result, indent=2)}")

    if args.create_template:
        print(f"\n  Creating template '{args.create_template}'...")
        result = create_template(
            name=args.create_template,
            image=args.image,
            disk_gb=args.disk,
            onstart=args.onstart,
        )
        print(f"  Result: {json.dumps(result, indent=2)}")

    if args.launch:
        offers = search_offers(min_gpu_ram_gb=args.min_vram, gpu_name=args.gpu)

        if args.offer:
            chosen = next((o for o in offers if o["id"] == args.offer), None)
            if not chosen:
                # Try launching directly even if not in search results
                print(f"  Offer {args.offer} not in search results, attempting anyway...")
                chosen = {"id": args.offer, "gpu_name": "?", "num_gpus": "?", "dph_total": 0}
        else:
            if not offers:
                print("ERROR: No offers available matching criteria.")
                print("  Try: --min-vram 24 or --gpu RTX_4090")
                sys.exit(1)
            chosen = offers[0]  # Cheapest first (already sorted)

        gpu_ram = chosen.get("gpu_ram", 0) / 1024 if chosen.get("gpu_ram") else 0
        price = chosen.get("dph_total", 0)

        print(f"\n  Launching offer {chosen['id']}:")
        print(f"    GPU: {chosen.get('num_gpus', '?')}x {chosen.get('gpu_name', '?')} ({gpu_ram:.0f}GB VRAM)")
        print(f"    Price: ${price:.2f}/hr")
        print(f"    Image: {args.image}")
        print(f"    Disk: {args.disk}GB")
        print(f"    Label: {args.label}")
        print()

        result = create_instance(
            offer_id=chosen["id"],
            image=args.image,
            disk_gb=args.disk,
            label=args.label,
        )

        if result.get("success"):
            instance_id = result.get("new_contract")
            print(f"  LAUNCHED! Instance ID: {instance_id}")
            print(f"  Check status: python scripts/vast_launch.py --instances")
            print(f"  Or visit: https://cloud.vast.ai/instances/")
            print()
            print("  Once SSH is ready, bootstrap with:")
            print("    scp scripts/providers/vast_setupv2.sh root@<host>:/root/")
            print("    ssh root@<host> 'bash /root/vast_setupv2.sh'")
        else:
            print(f"  Launch response: {json.dumps(result, indent=2)}")


if __name__ == "__main__":
    main()
