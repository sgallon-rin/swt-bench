import logging
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Optional

import docker
from docker.models.containers import Container

_SCRIPT_DIR = Path(__file__).resolve().parent          # inference/
_PROJECT_ROOT = _SCRIPT_DIR.parent                      # swt-bench/
_IMAGE_BUILD_LOG_DIR = _PROJECT_ROOT / "image_build_logs" / "inference"


def get_arch() -> str:
    if platform.machine() in {"aarch64", "arm64"}:
        return "arm64"
    return "x86_64"


def get_platform(arch: str) -> str:
    if arch == "x86_64":
        return "linux/amd64"
    elif arch == "arm64":
        return "linux/arm64/v8"
    else:
        raise ValueError(f"Unknown architecture: {arch}")


def get_base_image_name(arch: Optional[str] = None) -> str:
    if arch is None:
        arch = get_arch()
    return f"swt-inf.base.{arch}:latest"


def get_env_image_name(arch: str, repo: str, env_hash: str) -> str:
    return f"swt-inf.env.{arch}.{repo.replace('/', '__')}.{env_hash}:latest"


def ensure_base_image(
    arch: Optional[str] = None,
    force_rebuild: bool = False,
) -> str:
    if arch is None:
        arch = get_arch()
    image_name = get_base_image_name(arch)
    client = docker.from_env()

    if not force_rebuild:
        try:
            client.images.get(image_name)
            return image_name
        except docker.errors.ImageNotFound:
            pass

    dockerfile_basename = "Dockerfile.base"
    dockerfile_path = _SCRIPT_DIR / dockerfile_basename
    if not dockerfile_path.exists():
        raise FileNotFoundError(
            f"Inference base Dockerfile not found: {dockerfile_path}"
        )

    build_dir = _IMAGE_BUILD_LOG_DIR / image_name.replace(":", "__")
    build_dir.mkdir(parents=True, exist_ok=True)

    log_file = build_dir / "build_image.log"
    logger = logging.getLogger(f"inf-build-{image_name}")
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(str(log_file), mode="w")
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(fh)

    logger.info(f"Building inference base image: {image_name}")
    logger.info(f"Architecture: {arch} -> platform: {get_platform(arch)}")

    platform_str = get_platform(arch)

    try:
        logger.info("Starting docker build...")
        proc = subprocess.Popen(
            [
                "docker", "buildx", "build",
                "--platform", platform_str,
                "--tag", image_name,
                "--file", str(dockerfile_path),
                "--load",
                str(_SCRIPT_DIR),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        for line in proc.stdout:
            logger.info(line.rstrip())
        proc.wait()
        if proc.returncode != 0:
            raise RuntimeError(
                f"Docker build failed with return code {proc.returncode}"
            )
        logger.info("Inference base image built successfully.")
    except Exception:
        logger.exception(f"Failed to build inference base image {image_name}")
        raise
    finally:
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)

    return image_name


def run_in_container(
    image_name: str,
    workspace_path: Path,
    command: list[str],
    timeout: Optional[int] = None,
    working_dir: str = "/workspace",
) -> subprocess.CompletedProcess:
    workspace_abs = workspace_path.resolve()

    client = docker.from_env()
    container = client.containers.run(
        image_name,
        command,
        volumes={str(workspace_abs): {"bind": "/workspace", "mode": "rw"}},
        working_dir=working_dir,
        detach=True,
        tty=False,
        stdin_open=False,
        remove=False,
    )

    try:
        exit_code = container.wait(timeout=timeout)["StatusCode"]
        logs = container.logs(stdout=True, stderr=True).decode("utf-8", errors="replace")

        result = subprocess.CompletedProcess(
            args=command,
            returncode=exit_code,
            stdout=logs,
            stderr="",
        )
        return result
    except Exception:
        try:
            container.kill()
        except Exception:
            pass
        raise
    finally:
        try:
            container.remove(force=True)
        except Exception:
            pass
