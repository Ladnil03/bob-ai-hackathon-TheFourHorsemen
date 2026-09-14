"""
Production startup script for Render deployment.
Builds the React frontend, then serves it alongside the FastAPI backend.
"""
import os
import sys
import subprocess
from pathlib import Path

# Paths
ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
FRONTEND_DIR = ROOT_DIR / "frontend"
STATIC_DIR = SRC_DIR / "static"

def build_frontend():
    """Build the React frontend for production."""
    if not (FRONTEND_DIR / "package.json").exists():
        print("⚠️  No frontend/package.json found, skipping frontend build.")
        return

    print("🔨 Building React frontend...")
    subprocess.run(
        ["npm", "ci", "--prefer-offline"],
        cwd=str(FRONTEND_DIR),
        check=True,
    )
    subprocess.run(
        ["npm", "run", "build"],
        cwd=str(FRONTEND_DIR),
        check=True,
    )

    # Move dist to src/static for FastAPI to serve
    dist_dir = FRONTEND_DIR / "dist"
    if dist_dir.exists():
        import shutil
        if STATIC_DIR.exists():
            shutil.rmtree(STATIC_DIR)
        shutil.copytree(str(dist_dir), str(STATIC_DIR))
        print(f"✅ Frontend built and copied to {STATIC_DIR}")
    else:
        print("⚠️  Frontend build did not produce a dist/ directory.")


def start_server():
    """Start the FastAPI production server."""
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")

    print(f"🚀 Starting Semiconductor Yield Optimization API on {host}:{port}")

    os.chdir(str(SRC_DIR))
    sys.path.insert(0, str(SRC_DIR))

    import uvicorn
    uvicorn.run(
        "api:app",
        host=host,
        port=port,
        workers=1,  # Single worker to share in-memory model state
        log_level="info",
    )


if __name__ == "__main__":
    # Load .env if present
    env_file = ROOT_DIR / ".env"
    if env_file.exists():
        from dotenv import load_dotenv
        load_dotenv(env_file)
        print("📦 Loaded .env")

    # Build frontend if not already built
    if not STATIC_DIR.exists():
        try:
            build_frontend()
        except Exception as e:
            print(f"⚠️  Frontend build failed (non-critical): {e}")

    start_server()
