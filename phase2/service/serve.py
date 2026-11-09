"""Run the exact released service on loopback; no W&B/Kaggle account is needed after setup."""
from pathlib import Path
import argparse
import os


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", default=str(Path(__file__).resolve().parent))
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    from app import create_app
    import uvicorn
    uvicorn.run(create_app(Path(args.release).resolve()), host="127.0.0.1", port=args.port,
                workers=1, access_log=False, log_level="warning")
