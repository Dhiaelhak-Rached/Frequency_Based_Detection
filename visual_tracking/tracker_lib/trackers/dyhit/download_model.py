#!/usr/bin/env python3
"""Download the dyhit checkpoint into trackers/dyhit/models/ (Google Drive, needs gdown)."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from common.download import fetch_gdrive  # noqa: E402

FOLDER = "https://drive.google.com/drive/folders/15VTIJnUtJTdU6TcmGOixSEcErYV-h_xL"
FILES = {"DyHiT_ep0060.pth.tar": "16zfFdgA77KScEfPrEsCKja6VINVo_8nD"}

if __name__ == "__main__":
    sys.exit(0 if fetch_gdrive(FILES, os.path.join(HERE, "models"), FOLDER) else 1)
