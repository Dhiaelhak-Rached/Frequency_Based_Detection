"""Small helper used by the download_model.py scripts."""
import os
import urllib.request


def fetch(files, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    ok = True
    for name, url in files.items():
        dst = os.path.join(out_dir, name)
        if os.path.exists(dst) and os.path.getsize(dst) > 10_000:
            print(f"already there: {dst}")
            continue
        try:
            print(f"downloading {name} ...", flush=True)
            urllib.request.urlretrieve(url, dst)
            size = os.path.getsize(dst)
            if size < 10_000:  # an error page or a Git LFS pointer, not a model
                os.remove(dst)
                raise RuntimeError(f"got only {size} bytes")
            print(f"  ok ({size / 1e6:.1f} MB) -> {dst}")
        except Exception as e:
            ok = False
            print(f"  FAILED: {e}\n  -> download it by hand from {url}\n     and save it as {dst}")
    return ok


def fetch_gdrive(files, out_dir, folder_url):
    """files: {local name: Google Drive file id}. Needs gdown (MIT), installed in .venv-sutrack."""
    try:
        import gdown
    except ImportError:
        print("needs gdown (pip install gdown, in the same environment as the trackers), or download by hand from "
              f"{folder_url}")
        return False
    os.makedirs(out_dir, exist_ok=True)
    ok = True
    for name, fid in files.items():
        dst = os.path.join(out_dir, name)
        if os.path.exists(dst) and os.path.getsize(dst) > 1_000_000:
            print(f"already there: {dst}")
            continue
        print(f"downloading {name} ...", flush=True)
        try:
            if not gdown.download(id=fid, output=dst, quiet=False):
                raise RuntimeError("gdown returned nothing")
            print(f"  ok ({os.path.getsize(dst) / 1e6:.0f} MB) -> {dst}")
        except Exception as e:
            ok = False
            print(f"  FAILED: {e}\n  -> download it by hand from {folder_url}\n     and save it as {dst}")
    return ok
