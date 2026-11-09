"""Prepares VTAB TFDS datasets from a local archive of .tar.zst files.

The archive holds *extracted* dataset contents repacked as `<name>.tar.zst`,
not the original download files, so TFDS's checksummed downloads cannot use
them directly. This script:

  1. Unpacks the needed members of each archive into a local staging dir.
  2. Patches `tfds.download.DownloadManager` so that the URLs each builder
     requests resolve to those staged files/directories (and `iter_archive`
     can walk a directory as if it were the original zip/tar).
  3. Runs the normal `builder.download_and_prepare()`, so the generated TFDS
     data, splits and preprocessing are exactly TFDS's own.

URLs not listed here (e.g. small files already in the TFDS download cache)
fall through to the normal TFDS download path.

Usage:
  python scripts/prepare_from_archive.py --list
  python scripts/prepare_from_archive.py --datasets caltech101,dmlab
  python scripts/prepare_from_archive.py            # all VTAB-19 datasets
"""

import argparse
import concurrent.futures
import hashlib
import os
import pathlib
import pickle
import shutil
import subprocess
import sys
import time

from etils import epath
import numpy as np
import tensorflow_datasets as tfds

# Each entry: builder spec (as pinned in task_adaptation/data/*.py), archive
# file, tar members to extract, optional tar --transform, and how URLs map to
# staged paths. Map values are (path relative to staging, mode) where mode is
#   "dir"     - a pre-extracted directory (returned for download/extract),
#   "file"    - an already-extracted file, returned as-is,
#   "archive" - an original download file, which TFDS extracts normally.
_SUN = "https://vision.princeton.edu/projects/2010/SUN/"
_CAMELYON = "https://zenodo.org/record/2546921/files/camelyonpatch_level_2_split_"
_SVHN = "http://ufldl.stanford.edu/housenumbers/"
DATASETS = {
    "caltech101": dict(
        spec="caltech101:3.*.*",
        archive="caltech101.tar.zst",
        members=["caltech101/101_ObjectCategories"],
        # Builder extracts <zip>/caltech-101/101_ObjectCategories.tar.gz and
        # expects a single top-level dir inside it.
        transform="s|^caltech101/|caltech101/caltech-101/101_ObjectCategories.tar.gz/|",
        urls={
            "https://data.caltech.edu/records/mzrjq-6wc02/files/caltech-101.zip?download=1":
                ("caltech101", "dir"),
        },
    ),
    "cifar100": dict(
        spec="cifar100:3.*.*",
        # TFDS reads the *binary* CIFAR release; the archive has the python
        # pickles of the same data, which convert_cifar100() rewrites into the
        # binary layout (records: coarse label, fine label, 3072 pixel bytes).
        # (The original Toronto server is very slow and often refuses TLS.)
        archive="cifar-100-python.tar.zst",
        members=["cifar-100-python"],
        transform="s|^|cifar100/|",
        convert="cifar100",
        urls={
            "https://www.cs.toronto.edu/~kriz/cifar-100-binary.tar.gz":
                ("cifar100", "dir"),
        },
    ),
    "cifar10": dict(
        # Not a VTAB-19 task, but cifar_test.py covers it. Same conversion as
        # cifar100 (binary records: label, 3072 pixel bytes).
        spec="cifar10:3.*.*",
        archive="cifar-10-batches-py.tar.zst",
        members=["cifar-10-batches-py"],
        transform="s|^|cifar10/|",
        convert="cifar10",
        urls={
            "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz":
                ("cifar10", "dir"),
        },
    ),
    "oxford_flowers102": dict(
        spec="oxford_flowers102:2.*.*",
        archive="flowers-102.tar.zst",
        members=["flowers-102/jpg"],
        urls={
            "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz":
                ("flowers-102", "dir"),
            "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat":
                ("flowers-102/imagelabels.mat", "file"),
            "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/setid.mat":
                ("flowers-102/setid.mat", "file"),
        },
        extra_members=["flowers-102/imagelabels.mat", "flowers-102/setid.mat"],
    ),
    "svhn_cropped": dict(
        spec="svhn_cropped:3.*.*",
        archive="svhn.tar.zst",
        members=["svhn/train_32x32.mat", "svhn/test_32x32.mat",
                 "svhn/extra_32x32.mat"],
        urls={
            _SVHN + "train_32x32.mat": ("svhn/train_32x32.mat", "file"),
            _SVHN + "test_32x32.mat": ("svhn/test_32x32.mat", "file"),
            _SVHN + "extra_32x32.mat": ("svhn/extra_32x32.mat", "file"),
        },
    ),
    "sun397": dict(
        spec="sun397/tfds:4.*.*",
        archive="SUN397.tar.zst",
        members=["SUN397/SUN397"],
        # iter_archive must yield paths starting with "SUN397/".
        transform="s|^SUN397/|sun397/|",
        urls={
            _SUN + "SUN397.tar.gz": ("sun397", "dir"),
            # Only used by the "standard-part*" configs, not "tfds".
            _SUN + "download/Partitions.zip": ("sun397", "dir"),
        },
    ),
    "patch_camelyon": dict(
        spec="patch_camelyon:2.*.*",
        archive="patch_camelyon.tar.zst",
        members=[f"patch_camelyon/camelyonpatch_level_2_split_{s}_{xy}.h5"
                 for s in ("train", "valid", "test") for xy in ("x", "y")],
        urls={
            f"{_CAMELYON}{s}_{xy}.h5.gz":
                (f"patch_camelyon/camelyonpatch_level_2_split_{s}_{xy}.h5", "file")
            for s in ("train", "valid", "test") for xy in ("x", "y")
        },
    ),
    "eurosat": dict(
        spec="eurosat/rgb:2.*.*",
        archive="eurosat.tar.zst",
        members=["eurosat/2750"],
        urls={"http://madm.dfki.de/files/sentinel/EuroSAT.zip": ("eurosat", "dir")},
    ),
    "resisc45": dict(
        spec="resisc45:3.*.*",
        archive="NWPU-RESISC45.tar.zst",
        members=["NWPU-RESISC45"],
        transform="s|^|resisc45/|",
        manual_dir="resisc45",
    ),
    "diabetic_retinopathy_detection": dict(
        spec="diabetic_retinopathy_detection/btgraham-300:3.*.*",
        archive="retinopathy.tar.zst",
        members=["retinopathy/train", "retinopathy/test", "retinopathy/sample",
                 "retinopathy/trainLabels.csv"],
        manual_dir="retinopathy",
    ),
    "clevr": dict(
        spec="clevr:3.*.*",
        archive="CLEVR_v1.0.tar.zst",
        members=["CLEVR_v1.0"],
        transform="s|^|clevr/|",
        urls={"https://dl.fbaipublicfiles.com/clevr/CLEVR_v1.0.zip": ("clevr", "dir")},
    ),
    "dmlab": dict(
        spec="dmlab:2.0.1",
        archive="dmlab.tar.zst",
        members=["dmlab"],
        urls={"https://storage.googleapis.com/dmlab-vtab/dmlab.tar.gz": ("dmlab", "dir")},
    ),
    "kitti": dict(
        spec="kitti:3.3.0",
        archive="Kitti.tar.zst",
        members=["Kitti/raw/training/image_2"],
        # iter_archive must yield "training/image_2/xxxxxx.png".
        transform="s|^Kitti/raw/|kitti_images/|",
        urls={
            "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_2.zip":
                ("kitti_images", "dir"),
        },
        # The label and devkit zips are small and are downloaded by TFDS.
    ),
}
# Already handled by plain TFDS downloads (fast sources), listed for --list.
OTHER_VTAB = ["dtd", "oxford_iiit_pet", "dsprites", "smallnorb"]


def log(msg):
  print(f"[prepare] {msg}", flush=True)


def stage_archive(name, cfg, archive_dir, staging):
  """Extracts the needed members of the dataset's archive (once)."""
  if "archive" not in cfg:
    return
  marker = staging / f".{name}.extracted"
  if marker.exists():
    log(f"{name}: already staged")
    return
  archive = archive_dir / cfg["archive"]
  members = cfg["members"] + cfg.get("extra_members", [])
  cmd = ["tar", "-x", "-C", str(staging), "--use-compress-program=zstd -d -T0"]
  if "transform" in cfg:
    cmd += ["--transform", cfg["transform"]]
  cmd += ["-f", str(archive)] + members
  log(f"{name}: extracting {archive.name} ({archive.stat().st_size / 1e9:.1f} GB)")
  start = time.time()
  subprocess.run(cmd, check=True)
  marker.touch()
  log(f"{name}: extracted in {time.time() - start:.0f}s")


def convert_cifar100(staging):
  """Writes cifar-100-binary/ from the extracted cifar-100-python/ pickles."""
  src = staging / "cifar100" / "cifar-100-python"
  out = staging / "cifar100" / "cifar-100-binary"
  if (out / "test.bin").exists():
    return
  out.mkdir(parents=True, exist_ok=True)
  for split, expected in (("train", 50000), ("test", 10000)):
    with open(src / split, "rb") as f:
      d = pickle.load(f, encoding="bytes")
    data = np.asarray(d[b"data"], dtype=np.uint8)
    assert data.shape == (expected, 3072), data.shape
    records = np.concatenate([
        np.asarray(d[b"coarse_labels"], dtype=np.uint8)[:, None],
        np.asarray(d[b"fine_labels"], dtype=np.uint8)[:, None],
        data,
    ], axis=1)
    records.tofile(out / f"{split}.bin")
  with open(src / "meta", "rb") as f:
    meta = pickle.load(f, encoding="bytes")
  for key in ("coarse_label_names", "fine_label_names"):
    names = [n.decode() for n in meta[key.encode()]]
    (out / f"{key}.txt").write_text("\n".join(names) + "\n")
  log("cifar100: converted python pickles to binary format")


def convert_cifar10(staging):
  """Writes cifar-10-batches-bin/ from the extracted cifar-10-batches-py/."""
  src = staging / "cifar10" / "cifar-10-batches-py"
  out = staging / "cifar10" / "cifar-10-batches-bin"
  if (out / "test_batch.bin").exists():
    return
  out.mkdir(parents=True, exist_ok=True)
  for name in [f"data_batch_{i}" for i in range(1, 6)] + ["test_batch"]:
    with open(src / name, "rb") as f:
      d = pickle.load(f, encoding="bytes")
    data = np.asarray(d[b"data"], dtype=np.uint8)
    assert data.shape == (10000, 3072), data.shape
    labels = np.asarray(d[b"labels"], dtype=np.uint8)[:, None]
    np.concatenate([labels, data], axis=1).tofile(out / f"{name}.bin")
  with open(src / "batches.meta", "rb") as f:
    meta = pickle.load(f, encoding="bytes")
  names = [n.decode() for n in meta[b"label_names"]]
  (out / "batches.meta.txt").write_text("\n".join(names) + "\n")
  log("cifar10: converted python pickles to binary format")


CONVERTERS = {"cifar100": convert_cifar100, "cifar10": convert_cifar10}


def fetch_parallel(url, dest, connections=16):
  """Downloads `url` to `dest` with parallel ranged requests (via curl)."""
  if dest.exists():
    return
  dest.parent.mkdir(parents=True, exist_ok=True)
  head = subprocess.run(["curl", "-sIL", url], capture_output=True, text=True,
                        check=True).stdout
  size = int([l for l in head.lower().splitlines()
              if l.startswith("content-length")][-1].split(":")[1])
  chunk = -(-size // connections)
  parts = [dest.with_name(f"{dest.name}.part{i}") for i in range(connections)]
  log(f"fetching {url} ({size / 1e6:.0f} MB, {connections} connections)")

  def get(i):
    lo, hi = i * chunk, min(size, (i + 1) * chunk) - 1
    for _ in range(10):
      have = parts[i].stat().st_size if parts[i].exists() else 0
      if have == hi - lo + 1:
        return
      subprocess.run(["curl", "-sL", "--retry", "5", "-r", f"{lo + have}-{hi}",
                      url, "-o", "-"], stdout=open(parts[i], "ab"), check=False)
    raise RuntimeError(f"failed to fetch range {lo}-{hi} of {url}")

  with concurrent.futures.ThreadPoolExecutor(connections) as pool:
    list(pool.map(get, range(connections)))
  with open(dest.with_suffix(".tmp"), "wb") as out:
    for p in parts:
      with open(p, "rb") as f:
        shutil.copyfileobj(f, out)
  dest.with_suffix(".tmp").rename(dest)
  for p in parts:
    p.unlink()


def sha256(path):
  h = hashlib.sha256()
  with open(path, "rb") as f:
    for block in iter(lambda: f.read(1 << 22), b""):
      h.update(block)
  return h.hexdigest()


def install_download_patch(url_map):
  """Routes mapped URLs to staged paths inside tfds' DownloadManager."""
  dm = tfds.download.DownloadManager
  orig_download = dm.download
  orig_extract = dm.extract
  orig_dl_extract = dm.download_and_extract
  orig_iter_archive = dm.iter_archive

  def url_of(x):
    return x.url if isinstance(x, tfds.download.Resource) else x

  def nested(obj, fn):
    if isinstance(obj, dict):
      return {k: nested(v, fn) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
      return type(obj)(nested(v, fn) for v in obj)
    return fn(obj)

  def download(self, url_or_urls):
    def one(x):
      if url_of(x) in url_map:
        return epath.Path(url_map[url_of(x)][0])
      return orig_download(self, x)
    return nested(url_or_urls, one)

  def extract(self, path_or_paths):
    def one(x):
      path = x.path if isinstance(x, tfds.download.Resource) else x
      if path is not None and os.path.isdir(path):
        return epath.Path(path)
      return orig_extract(self, x)
    return nested(path_or_paths, one)

  def download_and_extract(self, url_or_urls):
    def one(x):
      if url_of(x) in url_map:
        path, mode = url_map[url_of(x)]
        if mode == "archive":
          return orig_extract(self, epath.Path(path))
        return epath.Path(path)
      return orig_dl_extract(self, x)
    return nested(url_or_urls, one)

  def iter_archive(self, resource):
    path = resource.path if isinstance(resource, tfds.download.Resource) else resource
    if not os.path.isdir(path):
      return orig_iter_archive(self, resource)

    def walk():
      root = os.fspath(path)
      for dirpath, dirnames, filenames in os.walk(root, followlinks=True):
        dirnames.sort()
        for fname in sorted(filenames):
          full = os.path.join(dirpath, fname)
          with open(full, "rb") as f:
            yield os.path.relpath(full, root), f
    return walk()

  dm.download = download
  dm.extract = extract
  dm.download_and_extract = download_and_extract
  dm.iter_archive = iter_archive


def main():
  parser = argparse.ArgumentParser(description=__doc__,
                                   formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("--archive_dir", default="~/bigbox/datasets")
  parser.add_argument("--staging_dir", default=None,
                      help="Where archives are unpacked (default: "
                      "<TFDS data dir>/staging). Safe to delete afterwards.")
  parser.add_argument("--data_dir", default=None,
                      help="TFDS data dir (default: TFDS default).")
  parser.add_argument("--datasets", default=",".join(DATASETS),
                      help="Comma-separated subset of: " + ", ".join(DATASETS))
  parser.add_argument("--no_verify", action="store_true",
                      help="Skip sha256 checks of original files.")
  parser.add_argument("--list", action="store_true")
  args = parser.parse_args()

  if args.list:
    for name, cfg in DATASETS.items():
      print(f"{name:32s} {cfg['spec']:50s} {cfg.get('archive', 'fetched')}")
    print("Prepared by plain TFDS downloads:", ", ".join(OTHER_VTAB))
    return

  archive_dir = pathlib.Path(args.archive_dir).expanduser()
  data_dir = args.data_dir or tfds.core.constants.DATA_DIR
  staging = pathlib.Path(args.staging_dir or os.path.join(data_dir, "staging")).expanduser()
  staging.mkdir(parents=True, exist_ok=True)
  names = [n.strip() for n in args.datasets.split(",") if n.strip()]
  unknown = set(names) - set(DATASETS)
  if unknown:
    sys.exit(f"Unknown datasets: {sorted(unknown)}")

  url_map = {}
  for name in names:
    for url, (rel, mode) in DATASETS[name].get("urls", {}).items():
      url_map[url] = (os.fspath(staging / rel), mode)
  install_download_patch(url_map)

  failed = []
  for name in names:
    cfg = DATASETS[name]
    try:
      builder = tfds.builder(cfg["spec"], data_dir=args.data_dir)
      if builder.info.splits:
        log(f"{name}: {builder.info.full_name} already prepared, skipping")
        continue
      stage_archive(name, cfg, archive_dir, staging)
      if "convert" in cfg:
        CONVERTERS[cfg["convert"]](staging)
      for url, rel in cfg.get("fetch", {}).items():
        fetch_parallel(url, staging / rel)
      if not args.no_verify:
        # Files that are byte-identical to the original downloads can be
        # checked against TFDS's registered checksums.
        for url, (rel, mode) in cfg.get("urls", {}).items():
          info = builder.url_infos.get(url) if builder.url_infos else None
          identical = mode == "archive" or (
              mode == "file" and os.path.basename(url) == os.path.basename(rel))
          if identical and info:
            got = sha256(staging / rel)
            if got != info.checksum:
              raise ValueError(f"checksum mismatch for {rel}: {got} != {info.checksum}")
            log(f"{name}: checksum OK for {rel}")
      config = tfds.download.DownloadConfig(
          manual_dir=os.fspath(staging / cfg["manual_dir"]) if "manual_dir" in cfg else None)
      log(f"{name}: generating {builder.info.full_name}")
      start = time.time()
      builder.download_and_prepare(download_config=config)
      splits = {k: v.num_examples for k, v in builder.info.splits.items()}
      log(f"{name}: done in {time.time() - start:.0f}s: {splits}")
    except Exception as e:  # pylint: disable=broad-except
      log(f"{name}: FAILED: {type(e).__name__}: {e}")
      failed.append(name)
  if failed:
    sys.exit(f"Failed: {failed}")


if __name__ == "__main__":
  main()
