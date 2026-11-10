# coding=utf-8
"""Export the 19 VTAB tasks to framework-free uint8 arrays for linear probing.

For each task, images go through VTAB's own pipeline: the task_adaptation data class
(its splits, label definitions and base preprocessing), then data_loader.preprocess_fn's
tf.image.resize to SIZE x SIZE (bilinear, no crop). They are stored as uint8.

Splits follow Huh et al. 2024 (Platonic Representation Hypothesis), App. C.1: train
and val are randomly subsampled to at most --max_trainval examples each (fixed seed),
and test is kept whole.

Output per task, under <out_dir>/<task>/:
  {train,val,test}_images.npy  uint8 [N, SIZE, SIZE, 3]
  {train,val,test}_labels.npy  int64 [N]
  {train,val,test}_index.npy   int64 [N], position of each example within the VTAB split
  meta.json                    spec, num_classes, split sizes, tfds version

Usage:
  python scripts/export_vtab_arrays.py --data_dir ~/datasets/tensorflow_datasets \
      --out_dir ~/datasets/vtab-224 [--tasks cifar100,dtd] [--max_test N]
"""
import json
import os
import time

from absl import app
from absl import flags
import numpy as np
import tensorflow.compat.v1 as tf

from task_adaptation import data_loader

FLAGS = flags.FLAGS
flags.DEFINE_string("data_dir", None, "TFDS data dir.")
flags.DEFINE_string("out_dir", None, "Output root.")
flags.DEFINE_list("tasks", None, "Subset of task names (default: all 19).")
flags.DEFINE_integer("size", 224, "Square output resolution.")
flags.DEFINE_integer("max_trainval", 10000, "Cap for train and for val (paper: 10k).")
flags.DEFINE_integer("max_test", None, "Optional cap for test (testing only; paper: none).")
flags.DEFINE_integer("seed", 0, "Subsampling seed.")
flags.DEFINE_integer("batch_size", 256, "Batch size for the tf.data pipeline.")

# Same 19 specs as scripts/run_all_tasks.sh, with short names.
TASKS = {
    "caltech101": "caltech101",
    "cifar100": "cifar(num_classes=100)",
    "dtd": "dtd",
    "flowers102": "oxford_flowers102",
    "pets": "oxford_iiit_pet",
    "sun397": "sun397",
    "svhn": "svhn",
    "camelyon": "patch_camelyon",
    "eurosat": "eurosat",
    "resisc45": "resisc45",
    "retinopathy": 'diabetic_retinopathy(config="btgraham-300")',
    "clevr_count": 'clevr(task="count_all")',
    "clevr_distance": 'clevr(task="closest_object_distance")',
    "dmlab": "dmlab",
    "dsprites_location": 'dsprites(predicted_attribute="label_x_position",num_classes=16)',
    "dsprites_orientation": 'dsprites(predicted_attribute="label_orientation",num_classes=16)',
    "kitti_distance": 'kitti(task="closest_vehicle_distance")',
    "smallnorb_azimuth": 'smallnorb(predicted_attribute="label_azimuth")',
    "smallnorb_elevation": 'smallnorb(predicted_attribute="label_elevation")',
}


def _resize_fn(size):
  def fn(data):
    image = data["image"]
    image = tf.cond(tf.equal(tf.shape(image)[-1], 1),
                    lambda: tf.tile(image, [1, 1, 3]), lambda: image)
    image = tf.image.resize(image, [size, size])  # = data_loader.preprocess_fn
    image = tf.cast(tf.clip_by_value(tf.round(image), 0, 255), tf.uint8)
    return {"image": image, "label": tf.cast(data["label"], tf.int64)}
  return fn


def export_split(data, split, out_prefix, cap, rng):
  n_total = data.get_num_samples(split)
  if cap is not None and n_total > cap:
    keep = np.sort(rng.choice(n_total, size=cap, replace=False))
  else:
    keep = np.arange(n_total)
  n = len(keep)
  mask = np.zeros(n_total, dtype=bool)
  mask[keep] = True

  ds = data._get_dataset_split(split, shuffle_files=False)  # deterministic TFDS order
  if n < n_total:
    mask_t = tf.constant(mask)

    def keep_fn(i, x):
      del x
      return tf.gather(mask_t, i)

    def drop_index_fn(i, x):
      del i
      return x

    ds = ds.enumerate().filter(keep_fn).map(drop_index_fn)
  ds = data._preprocess_and_batch_data(ds, FLAGS.batch_size, drop_remainder=False,
                                       preprocess_fn=_resize_fn(FLAGS.size))

  s = FLAGS.size
  tmp = out_prefix + "_images.npy.partial"
  images = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.uint8, shape=(n, s, s, 3))
  labels = np.empty(n, dtype=np.int64)
  pos = 0
  for batch in ds.as_numpy_iterator():
    b = len(batch["label"])
    images[pos:pos + b] = batch["image"]
    labels[pos:pos + b] = batch["label"]
    pos += b
  assert pos == n, f"{out_prefix}: got {pos} examples, expected {n}"
  images.flush()
  del images
  np.save(out_prefix + "_labels.npy", labels)
  np.save(out_prefix + "_index.npy", keep.astype(np.int64))
  os.replace(tmp, out_prefix + "_images.npy")  # images last: its presence marks the split done
  return n_total, n


def main(argv):
  del argv
  tf.enable_eager_execution()
  names = FLAGS.tasks or list(TASKS)
  for name in names:
    out = os.path.join(os.path.expanduser(FLAGS.out_dir), name)
    if os.path.exists(os.path.join(out, "meta.json")):
      print(f"{name}: done, skipping", flush=True)
      continue
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    data = data_loader.get_dataset_instance(
        {"dataset": "data." + TASKS[name], "data_dir": os.path.expanduser(FLAGS.data_dir)})
    meta = {"task": name, "spec": TASKS[name], "size": FLAGS.size,
            "num_classes": data.get_num_classes(), "tfds_version": data.get_version(),
            "seed": FLAGS.seed, "max_trainval": FLAGS.max_trainval,
            "max_test": FLAGS.max_test, "splits": {}}
    for i, (split, cap) in enumerate([("train", FLAGS.max_trainval), ("val", FLAGS.max_trainval),
                                      ("test", FLAGS.max_test)]):
      prefix = os.path.join(out, split)
      rng = np.random.RandomState(FLAGS.seed + i)
      n_total, n = export_split(data, split, prefix, cap, rng)
      meta["splits"][split] = {"vtab_size": n_total, "exported": n}
      print(f"{name}/{split}: {n}/{n_total}  ({time.time() - t0:.0f}s)", flush=True)
    labels = np.concatenate([np.load(os.path.join(out, f"{s}_labels.npy")) for s in ("train", "val", "test")])
    assert labels.min() >= 0 and labels.max() < meta["num_classes"], (name, labels.min(), labels.max())
    with open(os.path.join(out, "meta.json"), "w") as f:
      json.dump(meta, f, indent=1)


if __name__ == "__main__":
  flags.mark_flags_as_required(["data_dir", "out_dir"])
  app.run(main)
