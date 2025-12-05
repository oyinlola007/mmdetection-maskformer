import argparse
import os
import os.path as osp
from copy import deepcopy

from mmengine import ConfigDict
from mmengine.config import Config, DictAction
from mmengine.runner import Runner
from mmengine.hooks import Hook


class ImagePathLoggerHook(Hook):
    """Logs image paths during test iterations."""

    def __init__(self, interval: int = 1):
        self.interval = interval

    def after_test_iter(
        self, runner: Runner, batch_idx: int, data_batch=None, outputs=None
    ) -> None:
        if batch_idx % self.interval != 0:
            return
        if not outputs:
            return
        paths = []
        for sample in outputs:
            if hasattr(sample, "metainfo"):
                img_path = sample.metainfo.get("img_path")
            else:
                img_path = sample.get("img_path") if isinstance(sample, dict) else None
            if img_path:
                paths.append(img_path)
        if paths:
            runner.logger.info(f"[eval] batch {batch_idx + 1} image paths: {paths}")


from codecarbon import EmissionsTracker
import time
import csv
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a model on a subset (first N samples)"
    )
    parser.add_argument("config", help="config file path")
    parser.add_argument("checkpoint", help="checkpoint file path or URL")
    parser.add_argument(
        "--n", type=int, default=5, help="number of samples to evaluate"
    )
    parser.add_argument("--work-dir", help="directory to save logs/metrics")
    parser.add_argument(
        "--launcher", choices=["none", "pytorch", "slurm", "mpi"], default="none"
    )
    parser.add_argument(
        "--tta", action="store_true", help="enable TTA if available in config"
    )
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        help="override config keys (key=val)",
    )
    parser.add_argument(
        "--dataset-label",
        type=str,
        default=None,
        help="Optional label to record in logs for the evaluated dataset.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="unknown",
        help="Device name for metrics logging (e.g., macbook, zedbox, nvidia).",
    )
    # local rank passthrough
    parser.add_argument("--local_rank", "--local-rank", type=int, default=0)
    args = parser.parse_args()
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)
    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # work_dir precedence: CLI > config > default from config name
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get("work_dir", None) is None:
        cfg.work_dir = osp.join(
            "./work_dirs", osp.splitext(osp.basename(args.config))[0] + f"_n{args.n}"
        )

    # limit number of samples and stabilize dataloader on macOS
    test_data_cfg = cfg.test_dataloader
    # unwrap nested dataset if needed to reach the actual dataset cfg
    ds = test_data_cfg.get("dataset", test_data_cfg.get("dataset", None))
    if ds is None:
        ds = (
            test_data_cfg.get("dataset")
            if "dataset" in test_data_cfg
            else test_data_cfg
        )
    # Some configs use dict nesting under test_dataloader['dataset']
    dataset_cfg = test_data_cfg.get("dataset", test_data_cfg)
    # ensure indices on the leaf dataset (unwrap Concat/Repeat if present is out of scope -> apply at top)
    if args.n is None or args.n <= 0:
        dataset_cfg.pop("indices", None)
    else:
        dataset_cfg["indices"] = list(range(max(args.n, 1)))
    # single-worker, non-persistent workers to avoid hangs
    cfg.test_dataloader["num_workers"] = 0
    cfg.test_dataloader["persistent_workers"] = False
    if "sampler" in cfg.test_dataloader:
        cfg.test_dataloader["sampler"]["shuffle"] = False

    # improve progress visibility
    cfg.setdefault("default_hooks", {})
    logger_hook = cfg.default_hooks.get("logger", {"type": "LoggerHook", "interval": 1})
    logger_hook["interval"] = 1
    cfg.default_hooks["logger"] = logger_hook
    # remove visualization hook entirely to avoid version arg mismatches
    if "visualization" in cfg.default_hooks:
        cfg.default_hooks.pop("visualization")

    # Optional TTA support (mirrors tools/test.py minimal)
    if args.tta:
        if "tta_model" not in cfg:
            cfg.tta_model = dict(
                type="DetTTAModel",
                tta_cfg=dict(nms=dict(type="nms", iou_threshold=0.5), max_per_img=100),
            )
        if "tta_pipeline" not in cfg:
            # get the actual dataset pipeline
            dcfg = (
                cfg.test_dataloader["dataset"]
                if "dataset" in cfg.test_dataloader
                else cfg.test_dataloader
            )
            while isinstance(dcfg, dict) and "dataset" in dcfg:
                dcfg = dcfg["dataset"]
            cfg.tta_pipeline = deepcopy(dcfg["pipeline"])
        cfg.model = ConfigDict(**cfg.tta_model, module=cfg.model)
        cfg.test_dataloader["dataset"]["pipeline"] = cfg.tta_pipeline

    cfg.load_from = args.checkpoint

    # build and run
    runner = Runner.from_cfg(cfg)
    runner.register_hook(ImagePathLoggerHook(interval=1), priority="NORMAL")

    # emissions tracking
    work_dir = cfg.work_dir if cfg.get("work_dir", None) else "./work_dirs"
    Path(work_dir).mkdir(parents=True, exist_ok=True)
    tracker = EmissionsTracker(output_dir=work_dir, project_name=osp.basename(work_dir))
    tracker.start()
    t0 = time.time()

    metrics = runner.test()

    total_s = time.time() - t0
    emissions_kg = tracker.stop()

    # Get actual number of samples processed
    # When --n 0 is used, args.n is 0 but we need the actual dataset size
    actual_num_samples = args.n
    if args.n is None or args.n <= 0:
        # Get actual dataset length from the test dataloader
        try:
            test_dataloader = runner.test_loop.dataloader
            actual_num_samples = len(test_dataloader.dataset)
        except (AttributeError, TypeError):
            # Fallback: try to get from config
            try:
                leaf_dataset_cfg = cfg.test_dataloader.get("dataset", cfg.test_dataloader)
                while isinstance(leaf_dataset_cfg, dict) and "dataset" in leaf_dataset_cfg:
                    leaf_dataset_cfg = leaf_dataset_cfg["dataset"]
                # If we have indices, use that length; otherwise we can't determine it
                if "indices" in leaf_dataset_cfg:
                    actual_num_samples = len(leaf_dataset_cfg["indices"])
                else:
                    # Can't determine, use args.n (which is 0)
                    actual_num_samples = args.n if args.n else 0
            except (AttributeError, TypeError, KeyError):
                actual_num_samples = args.n if args.n else 0

    # summary CSV (append)
    out_csv = Path("outputs/run_metrics.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    # determine dataset label for logging
    def _extract_dataset_cfg(dcfg):
        leaf = dcfg
        while isinstance(leaf, dict) and "dataset" in leaf:
            leaf = leaf["dataset"]
        return leaf

    leaf_dataset_cfg = _extract_dataset_cfg(
        cfg.test_dataloader.get("dataset", cfg.test_dataloader)
    )
    dataset_label = args.dataset_label
    if dataset_label is None:
        dataset_label = leaf_dataset_cfg.get("type")
    if dataset_label is None:
        data_root = leaf_dataset_cfg.get("data_root")
        if data_root:
            dataset_label = osp.basename(osp.normpath(data_root))
    if dataset_label is None:
        ann_file = leaf_dataset_cfg.get("ann_file")
        if ann_file:
            dataset_label = osp.basename(ann_file)
    if dataset_label is None:
        dataset_label = "unknown_dataset"

    row = {
        "device": args.device,
        "experiment": osp.basename(work_dir),
        "model": osp.basename(args.config),
        "dataset": dataset_label,
        "num_samples": actual_num_samples,
        "total_seconds": round(total_s, 3),
        "fps": round(actual_num_samples / total_s, 3) if total_s > 0 and actual_num_samples > 0 else 0.0,
        "emissions_kg": emissions_kg if emissions_kg is not None else None,
        "PQ": metrics.get("coco_panoptic/PQ") if isinstance(metrics, dict) else None,
        "SQ": metrics.get("coco_panoptic/SQ") if isinstance(metrics, dict) else None,
        "RQ": metrics.get("coco_panoptic/RQ") if isinstance(metrics, dict) else None,
    }
    file_exists = out_csv.exists()
    
    # Ensure file ends with newline before appending (fixes manual edits without trailing newline)
    if file_exists:
        with open(out_csv, "rb") as f:
            f.seek(-1, 2)  # Go to last byte
            last_char = f.read(1)
            if last_char != b'\n':
                # File doesn't end with newline, add one
                with open(out_csv, "a") as f_append:
                    f_append.write('\n')
    
    with open(out_csv, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
    print("Subset eval complete:", row)


if __name__ == "__main__":
    main()
