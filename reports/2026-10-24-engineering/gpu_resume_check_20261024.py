"""Actual CUDA epoch-boundary replay check; generated TRAIN windows only.

Standalone artifact, adapted from tests/test_metric_resume.py. No project files,
real datasets, Git commands, package installs, or CUDA mocks are used.
"""
from __future__ import annotations

import os

# Must precede the torch import and any CUDA context creation.
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

import json
import platform
import struct
import sys
import time
import traceback
from contextlib import ExitStack
from pathlib import Path
from unittest import mock

sys.dont_write_bytecode = True
REPO_ROOT = Path(__file__).resolve().parents[1] / "gait-biometrics"
ARTIFACTS = Path(__file__).resolve().parent
RUN_ROOT = ARTIFACTS / "gpu-resume-run-20261024"
REPORT_PATH = ARTIFACTS / "gpu-resume-20261024.json"
FIXED_CLOCK = 123.0
STATE_KEYS = (
    "model_state_dict", "classifier_state_dict", "optimizer_state_dict",
    "scheduler_state_dict", "scaler_state_dict", "rng_state", "sampler_state",
    "history", "completed_epochs", "best_epoch", "best_val_eer", "best_checkpoint",
    "elapsed_seconds",
)
CONFIG = {
    "source": "har", "device": "cuda", "torch_threads": 1,
    "epochs": 4, "steps_per_epoch": 2, "seed": 42, "split_seed": 42,
    "val_seed": 42, "window_size": 16, "validation_pairs": 20,
    "identities_per_batch": 2, "windows_per_identity": 2,
    "train_subject_ids": [3, 11, 29], "val_subject_ids": [41, 53],
    "objective": "hybrid", "classifier_type": "linear", "ce_weight": 0.5,
    "precision": "float32", "augment": {"jitter_std": 0.02},
    "model_args": {"width": 4, "embed_dim": 6, "dropout": 0.25},
    # An unused path inside this run; load_windows is replaced with generated data.
    "data_root": str(RUN_ROOT / "synthetic-only-no-dataset"),
}
REPORT = {
    "status": "running", "success": False, "reason": None,
    "scope": "actual local CUDA; synthetic TRAIN tensors only; no newdata",
    "fixture_source": "tests/test_metric_resume.py::_data/_config/test_split_run_matches_uninterrupted_with_dropout_augmentation_and_adam",
    "paths": {"script": str(Path(__file__).resolve()), "run_root": str(RUN_ROOT),
              "report": str(REPORT_PATH), "log": str(ARTIFACTS / "gpu-resume-20261024.log")},
    "versions": {"python": platform.python_version(), "python_executable": sys.executable},
    "config": CONFIG,
    "synthetic": {"shape": [24, 9, 16], "generation_seed": 73,
                  "subject_ids": [3, 11, 29, 41, 53, 67], "windows_per_subject": 4,
                  "optimizer_train_windows": 12, "held_out_train_validation_windows": 8,
                  "unused_windows": 4, "all_loader_splits": [], "real_data_read": False},
    "run_plan": {"total_epochs": 4, "steps_per_epoch": 2,
                 "uninterrupted_epochs": 4, "interrupted_after_committed_epoch": 2,
                 "resumed_remaining_epochs": 2,
                 "expected_optimizer_steps": {"full": 8, "split_initial": 4, "split_resume": 4}},
    "exact_checks": {key: None for key in STATE_KEYS},
    "additional_checks": {}, "observations": {},
    "timing": {"training_monotonic_clock_fixed_to": FIXED_CLOCK,
               "raw_wall_seconds": {},
               "note": "Trainer observational clock fixed at 123.0, so recorded epoch/total elapsed values are zero. Raw perf_counter wall times include startup, CUDA initialization, compilation/kernel setup, I/O, and instrumentation; no GPU performance claim."},
    "mocked_boundaries": ["generated load_windows", "trainer observational monotonic clock", "Git metadata synthetic-test", "post-commit interruption"],
    "cuda_mocked": False,
}


class InterruptedTraining(Exception):
    pass


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def exact(actual, expected, path="state"):
    """Recursive CPU tensor byte equality, including dtype/shape and float bits."""
    if isinstance(expected, torch.Tensor):
        require(isinstance(actual, torch.Tensor), f"{path}: expected tensor")
        require(actual.device.type == expected.device.type == "cpu", f"{path}: comparison must be CPU")
        require(actual.dtype == expected.dtype, f"{path}: dtype differs")
        require(actual.shape == expected.shape, f"{path}: shape differs")
        require(actual.layout == expected.layout, f"{path}: layout differs")
        left = actual.detach().contiguous().reshape(-1).view(torch.uint8)
        right = expected.detach().contiguous().reshape(-1).view(torch.uint8)
        require(torch.equal(left, right), f"{path}: tensor bytes differ")
    elif isinstance(expected, dict):
        require(isinstance(actual, dict), f"{path}: expected dict")
        require(actual.keys() == expected.keys(), f"{path}: keys differ")
        for key in expected:
            exact(actual[key], expected[key], f"{path}.{key}")
    elif isinstance(expected, (list, tuple)):
        require(type(actual) is type(expected), f"{path}: sequence type differs")
        require(len(actual) == len(expected), f"{path}: sequence length differs")
        for index, (left, right) in enumerate(zip(actual, expected)):
            exact(left, right, f"{path}[{index}]")
    else:
        require(type(actual) is type(expected), f"{path}: scalar type differs")
        if isinstance(expected, float):
            require(struct.pack("!d", actual) == struct.pack("!d", expected), f"{path}: float bits differ")
        else:
            require(actual == expected, f"{path}: value differs ({actual!r} != {expected!r})")


def record_exact(name, actual, expected, extra=False):
    checks = REPORT["additional_checks"] if extra else REPORT["exact_checks"]
    try:
        exact(actual, expected, name)
    except Exception as error:
        checks[name] = {"exact": False, "reason": str(error)}
        return False
    checks[name] = {"exact": True}
    return True


def load(name, artifact="last.pt"):
    return torch.load(RUN_ROOT / name / artifact, map_location="cpu", weights_only=True)


def assert_cuda_rng(payload, name):
    states = payload["rng_state"]["torch_cuda"]
    require(len(states) == torch.cuda.device_count() > 0, f"{name}: missing CUDA RNG state")
    for state in states:
        require(isinstance(state, torch.Tensor) and state.dtype == torch.uint8 and state.numel() > 0,
                f"{name}: invalid CUDA RNG state")
    require(payload["provenance"]["runtime"]["device"] == "cuda", f"{name}: device is not cuda")
    REPORT["observations"].setdefault("checkpoint_cuda_rng", {})[name] = {
        "device": payload["provenance"]["runtime"]["device"],
        "cuda_device_count": len(states), "rng_state_bytes": [state.numel() for state in states],
    }


def main():
    require(Path.cwd().resolve() == REPO_ROOT, "Unexpected working directory")
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "0", "Explicit CUDA_VISIBLE_DEVICES=0 required")
    require(not REPORT_PATH.exists(), "Report already exists; refusing overwrite")
    # Exclusive creation is intentional: do not reuse or remove previous artifacts.
    RUN_ROOT.mkdir(exist_ok=False)
    sys.path.insert(0, str(REPO_ROOT))
    global torch
    import torch
    from data.adapter import WindowData
    from engine import train_metric as metric

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    REPORT["versions"].update(torch=str(torch.__version__), cuda=torch.version.cuda,
                              cudnn=torch.backends.cudnn.version())
    REPORT["settings"] = {
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "torch_threads": torch.get_num_threads(),
    }
    require(torch.cuda.is_available(), "CUDA unavailable; no CPU fallback")
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    properties = torch.cuda.get_device_properties(device)
    REPORT["device"] = {
        "requested": "cuda", "actual": str(device), "name": properties.name,
        "index": torch.cuda.current_device(), "capability": list(torch.cuda.get_device_capability(device)),
        "total_memory_bytes": properties.total_memory,
        "cuda_device_count": torch.cuda.device_count(),
    }
    print(json.dumps({"device": REPORT["device"], "versions": REPORT["versions"],
                      "settings": REPORT["settings"]}), flush=True)
    observations = REPORT["observations"]
    observations.update(optimizer_steps={"full": 0, "split_initial": 0, "split_resume": 0},
                        encoder_forwards={}, classifier_forwards={}, augmentation_calls={},
                        live_parameter_devices=[], live_gradient_devices=[],
                        live_optimizer_moment_devices=[], dropout_probabilities=[])
    phase = {"name": None}
    loader_calls = []

    def synthetic_loader(source, split, **options):
        require(source == "har" and split == "train", "Only synthetic TRAIN loader allowed")
        loader_calls.append({"phase": phase["name"], "source": source, "split": split})
        REPORT["synthetic"]["all_loader_splits"].append(split)
        subjects = torch.tensor([3, 11, 29, 41, 53, 67]).repeat_interleave(4)
        windows = torch.randn(24, 9, 16, generator=torch.Generator().manual_seed(73))
        windows += subjects[:, None, None].float() / 40
        return WindowData(windows, torch.zeros(24, dtype=torch.long), subjects,
                          ["Walking"], subjects.unique().tolist(), 50.0)

    def observe_forward(kind):
        def hook(module, inputs, output):
            require(inputs[0].device.type == "cuda", f"{kind}: forward input not CUDA")
            require(output.device.type == "cuda", f"{kind}: forward output not CUDA")
            require(all(p.device.type == "cuda" for p in module.parameters()), f"{kind}: non-CUDA parameters")
            key = phase["name"] + (":train" if module.training else ":eval")
            counts = observations[kind + "_forwards"]
            counts[key] = counts.get(key, 0) + 1
        return hook

    encoder_factory = metric.build_gait_encoder
    classifier_factory = metric._build_classifier
    real_step = torch.optim.AdamW.step
    real_augment = metric._augment
    real_save = metric._atomic_torch_save

    def observed_encoder(*args, **kwargs):
        encoder = encoder_factory(*args, **kwargs)
        probabilities = [module.p for module in encoder.modules() if isinstance(module, torch.nn.Dropout)]
        require(probabilities and all(p == 0.25 for p in probabilities), "Dropout 0.25 must be active")
        observations["dropout_probabilities"].append(probabilities)
        encoder.register_forward_hook(observe_forward("encoder"))
        return encoder

    def observed_classifier(*args, **kwargs):
        classifier = classifier_factory(*args, **kwargs)
        require(classifier is not None, "Classifier must be enabled")
        classifier.register_forward_hook(observe_forward("classifier"))
        return classifier

    def observed_augment(x, options):
        require(x.device.type == "cuda" and options == {"jitter_std": 0.02}, "Jitter must run on CUDA")
        result = real_augment(x, options)
        require(result.device.type == "cuda", "Augmentation output not CUDA")
        require(not torch.equal(x, result), "Jitter made no change")
        name = phase["name"]
        observations["augmentation_calls"][name] = observations["augmentation_calls"].get(name, 0) + 1
        return result

    def observed_step(optimizer, *args, **kwargs):
        params = [p for group in optimizer.param_groups for p in group["params"]]
        require(params and all(p.device.type == "cuda" for p in params), "AdamW parameters not CUDA")
        require(all(p.grad is not None and p.grad.device.type == "cuda" for p in params), "AdamW gradients not CUDA")
        result = real_step(optimizer, *args, **kwargs)
        moments = [state[key] for state in optimizer.state.values() for key in ("exp_avg", "exp_avg_sq")]
        require(moments and all(value.device.type == "cuda" for value in moments), "AdamW moments not CUDA")
        observations["optimizer_steps"][phase["name"]] += 1
        for key, values in (("live_parameter_devices", params), ("live_gradient_devices", [p.grad for p in params]),
                            ("live_optimizer_moment_devices", moments)):
            observations[key] = sorted(set(observations[key]) | {str(value.device) for value in values})
        return result

    def training(name, directory, **options):
        phase["name"] = name
        raw_start = time.perf_counter_ns()
        try:
            return metric.train_metric(CONFIG, str(RUN_ROOT / directory), **options)
        finally:
            REPORT["timing"]["raw_wall_seconds"][name] = (time.perf_counter_ns() - raw_start) / 1e9
            print(json.dumps({"phase": name, "actual_optimizer_steps": observations["optimizer_steps"][name],
                              "raw_wall_seconds": REPORT["timing"]["raw_wall_seconds"][name]}), flush=True)

    # Wrap real operations to observe them; no CUDA API, kernel, or training computation is mocked.
    with ExitStack() as stack:
        stack.enter_context(mock.patch.object(metric, "load_windows", side_effect=synthetic_loader))
        stack.enter_context(mock.patch("engine.train_metric.time.monotonic", return_value=FIXED_CLOCK))
        stack.enter_context(mock.patch.object(metric, "_git_commit", return_value="synthetic-test"))
        stack.enter_context(mock.patch.object(metric, "build_gait_encoder", side_effect=observed_encoder))
        stack.enter_context(mock.patch.object(metric, "_build_classifier", side_effect=observed_classifier))
        stack.enter_context(mock.patch.object(metric, "_augment", side_effect=observed_augment))
        stack.enter_context(mock.patch.object(torch.optim.AdamW, "step", observed_step))
        print("Running uninterrupted four epochs / eight actual CUDA optimizer steps", flush=True)
        full = training("full", "full", resumable=True)

        def interrupt_after_commit(payload, path):
            real_save(payload, path)
            if Path(path).name == "last.pt" and payload["completed_epochs"] == 2:
                raise InterruptedTraining("last.pt epoch 2 fully committed")

        print("Running split prefix; interrupt immediately after committed epoch 2", flush=True)
        interrupted = False
        with mock.patch.object(metric, "_atomic_torch_save", side_effect=interrupt_after_commit):
            try:
                training("split_initial", "split", resumable=True)
            except InterruptedTraining as error:
                interrupted = True
                REPORT["interruption"] = {"after_committed_epoch": 2, "message": str(error)}
        require(interrupted, "Expected post-commit interruption did not occur")
        committed = load("split")
        require(committed["completed_epochs"] == 2 and len(committed["history"]) == 2,
                "Interruption boundary must contain two committed epochs")
        assert_cuda_rng(committed, "split_committed_epoch2")
        saved_history = committed["history"]
        # Preserve the original committed boundary before recovery modifies last.pt.
        real_save(committed, RUN_ROOT / "committed-epoch2.pt")
        # Same recovery challenge as the fixture: partial/ahead log and broken best copy.
        with (RUN_ROOT / "split" / "per_epoch.jsonl").open("a") as stream:
            stream.write('{"epoch": 999}\n{"partial":')
        (RUN_ROOT / "split" / "best.pt").write_bytes(b"interrupted best write")
        print("Resuming original four-epoch budget for two remaining epochs", flush=True)
        resumed = training("split_resume", "split", resume=True)
        torch.cuda.synchronize()

    left, right = load("split"), load("full")
    assert_cuda_rng(left, "split_final_epoch4")
    assert_cuda_rng(right, "full_final_epoch4")
    for key in STATE_KEYS:
        record_exact(key, left[key], right[key])
    record_exact("best_pt_entire_payload", load("split", "best.pt"), load("full", "best.pt"), extra=True)
    record_exact("returned_per_epoch", resumed["per_epoch"], full["per_epoch"], extra=True)
    record_exact("committed_history_prefix_preserved", resumed["per_epoch"][:2], saved_history, extra=True)
    record_exact("selected_epoch", resumed["selected_epoch"], full["selected_epoch"], extra=True)
    record_exact("selected_val_eer", resumed["selected_val_eer"], full["selected_val_eer"], extra=True)
    logs = [json.loads(line) for line in (RUN_ROOT / "split" / "per_epoch.jsonl").read_text().splitlines()]
    record_exact("repaired_log_matches_committed_history", logs, resumed["per_epoch"], extra=True)
    record_exact("actual_optimizer_step_counts", observations["optimizer_steps"],
                 REPORT["run_plan"]["expected_optimizer_steps"], extra=True)
    record_exact("actual_augmentation_counts", observations["augmentation_calls"],
                 REPORT["run_plan"]["expected_optimizer_steps"], extra=True)
    record_exact("actual_encoder_forward_counts", observations["encoder_forwards"],
                 {"full:train": 8, "full:eval": 4, "split_initial:train": 4,
                  "split_initial:eval": 2, "split_resume:train": 4, "split_resume:eval": 2}, extra=True)
    record_exact("actual_classifier_forward_counts", observations["classifier_forwards"],
                 {"full:train": 8, "split_initial:train": 4, "split_resume:train": 4}, extra=True)
    record_exact("completed_epochs", left["completed_epochs"], 4, extra=True)
    record_exact("cosine_total_epoch_budget", left["scheduler_state_dict"]["T_max"], 4, extra=True)
    record_exact("cosine_completed_epochs", left["scheduler_state_dict"]["last_epoch"], 4, extra=True)
    record_exact("nonempty_classifier_state", bool(left["classifier_state_dict"]), True, extra=True)
    record_exact("observational_elapsed_seconds_zero", [row["elapsed_seconds"] for row in left["history"]],
                 [0.0] * 4, extra=True)
    record_exact("all_loaders_synthetic_train", REPORT["synthetic"]["all_loader_splits"], ["train"] * 3, extra=True)
    REPORT["synthetic"]["loader_calls"] = loader_calls
    REPORT["results"] = {"completed_epochs": left["completed_epochs"],
                         "selected_epoch": resumed["selected_epoch"], "selected_val_eer": resumed["selected_val_eer"],
                         "history": resumed["per_epoch"],
                         "comparisons_loaded_with": {"map_location": "cpu", "weights_only": True},
                         "tensor_equality": "dtype/shape/layout and exact contiguous byte equality",
                         "scalar_float_equality": "IEEE-754 double byte equality",
                         "whole_last_pt_compared": False,
                         "whole_last_pt_note": "All listed replay state keys compared exactly. Output-directory provenance intentionally differs between full and split runs."}
    all_checks = {**REPORT["exact_checks"], **REPORT["additional_checks"]}
    failures = [name for name, result in all_checks.items() if not result or not result["exact"]]
    require(not failures, "Exact comparison failures: " + ", ".join(failures))
    REPORT["success"] = True
    REPORT["status"] = "success"
    REPORT["reason"] = "Actual CUDA full versus committed epoch-2 resume matched exactly for every listed replay state and full best.pt payload."
    print(REPORT["reason"], flush=True)


if __name__ == "__main__":
    raw_started = time.perf_counter_ns()
    exit_code = 0
    try:
        main()
    except BaseException as error:
        exit_code = 1
        REPORT["status"] = "failure"
        REPORT["success"] = False
        REPORT["reason"] = f"{type(error).__name__}: {error}"
        REPORT["exception"] = {"type": type(error).__name__, "message": str(error),
                               "traceback": traceback.format_exc()}
        traceback.print_exc()
    finally:
        REPORT["timing"]["raw_total_script_wall_seconds"] = (time.perf_counter_ns() - raw_started) / 1e9
        # Exclusive report creation; an old report is never overwritten.
        with REPORT_PATH.open("x", encoding="utf-8") as stream:
            json.dump(REPORT, stream, indent=2, allow_nan=False)
            stream.write("\n")
        print(json.dumps({"status": REPORT["status"], "success": REPORT["success"],
                          "reason": REPORT["reason"], "report": str(REPORT_PATH)}), flush=True)
    sys.exit(exit_code)
