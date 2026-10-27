"""Detached v3 launcher: project-local logs, inherited flock and honest exit status."""
import argparse
from datetime import datetime
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from common import ROOT, read_json, write_json

STATE = ROOT / "runs/launcher/state.json"
LOCK = ROOT / "runs/launcher/run.lock"


def environment():
    env = dict(os.environ)
    env.update(HF_HOME=str(ROOT / ".cache/huggingface"), TOKENIZERS_PARALLELISM="false",
               PYTHONUNBUFFERED="1", HW2_DEVICE="cuda:0")
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
    return env


def prerequisites(version="v4"):
    config = read_json(ROOT / "configs/v3/downloads.json")
    for entry in config["models"]:
        folder = ROOT / entry["destination"]
        if not (folder / "config.json").is_file() or not list(folder.glob("*.safetensors")):
            raise SystemExit(f"Model missing/incomplete: {folder}. Run bash download_v3.sh first.")
    for name in ("data/train/dialogues.jsonl", "data/train/products.jsonl"):
        if not (ROOT / name).is_file():
            raise SystemExit(f"Official training data missing: {name}")
    if version == "v4" and not (ROOT / "models/e5-base-v2/model.safetensors").exists():
        raise SystemExit("Missing E5 weights; run bash download_v4.sh first")


def busy():
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    with LOCK.open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
    return False


def status():
    active = busy()
    state = read_json(STATE) if STATE.exists() else {}
    print("RUNNING (lock held)" if active else "NOT RUNNING")
    if state:
        import json
        print(json.dumps(state, indent=2, ensure_ascii=False))
        if not active and state.get("status") == "running":
            print("Previous worker disappeared without a completion receipt; this is NOT success.")


def worker(fd, log_path, public=False, version="v4"):
    # No controlling terminal; descendants also ignore terminal hangups.
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    os.fstat(fd)  # Fail if launched without the inherited lock descriptor.
    command = ([sys.executable, "-u", str(ROOT / "code/script/public_evaluation.py"), "--version", version] if public
               else ["bash", str(ROOT / f"run_{version}.sh")])
    if STATE.exists():
        previous = read_json(STATE)
        if previous.get("finished_at"):
            write_json(STATE.parent / "history" / f"job_{previous['pid']}.json", previous)
    state = {"status": "running", "pid": os.getpid(), "log": log_path,
             "started_at": datetime.now().astimezone().isoformat(), "command": " ".join(command),
             "task": "public evaluation" if public else f"{version} train-only experiments"}
    write_json(STATE, state)
    print(f"START {state['started_at']} pid={os.getpid()}", flush=True)
    code = 1
    try:
        # Keep the lock in the shell as well: even if this wrapper disappears,
        # a still-running training shell prevents a duplicate launch.
        code = subprocess.call(command, cwd=ROOT,
                               env=environment(), pass_fds=(fd,))
    finally:
        state.update(status="succeeded" if code == 0 else "failed", exit_code=code,
                     finished_at=datetime.now().astimezone().isoformat())
        write_json(STATE, state)
        print(f"{'SUCCESS' if code == 0 else 'FAILED'} exit_code={code} {state['finished_at']}", flush=True)
    return code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--status", action="store_true")
    mode.add_argument("--foreground", action="store_true", help="Wait for training, e.g. for reproduction")
    parser.add_argument("--worker-fd", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--log", help=argparse.SUPPRESS)
    parser.add_argument("--public", action="store_true", help="Evaluate the frozen selection instead of training")
    parser.add_argument("--version", choices=["v3", "v4"], default="v4")
    args = parser.parse_args()
    os.chdir(ROOT)
    if args.worker_fd is not None:
        return worker(args.worker_fd, args.log, args.public, args.version)
    if args.status:
        status()
        return 0
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    with LOCK.open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("Already running; no second job launched. Use bash run_background.sh --status for the active job and log.")
            return 0
        prerequisites(args.version)
        if args.public:
            for name in (f"configs/{args.version}/selected.json", f"results/{args.version}/selection.json",
                         "data/public/dialogues.jsonl", "data/public/products.jsonl"):
                if not (ROOT / name).is_file():
                    raise SystemExit(f"Missing public evaluation prerequisite: {name}")
            if (ROOT / f"results/{args.version}/evaluation.json").exists():
                raise SystemExit("Public evaluation already exists; refusing to overwrite results.")
        if args.foreground:
            return worker(handle.fileno(), "foreground terminal", args.public, args.version)
        logs = ROOT / "logs"
        logs.mkdir(exist_ok=True)
        prefix = "public" if args.public else "train"
        log = logs / f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{time.time_ns()}.log"
        with log.open("x") as output:
            child = subprocess.Popen([sys.executable, "-u", str(Path(__file__).resolve()),
                                      "--worker-fd", str(handle.fileno()), "--log", str(log), "--version", args.version] +
                                     (["--public"] if args.public else []),
                                     cwd=ROOT, env=environment(), stdin=subprocess.DEVNULL,
                                     stdout=output, stderr=subprocess.STDOUT, start_new_session=True,
                                     pass_fds=(handle.fileno(),))
        latest = logs / f"{prefix}_latest.log"
        # Replace only this launcher-owned symlink, never an existing regular log.
        if latest.is_symlink():
            latest.unlink()
        if not latest.exists():
            latest.symlink_to(log.name)
        print(f"Background job started: PID={child.pid}\nLog: {log}")
        print(f"View: tail -n 50 -F {log}\nShortcut: tail -n 50 -F logs/{prefix}_latest.log")
        print("Status: bash run_background.sh --status\nClosing this terminal is safe; shutdown/suspend stops progress.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
