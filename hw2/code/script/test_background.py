"""Exercise detached launch/locking/hangup/status with a disposable dummy job, not training."""
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from common import ROOT


class BackgroundTest(unittest.TestCase):
    def test_detachment_lock_hangup_and_exit_receipt(self):
        with tempfile.TemporaryDirectory(prefix="hw2-background-test-") as directory:
            root = Path(directory)
            (root / "scripts").mkdir()
            for filename in ["background_run.py", "common.py"]:
                shutil.copy2(ROOT / "code" / "script" / filename, root / "scripts" / filename)
            (root / "configs/v3").mkdir(parents=True)
            (root / "configs/v3/downloads.json").write_text('{"models": []}')
            (root / "data/train").mkdir(parents=True)
            for filename in ["dialogues.jsonl", "products.jsonl"]:
                (root / "data/train" / filename).write_text("")
            (root / "run_v3.sh").write_text(
                '#!/bin/bash\npython -u -c "import time; print(\'DUMMY ONLY\'); time.sleep(2); print(\'DONE\')"\n')
            command = [sys.executable, str(root / "code/script/background_run.py"), "--version", "v3"]
            first = subprocess.run(command, capture_output=True, text=True, check=True, timeout=10)
            self.assertIn("Background job started", first.stdout)
            receipt = root / "runs/launcher/state.json"
            deadline = time.monotonic() + 10
            while not receipt.exists() and time.monotonic() < deadline:
                time.sleep(0.05)
            state = json.loads(receipt.read_text())
            pid = state["pid"]
            self.assertEqual(os.getsid(pid), pid)
            second = subprocess.run(command, capture_output=True, text=True, check=True, timeout=10)
            self.assertIn("Already running", second.stdout)
            # The first launcher process has already exited. Hang up only the
            # isolated dummy session, never the user's terminal/process group.
            os.killpg(pid, signal.SIGHUP)
            while state["status"] == "running" and time.monotonic() < deadline:
                time.sleep(0.05)
                state = json.loads(receipt.read_text())
            self.assertEqual(state["status"], "succeeded")
            self.assertEqual(state["exit_code"], 0)
            log = (root / "logs/train_latest.log").read_text()
            self.assertIn("DUMMY ONLY", log)
            self.assertIn("SUCCESS exit_code=0", log)
            (root / "run_v3.sh").write_text("#!/bin/bash\nexit 7\n")
            failed = subprocess.run(command, capture_output=True, text=True, check=True, timeout=10)
            self.assertIn("Background job started", failed.stdout)
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                state = json.loads(receipt.read_text())
                if state["status"] == "failed":
                    break
                time.sleep(0.05)
            self.assertEqual(state["exit_code"], 7)
            status = subprocess.run(command + ["--status"], capture_output=True, text=True, check=True)
            self.assertIn("NOT RUNNING", status.stdout)


if __name__ == "__main__":
    unittest.main(verbosity=2)
