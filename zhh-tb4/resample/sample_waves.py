#!/usr/bin/env python3
"""Wave sampler for tasks that still have fewer than 10 kept trajectories.

Each wave launches one harbor job: every still-short task gets 8 trials,
concurrency 16. A task leaves the queue only after a wave has fully
finished and its pass count is at least 15. Extra passes above 15 are kept.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path

ROOT = Path("/kwkj-k8s/llm_team/lys/megatron-swift/zhh-tb4")
RESAMPLE = ROOT / "resample"
TB = Path("/kwkj-k8s/llm_team/lys/llm_beachmark-new/terminal-bench")
SUMMARY = ROOT / "traj_qwen_glm_top10_by_turns.summary.json"
RUNS = RESAMPLE / "runs"
PASSES = RESAMPLE / "passes"
PROGRESS = RESAMPLE / "PROGRESS.json"
LOG = RESAMPLE / "progress.log"
KEY_FILE = RESAMPLE / "api.key"

BASE = "https://agrouter-test.agnes-ai.com/v1"
MODEL = "openai/glm-5.3"
TARGET = 15
K = 8
NC = 16
MAX_WAVES = 30
POLL_SEC = 30


def log(msg: str) -> None:
    line = f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {msg}"
    print(line, flush=True)
    with LOG.open("a") as f:
        f.write(line + "\n")


def load_baseline() -> dict[str, int]:
    summary = json.loads(SUMMARY.read_text())
    return {task: int(n) for task, n in summary["tasks_under_10"].items()}


def write_progress(payload: dict) -> None:
    PROGRESS.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def disk_ok() -> bool:
    usage = shutil.disk_usage("/kwkj-k8s")
    free_gb = usage.free / (1024 ** 3)
    return free_gb >= 50


def trial_rows(job_dir: Path) -> list[dict]:
    rows = []
    if not job_dir.is_dir():
        return rows
    for result in sorted(job_dir.glob("*/result.json")):
        trial = result.parent
        task = trial.name.split("__", 1)[0]
        row = {"task": task, "trial": trial.name, "dir": str(trial), "scored": False, "passed": False, "steps": None}
        try:
            data = json.loads(result.read_text())
        except json.JSONDecodeError:
            row["broken"] = True
            rows.append(row)
            continue
        if (data.get("exception_info") or {}).get("exception_type"):
            row["exception"] = data["exception_info"]["exception_type"]
            rows.append(row)
            continue
        reward = ((data.get("verifier_result") or {}).get("rewards") or {}).get("reward")
        if reward is None:
            rows.append(row)
            continue
        row["scored"] = True
        row["passed"] = float(reward) == 1.0
        traj = trial / "agent" / "mini-swe-agent.trajectory.json"
        if traj.is_file():
            try:
                info = json.loads(traj.read_text()).get("info") or {}
                row["steps"] = (info.get("model_stats") or {}).get("api_calls")
            except json.JSONDecodeError:
                pass
        rows.append(row)
    return rows


def snapshot(wave: int, phase: str, todo: list[str], baseline: dict[str, int], job_dir: Path | None) -> None:
    live = trial_rows(job_dir) if job_dir else []
    by_task: dict[str, dict] = {}
    for task in todo:
        mine = [r for r in live if r["task"] == task]
        by_task[task] = {
            "have_before_wave": baseline[task],
            "target": TARGET,
            "wave_trials_done": len(mine),
            "wave_scored": sum(1 for r in mine if r["scored"]),
            "wave_passed": sum(1 for r in mine if r["passed"]),
            "projected": baseline[task] + (0 if phase == "wave_done" else sum(1 for r in mine if r["passed"])),
        }
    still = [t for t, n in baseline.items() if n < TARGET]
    payload = {
        "wave": wave,
        "phase": phase,
        "target": TARGET,
        "trials_per_task": K,
        "concurrency": NC,
        "model": MODEL,
        "base": BASE,
        "tasks_this_wave": todo,
        "tasks_still_short": len(still),
        "still_short": still,
        "wave_trials_done": len(live),
        "wave_trials_expected": len(todo) * K,
        "tasks": by_task,
        "job_dir": str(job_dir) if job_dir else None,
    }
    write_progress(payload)
    bits = [f"{t} {by_task[t]['projected']}/{TARGET} (本波过{by_task[t]['wave_passed']})" for t in todo]
    log(
        f"第{wave}波 {phase} 进度 {len(live)}/{len(todo)*K} 条trial "
        f"还不够15的任务 {len(still)} 个 | " + " ; ".join(bits)
    )


def export_passes(wave: int, job_dir: Path, seen: set[str]) -> int:
    copied = 0
    for row in trial_rows(job_dir):
        if not row["passed"] or row["dir"] in seen:
            continue
        src = Path(row["dir"])
        dst = PASSES / f"wave{wave}" / src.name
        dst.mkdir(parents=True, exist_ok=True)
        for name in ("result.json",):
            if (src / name).is_file():
                shutil.copy2(src / name, dst / name)
        agent = src / "agent"
        if agent.is_dir():
            shutil.copytree(agent, dst / "agent", dirs_exist_ok=True)
        seen.add(row["dir"])
        copied += 1
    return copied


def run_wave(wave: int, todo: list[str], baseline: dict[str, int]) -> Path:
    job = f"glm53_resample_wave{wave}"
    job_dir = RUNS / job
    if job_dir.exists():
        raise SystemExit(f"job dir already exists: {job_dir}")
    while not disk_ok():
        log("共享盘剩余不足 50G，10 分钟后再看")
        time.sleep(600)
    key = KEY_FILE.read_text().strip()
    cmd = [
        str(TB / ".venv/bin/harbor"), "run", "-p", "tasks",
        "-a", "mini-swe-agent", "-m", MODEL,
        "--ak", "version=2.4.6",
        "--ak", f"config_file={TB / '_setup/aa_mini_config_default.yaml'}",
        "--ae", f"OPENAI_BASE_URL={BASE}",
        "--ae", f"OPENAI_API_KEY={key}",
        "-k", str(K), "-n", str(NC),
        "--environment-build-timeout-multiplier", "4",
        "--agent-setup-timeout-multiplier", "4",
        "--agent-timeout-multiplier", "1.0",
        "--max-retries", "2",
        "--retry-include", "AgentSetupTimeoutError",
        "--retry-include", "NetworkConnectionError",
        "--retry-include", "NonZeroAgentExitCodeError",
        "-o", str(RUNS), "--job-name", job, "-y",
    ]
    for task in todo:
        cmd.extend(["-i", task])
    env = os.environ.copy()
    env["PATH"] = f"{TB / '.venv/bin'}:{env.get('PATH', '')}"
    env["UV_CACHE_DIR"] = str(TB / ".uv-cache")
    env["HARBOR_ALLOW_INSECURE_MODEL_BASE_URL"] = "1"
    env["NO_PROXY"] = "agrouter-test.agnes-ai.com,192.168.1.19,192.168.1.20,192.168.1.22,localhost,127.0.0.1"
    env["no_proxy"] = env["NO_PROXY"]
    log(f"第{wave}波开始 任务{len(todo)}个 每题{K}次 并发{NC} job={job}")
    snapshot(wave, "running", todo, baseline, job_dir)
    proc = subprocess.Popen(cmd, cwd=TB, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    harbor_log = RESAMPLE / f"harbor_wave{wave}.log"

    def _pump() -> None:
        assert proc.stdout is not None
        with harbor_log.open("w") as handle:
            for line in proc.stdout:
                handle.write(line)
                handle.flush()

    threading.Thread(target=_pump, daemon=True).start()
    while proc.poll() is None:
        snapshot(wave, "running", todo, baseline, job_dir)
        time.sleep(POLL_SEC)
    if proc.returncode != 0:
        log(f"第{wave}波 harbor 退出码 {proc.returncode}，日志 {harbor_log}")
    return job_dir


def main() -> None:
    RESAMPLE.mkdir(parents=True, exist_ok=True)
    RUNS.mkdir(parents=True, exist_ok=True)
    PASSES.mkdir(parents=True, exist_ok=True)
    os.chmod(KEY_FILE, 0o600)
    baseline = load_baseline()
    seen: set[str] = set()
    log(f"补采样启动 不够10的任务 {len(baseline)} 个，目标每题 {TARGET} 条通过")
    for wave in range(1, MAX_WAVES + 1):
        todo = sorted(t for t, n in baseline.items() if n < TARGET)
        if not todo:
            snapshot(wave - 1, "done", [], baseline, None)
            log("全部任务已达到 15，停止")
            return
        job_dir = run_wave(wave, todo, baseline)
        new_pass = 0
        for row in trial_rows(job_dir):
            if row["passed"]:
                baseline[row["task"]] = baseline.get(row["task"], 0) + 1
                new_pass += 1
        copied = export_passes(wave, job_dir, seen)
        snapshot(wave, "wave_done", todo, baseline, job_dir)
        short = [t for t in todo if baseline[t] < TARGET]
        log(f"第{wave}波结束 新通过 {new_pass} 条，已复制轨迹 {copied} 份，还不够15的 {len(short)} 个: " +
            ", ".join(f"{t}={baseline[t]}" for t in short))
        (RESAMPLE / "counts.json").write_text(json.dumps(baseline, ensure_ascii=False, indent=2) + "\n")
    log(f"达到最大波数 {MAX_WAVES}，仍有任务不足 {TARGET}")


if __name__ == "__main__":
    main()
