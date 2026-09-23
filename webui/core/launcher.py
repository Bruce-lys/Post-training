"""Turn UI state into run files and a detached ``megatron sft`` process; abort and finalize runs."""
from __future__ import annotations

import json
import os
import pathlib
import re
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any

from . import command as cmd, monitor, param_schema as ps, registry
from .constants import MASTER_PORT_BASE, MASTER_PORT_RANGE, PROJECT_ROOT, WEBUI_ROOT, hostname
from .settings import UiSettings


@dataclass
class Check:
    name: str
    level: str      # ok | warn | error
    message: str


@dataclass
class LaunchPlan:
    run_id: str
    name: str
    run_root: str
    model_path: str
    dataset: str
    state: dict
    argv: list[str]
    env: dict[str, str]
    errors: list[str]
    warnings: list[str]
    fake: bool
    train_venv: str
    fake_cmd: list[str] = field(default_factory=list)

    @property
    def command(self) -> list[str]:
        return self.fake_cmd if self.fake else self.argv


def slug(name: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_.\-]+", "-", (name or "").strip()).strip("-.")
    return s[:48] or "run"


def new_run_id(name: str) -> str:
    return f"{slug(name)}-{time.strftime('%Y%m%dT%H%M%S')}"


def alloc_master_port() -> int:
    """First free torchrun rendezvous port; also avoids ports used by runs registered on this host."""
    taken = {int(d.get("master_port", 0)) for d in registry.active_runs(hostname())}
    for port in range(MASTER_PORT_BASE, MASTER_PORT_BASE + MASTER_PORT_RANGE):
        if port in taken or monitor.port_in_use(port):
            continue
        return port
    return MASTER_PORT_BASE


def fake_command(run_root: str, state: dict, ui: UiSettings) -> list[str]:
    s = ps.compute_derived(state)
    iters = int(s.get("train_iters") or 0) or max(1, int(s.get("num_train_epochs") or 1)) * 10
    return [str(WEBUI_ROOT / ".venv" / "bin" / "python"), str(WEBUI_ROOT / "scripts" / "fake_train.py"),
            "--out", run_root, "--iters", str(iters), "--interval", str(ui.fake_interval), "--run-id", pathlib.Path(run_root).name]


def plan(state: dict, name: str, model_path: str, dataset: str, paths: dict[str, str], ui: UiSettings,
         run_id: str | None = None) -> LaunchPlan:
    run_id = run_id or new_run_id(name)
    output_root = (paths.get("OUTPUT_ROOT") or str(PROJECT_ROOT / "outputs")).rstrip("/")
    run_root = f"{output_root}/{run_id}"
    train_venv = (paths.get("TRAIN_VENV") or str(PROJECT_ROOT / ".venv-swift")).rstrip("/")
    s = ps.compute_derived(state)
    errors, warnings = ps.validate_state(s)
    if not (model_path or "").strip():
        errors.append("模型路径为空")
    if not (dataset or "").strip():
        errors.append("数据集为空")
    argv = cmd.build_argv(s, model_path.strip(), dataset.strip(), run_root, train_venv)
    env = cmd.build_env(s, train_venv, alloc_master_port(), paths.get("MODELSCOPE_CACHE", ""), paths.get("HF_HOME", ""))
    fake = bool(ui.fake_train)
    return LaunchPlan(run_id=run_id, name=name, run_root=run_root, model_path=model_path.strip(), dataset=dataset.strip(),
                      state=s, argv=argv, env=env, errors=errors, warnings=warnings, fake=fake, train_venv=train_venv,
                      fake_cmd=fake_command(run_root, s, ui) if fake else [])


# --------------------------------------------------------------------------- preflight
def preflight_host(p: LaunchPlan) -> list[Check]:
    checks: list[Check] = []
    for e in p.errors:
        checks.append(Check("参数", "error", e))
    for w in p.warnings:
        checks.append(Check("参数", "warn", w))
    checks.append(Check("输出目录未占用", "error" if pathlib.Path(p.run_root).exists() else "ok", p.run_root))
    mp = pathlib.Path(p.model_path)
    checks.append(Check("模型路径", "ok" if mp.is_dir() else "error", p.model_path if mp.is_dir() else f"目录不存在: {p.model_path}"))
    ds = p.dataset
    if "/" in ds or ds.endswith(".jsonl") or ds.endswith(".json"):
        ok = pathlib.Path(ds).is_file()
        checks.append(Check("数据集文件", "ok" if ok else "error", ds if ok else f"文件不存在: {ds}"))
    else:
        checks.append(Check("数据集", "warn", f"{ds} 按 ModelScope/HF 数据集 ID 处理（需要联网）"))
    if p.fake:
        checks.append(Check("假训练模式", "warn", "不会启动 megatron，只写模拟指标"))
        return checks
    mb = pathlib.Path(cmd.megatron_bin(p.train_venv))
    checks.append(Check("训练环境", "ok" if mb.is_file() else "error", str(mb) if mb.is_file() else f"缺少 {mb}"))
    gpus = ps.gpu_list(p.state)
    mem = monitor.gpu_memory_used_mib()
    if mem:
        bad = [g for g in gpus if not g.isdigit() or int(g) >= len(mem)]
        if bad:
            checks.append(Check("GPU 编号", "error", f"本机只看到 {len(mem)} 张卡，无效编号: {bad}"))
        used = {g: mem[int(g)] for g in gpus if g.isdigit() and int(g) < len(mem)}
        busy = [f"{g}({m // 1024}GB)" for g, m in used.items() if m > 20 * 1024]
        hot = [f"{g}({m // 1024}GB)" for g, m in used.items() if 2048 < m <= 20 * 1024]
        if busy:
            checks.append(Check("GPU 显存", "error", f"GPU {busy} 已被别的任务占用，装不下本模型；换空闲卡或等待（UI 不会杀任何进程）"))
        elif hot:
            checks.append(Check("GPU 显存", "warn", f"GPU {hot} 已有少量占用（可能有别的进程；UI 不会杀任何进程）"))
        else:
            checks.append(Check("GPU 显存", "ok", "所选 GPU 全部空闲"))
    else:
        checks.append(Check("GPU", "error", "nvidia-smi 不可用"))
    busy = registry.active_runs(hostname())
    if busy:
        used = set()
        for d in busy:
            used.update(ps.gpu_list(d.get("state", {})))
        overlap = sorted(set(gpus) & used)
        checks.append(Check("本机其他训练", "error" if overlap else "warn",
                            f"GPU {overlap} 正被 {', '.join(d['run_id'] for d in busy)} 使用" if overlap
                            else f"本机有活动 run: {', '.join(d['run_id'] for d in busy)}（GPU 不重叠）"))
    return checks


# --------------------------------------------------------------------------- files
def write_run_files(p: LaunchPlan) -> dict[str, str]:
    root = pathlib.Path(p.run_root)
    (root / "ui").mkdir(parents=True, exist_ok=True)
    for k in ("MODELSCOPE_CACHE", "HF_HOME"):
        pathlib.Path(p.env[k]).mkdir(parents=True, exist_ok=True)
    script = root / "ui" / "run.sh"
    script.write_text(cmd.render_script(p.env, p.command, p.run_root), encoding="utf-8")
    script.chmod(0o755)
    cfg = {"schema_version": 1, "run_id": p.run_id, "name": p.name, "hostname": hostname(), "timestamp": monitor.now_iso(),
           "model": p.model_path, "dataset": p.dataset, "state": p.state, "argv": p.argv,
           "argv_sha256": cmd.argv_sha256(p.argv), "env": p.env, "fake_train": p.fake}
    (root / "ui" / "train_config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return {"script": str(script), "config": str(root / "ui" / "train_config.json")}


def preview_text(p: LaunchPlan) -> str:
    lines = [f"# run_id : {p.run_id}", f"# 输出   : {p.run_root}", f"# 主机   : {hostname()}",
             f"# 模式   : {'假训练（不占 GPU）' if p.fake else 'GPU 训练'}", ""]
    for k, v in p.env.items():
        lines.append(f"export {k}={v}")
    lines.append("")
    lines.append(cmd.render_shell(p.command))
    return "\n".join(lines)


# --------------------------------------------------------------------------- launch / abort / finalize
def launch(p: LaunchPlan) -> tuple[bool, str]:
    checks = preflight_host(p)
    errors = [c for c in checks if c.level == "error"]
    if errors:
        return False, "启动前检查未通过:\n" + "\n".join(f"- {c.name}: {c.message}" for c in errors)
    try:
        files = write_run_files(p)
        log = open(monitor.train_log_path(p.run_root), "ab")
        env = dict(os.environ)
        env.pop("VIRTUAL_ENV", None)
        proc = subprocess.Popen(["bash", files["script"]], cwd=str(PROJECT_ROOT), env=env, stdout=log, stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL, start_new_session=True)
        log.close()
        (pathlib.Path(p.run_root) / "ui" / "pid").write_text(str(proc.pid))
    except Exception as exc:  # noqa: BLE001 - surface everything to the UI
        registry.register(_doc(p, status="failed", note=str(exc)))
        return False, f"启动失败: {exc}"
    time.sleep(1.5)
    if proc.poll() is not None and proc.returncode != 0:
        registry.register(_doc(p, status="failed", pid=proc.pid, exit_code=proc.returncode))
        return False, f"进程立即退出 rc={proc.returncode}，见 {monitor.train_log_path(p.run_root)}"
    registry.register(_doc(p, status="running", pid=proc.pid))
    warns = [c for c in checks if c.level == "warn"]
    msg = [f"已启动 pid {proc.pid}（进程组独立，关掉 UI 不影响训练）", f"输出目录: {p.run_root}",
           f"日志: {monitor.train_log_path(p.run_root)}"]
    msg += [f"提示: {c.message}" for c in warns]
    return True, "\n".join(msg)


def _doc(p: LaunchPlan, **extra: Any) -> dict:
    s = p.state
    return {"run_id": p.run_id, "name": p.name, "hostname": hostname(), "run_root": p.run_root,
            "model": p.model_path, "dataset": p.dataset, "tuner_type": s.get("tuner_type"),
            "gpus": s.get("cuda_visible_devices"), "master_port": int(p.env.get("MASTER_PORT", 0)),
            "fake_train": p.fake, "state": s, "argv_sha256": cmd.argv_sha256(p.argv), **extra}


def abort(run_id: str) -> str:
    doc = registry.get(run_id)
    if doc is None:
        return f"未知 run: {run_id}"
    if doc.get("hostname") != hostname():
        return f"该 run 属于 {doc.get('hostname')}，只能在那台机器的 UI 上中止"
    run_root = doc.get("run_root", "")
    pid = monitor.read_pid(run_root)
    msgs = []
    if pid and monitor.pid_alive(pid, run_root):
        msgs.append(monitor.kill_group(pid))
    else:
        msgs.append("主进程已不在")
    stray = monitor.find_procs(f"--output_dir {run_root}") + monitor.find_procs(f"--out {run_root}")
    stray = [x for x in stray if x != os.getpid()]
    if stray:
        for x in stray:
            try:
                os.kill(x, 9)
            except ProcessLookupError:
                pass
        msgs.append(f"清理残留进程 {stray}")
    registry.update(run_id, status="aborted", finished_at=monitor.now_iso(), exit_code=monitor.exit_code(run_root))
    return "\n".join(msgs)


def finalize_if_exited(doc: dict) -> dict | None:
    """When the process has exited on this host, record the final status in the registry."""
    if doc.get("hostname") != hostname() or doc.get("status") not in ("launching", "running"):
        return None
    run_root = doc.get("run_root", "")
    pid = monitor.read_pid(run_root)
    if monitor.pid_alive(pid, run_root):
        return None
    code = monitor.exit_code(run_root)
    if code is None:
        status = "failed"
    elif code == 0:
        status = "finished"
    else:
        status = "failed"
    return registry.update(doc["run_id"], status=status, exit_code=code, finished_at=monitor.now_iso(),
                           last_checkpoint=monitor.last_checkpoint(run_root))


def refresh_all() -> list[dict]:
    for d in registry.list_runs():
        if d.get("status") in ("launching", "running") and d.get("hostname") == hostname():
            finalize_if_exited(d)
    return registry.list_runs()
