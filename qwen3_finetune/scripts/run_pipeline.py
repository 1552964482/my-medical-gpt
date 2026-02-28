#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配置驱动的训练/评测流水线入口。

用法示例:
  python3 scripts/run_pipeline.py --config configs/pipeline_quick.json
  python3 scripts/run_pipeline.py --config configs/pipeline_full.json --from-stage sft
  python3 scripts/run_pipeline.py --config configs/pipeline_full.json --dry-run
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="配置驱动Pipeline执行器")
    parser.add_argument("--config", required=True, help="Pipeline配置JSON路径")
    parser.add_argument("--dry-run", action="store_true", help="仅打印执行计划，不实际执行命令")
    parser.add_argument("--from-stage", default="", help="从某个阶段开始执行（包含该阶段）")
    parser.add_argument("--to-stage", default="", help="执行到某个阶段结束（包含该阶段）")
    parser.add_argument(
        "--only-stages",
        default="",
        help="仅执行指定阶段，逗号分隔，例如: pretrain,sft,eval",
    )
    parser.add_argument(
        "--var",
        action="append",
        default=[],
        help="覆盖变量，格式 KEY=VALUE，可重复传入",
    )
    return parser.parse_args()


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("配置文件根节点必须是JSON对象")
    return data


def format_seconds(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    remain = int(seconds % 60)
    return f"{minutes}m{remain}s"


def parse_cli_vars(items: List[str]) -> Dict[str, str]:
    result: Dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"--var 参数格式错误（需要 KEY=VALUE）: {item}")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"--var 参数KEY为空: {item}")
        result[key] = value
    return result


def render_string(template: str, vars_map: Dict[str, str]) -> str:
    out = template
    for key, value in vars_map.items():
        out = out.replace("${" + key + "}", str(value))
    return out


def resolve_vars(raw_vars: Dict[str, Any], base_vars: Dict[str, str], max_rounds: int = 5) -> Dict[str, str]:
    resolved = {str(k): str(v) for k, v in raw_vars.items()}
    for _ in range(max_rounds):
        changed = False
        context = {**base_vars, **resolved}
        next_vars: Dict[str, str] = {}
        for key, value in resolved.items():
            new_value = render_string(value, context)
            next_vars[key] = new_value
            if new_value != value:
                changed = True
        resolved = next_vars
        if not changed:
            break
    return resolved


def render_obj(obj: Any, vars_map: Dict[str, str]) -> Any:
    if isinstance(obj, str):
        return render_string(obj, vars_map)
    if isinstance(obj, list):
        return [render_obj(x, vars_map) for x in obj]
    if isinstance(obj, dict):
        return {k: render_obj(v, vars_map) for k, v in obj.items()}
    return obj


def should_run_stage(
    stage_name: str,
    ordered_names: List[str],
    from_stage: str,
    to_stage: str,
    only_stages: List[str],
) -> bool:
    if only_stages:
        return stage_name in only_stages

    idx = ordered_names.index(stage_name)
    start_idx = ordered_names.index(from_stage) if from_stage else 0
    end_idx = ordered_names.index(to_stage) if to_stage else len(ordered_names) - 1
    return start_idx <= idx <= end_idx


def run_command(command: str, cwd: Path, env: Dict[str, str], log_path: Path) -> int:
    with log_path.open("a", encoding="utf-8") as log_file:
        header = f"\n$ {command}\n"
        log_file.write(header)
        log_file.flush()
        print(header.rstrip())

        process = subprocess.Popen(
            command,
            cwd=str(cwd),
            env=env,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_file.write(line)
        process.wait()
        return process.returncode


def main() -> int:
    args = parse_args()
    config_path = Path(args.config).expanduser().resolve()
    config = load_json(config_path)

    qwen3_root = config_path.parent.parent.resolve()
    project_root = qwen3_root.parent.resolve()
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    vars_map: Dict[str, str] = {
        "project_root": str(project_root),
        "qwen3_root": str(qwen3_root),
        "config_path": str(config_path),
        "timestamp": timestamp,
    }
    file_vars = resolve_vars(config.get("vars", {}), vars_map)
    vars_map.update(file_vars)
    cli_vars = resolve_vars(parse_cli_vars(args.var), vars_map)
    vars_map.update(cli_vars)

    rendered = render_obj(config, vars_map)
    stages = rendered.get("stages", [])
    if not isinstance(stages, list) or not stages:
        raise ValueError("配置中 stages 不能为空，且必须是数组")

    stage_names = []
    for stage in stages:
        if not isinstance(stage, dict) or "name" not in stage:
            raise ValueError("每个stage必须是对象并包含 name 字段")
        stage_names.append(stage["name"])

    if args.from_stage and args.from_stage not in stage_names:
        raise ValueError(f"--from-stage 不存在: {args.from_stage}")
    if args.to_stage and args.to_stage not in stage_names:
        raise ValueError(f"--to-stage 不存在: {args.to_stage}")
    if args.from_stage and args.to_stage:
        if stage_names.index(args.from_stage) > stage_names.index(args.to_stage):
            raise ValueError("--from-stage 不能晚于 --to-stage")

    only_stages = [x.strip() for x in args.only_stages.split(",") if x.strip()]
    for stage in only_stages:
        if stage not in stage_names:
            raise ValueError(f"--only-stages 中包含未知阶段: {stage}")

    logs_dir = Path(rendered.get("logs_dir", "${qwen3_root}/logs/pipeline"))
    logs_dir = Path(render_string(str(logs_dir), vars_map)).expanduser().resolve()
    logs_dir.mkdir(parents=True, exist_ok=True)

    pipeline_name = rendered.get("name", config_path.stem)
    print(f"Pipeline: {pipeline_name}")
    print(f"Config:   {config_path}")
    print(f"Logs:     {logs_dir}")
    print("-" * 80)

    global_env = os.environ.copy()
    global_env.update({str(k): str(v) for k, v in rendered.get("env", {}).items()})
    default_workdir = Path(rendered.get("workdir", str(qwen3_root))).expanduser().resolve()
    fail_fast = bool(rendered.get("fail_fast", True))

    selected: List[Dict[str, Any]] = []
    for stage in stages:
        stage_enabled = stage.get("enabled", True)
        stage_name = stage["name"]
        if not stage_enabled and stage_name not in only_stages:
            continue
        if should_run_stage(stage_name, stage_names, args.from_stage, args.to_stage, only_stages):
            selected.append(stage)

    if not selected:
        print("没有匹配到可执行阶段。")
        return 0

    print("执行阶段:")
    for stage in selected:
        print(f"- {stage['name']}")
    print("-" * 80)

    if args.dry_run:
        for stage in selected:
            commands = stage.get("commands", [])
            print(f"[DRY-RUN] Stage: {stage['name']}")
            for command in commands:
                print(f"  {command}")
        return 0

    total_start = time.time()
    for stage in selected:
        stage_name = stage["name"]
        stage_start = time.time()
        stage_log = logs_dir / f"{timestamp}_{stage_name}.log"
        stage_env = global_env.copy()
        stage_env.update({str(k): str(v) for k, v in stage.get("env", {}).items()})
        stage_workdir = Path(stage.get("workdir", str(default_workdir))).expanduser().resolve()
        commands = stage.get("commands", [])

        if not commands:
            print(f"[SKIP] {stage_name}: commands 为空")
            continue

        print(f"\n[START] {stage_name}")
        print(f"cwd: {stage_workdir}")
        print(f"log: {stage_log}")

        stage_failed = False
        for command in commands:
            code = run_command(str(command), cwd=stage_workdir, env=stage_env, log_path=stage_log)
            if code != 0:
                stage_failed = True
                print(f"[FAIL] {stage_name}: 命令退出码 {code}")
                break

        stage_cost = time.time() - stage_start
        if stage_failed:
            print(f"[END ] {stage_name} 失败，用时 {format_seconds(stage_cost)}")
            if fail_fast and stage.get("continue_on_error", False) is not True:
                print("Fail-fast 已启用，停止执行后续阶段。")
                return 1
        else:
            print(f"[END ] {stage_name} 成功，用时 {format_seconds(stage_cost)}")

    total_cost = time.time() - total_start
    print("\n" + "-" * 80)
    print(f"Pipeline完成，用时 {format_seconds(total_cost)}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[ERROR] {exc}")
        raise SystemExit(2)
