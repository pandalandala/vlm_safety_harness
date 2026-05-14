# Session Logging Plan

## 目标

让 `scripts/` 下每次运行都自动把**全部终端输出**（stdout + stderr，包括所有 `print()` 和 tracebacks）同步到一个带时间戳的 log 文件，并在终端末尾显示 log 路径。

## 现状

`harness/utils/logger.py` 已有基于 `logging` 模块的结构化日志，但只捕获 `log.info()` / `log.error()` 调用；所有脚本里的 `print("[train] Done → ...")` 等输出不会落盘。

## 设计

### log 目录结构

```
logs/
├── main/
│   ├── 20260514_143022_run_experiment_main_dreams_internvl3_5.log
│   └── 20260514_160000_run_main_E1.log
├── prelim/
│   └── 20260514_090000_run_prelim_A1_A2_A3_A4.log
├── capability/
│   └── 20260514_100000_run_capability_V0.log
├── eval/
│   └── 20260514_120000_run_eval_only.log
└── commands_2026-05-14.log    ← 每日滚动摘要
```

文件名格式：`{YYYYMMDD_HHMMSS}_{script_name}_{tag}.log`
`tag` = 实验 YAML / `--experiment-id` / `--models` 首个参数，长度截断到 40 字符。

### 核心机制：TeeSession

在 `harness/utils/logger.py` 增加一个 `TeeSession` 类，用 Python 的 `io.TextIOWrapper` 替换 `sys.stdout` / `sys.stderr`，写入同时转发到原始 fd。

```python
class TeeSession:
    """Tee stdout + stderr to a log file for the lifetime of a script run."""

    def start(self, script_name, tag="", category="main") -> Path: ...
    def stop(self) -> None: ...   # 恢复 sys.stdout/stderr，打印 log 路径，flush
    def __enter__(self): ...
    def __exit__(self): ...
```

脚本只需在 `main()` 开头加一行：

```python
from harness.utils.logger import init_session
log_path = init_session(script_name="run_experiment", tag=args.config, category="main")
```

`init_session()` 返回 log 文件路径；`atexit` 注册 `session.stop()`，保证正常退出和异常退出（SIGINT/SIGTERM）都能 flush + 打印路径。

### 终端末尾输出格式

```
═══════════════════════════════════════════════════════
 Session log saved → logs/main/20260514_143022_run_experiment_main_dreams_internvl3_5.log
═══════════════════════════════════════════════════════
```

## 涉及改动

### 新增 / 修改文件

| 文件 | 改动 |
|------|------|
| `harness/utils/logger.py` | 新增 `TeeSession`, `init_session()` |
| `scripts/run_experiment.py` | `main()` 开头加 `init_session` |
| `scripts/run_main.py` | 同上 |
| `scripts/run_prelim.py` | 同上 |
| `scripts/run_capability.py` | 同上 |
| `scripts/run_closed_source.py` | 同上 |
| `scripts/run_eval_only.py` | 同上 |
| `scripts/run_inference_only.py` | 同上 |
| `scripts/generate_report.py` | 同上 |

每个 script 改动：**1 行 import + 1 行 `init_session()`**，其余不动。

### 不改动

- 已有的 `get_logger()` 接口不变，保持向后兼容
- `logs/` 目录已在 `.gitignore` 中（或加入）

## 关键实现细节

- `TeeSession.start()` 用 `os.dup2` 级别复制 fd 而非替换 Python 层对象，确保 LlamaFactory subprocess（`torchrun` 等）的输出也能被捕获到同一 log。
  - 具体：`os.dup(1)` 保存原 stdout fd → 打开 log 文件 fd → `os.dup2(log_fd, 1)` 和 `os.dup2(log_fd, 2)` → Python 层用 `tee`-style wrapper 同时写到两个 fd。
  - 子进程输出默认继承 fd 1/2，不需额外处理。
- `atexit.register(session.stop)` 注册一次，脚本正常退出时自动打印路径。
- `SIGTERM` handler 同样调用 `session.stop()`。
- 日志文件使用 `utf-8` 编码，带 BOM 或无均可。

## 验证步骤

```bash
# 1. 单元测试
python -c "
from harness.utils.logger import init_session
init_session('test_script', tag='smoke', category='main')
print('hello stdout')
import sys; print('hello stderr', file=sys.stderr)
"
# 预期：终端输出两行 + 末尾 log 路径；logs/main/XXXXXX_test_script_smoke.log 内容一致

# 2. dry-run 集成测试
python scripts/run_experiment.py main/main_dreams_internvl3_5.yaml --dry-run --limit 2
# 预期：logs/main/XXXXXX_run_experiment_*.log 存在，内容完整

# 3. 错误捕获测试
python scripts/run_experiment.py not_exist.yaml 2>&1 | head -5
# 预期：traceback 也出现在 log 文件中
```
