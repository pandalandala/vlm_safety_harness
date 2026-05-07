---
date: 2026-05-06
topic: docs/ 文档创建 + session routing hooks 更新
---

## 本次完成的工作

### 1. docs/ 目录结构
```
/mnt/hdd/xuran/vlm_safety_harness/docs/
├── A_experiments.md        ← A1-A4 详细设计
├── project_overview.md     ← 整体架构 + 实验流程
└── session_outputs/        ← 本会话回答存档
```

### 2. docs/A_experiments.md
- A1 (P2): 文本捷径 — 黑图替换实验，ΔASR 指标
- A2 (P3): 关系单一 — tool→target 占比分析 + 手工 probe
- A3 (P4): 合成-真实差距 — MIS easy/hard vs. mis_real ASR 对比（**可立即运行**）
- A4 (P7): Counterfactual — MSSBench FPR + Consistency 指标
- 执行顺序: A3 → A1 → A4 → A2

### 3. docs/project_overview.md
- 完整目录结构
- 数据状态表（含 blockers）
- 模型列表（训练 + 推理-only）
- 端到端实验流程（Phase A→B→C→D）
- GPU 分配规则
- 评测协议
- 关键路径

### 4. Hooks 路由更新

**Session marker** (`/tmp/.claude_active_project`):
- 项目 SessionStart.sh 写入项目路径 → 全局 hooks 读取 → 路由到项目目录

**log_command.sh 路由优先级**:
1. `/tmp/.claude_active_project` 存在 → `<project>/logs/`
2. 命令文本含 `vlm_safety_harness` → `vlm_safety_harness/logs/`
3. 其他 → `~/claude_logs/`（全局）

**save_response.py 路由优先级**:
1. `/tmp/.claude_active_project` 存在 → `<project>/docs/session_outputs/`
2. transcript 含项目路径 → `vlm_safety_harness/docs/session_outputs/`
3. 其他 → `~/claude_outputs/`（全局）

### 5. 原则落实
所有项目相关文档/日志/回答存档在 `vlm_safety_harness/` 目录树内。  
`~/claude_outputs/` 保留给跨项目全局输出。
