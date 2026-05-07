# /gen-table — Generate Paper Tables

Aggregate all experiment results and generate LaTeX/Markdown tables for the paper.

## Usage

```
/gen-table [--group GROUP] [--format FORMAT] [--output FILE]
```

## Arguments

| Argument | Description |
|----------|-------------|
| `--group` | `prelim`, `main`, `ablation`, or `all` (default: `all`) |
| `--format` | `latex`, `markdown`, or `both` (default: `both`) |
| `--output` | Output file path (default: print to stdout) |

## Examples

```bash
# Generate all tables
/gen-table

# Generate main experiment table in LaTeX
/gen-table --group main --format latex --output paper_tables/main_results.tex

# Generate A experiment summary in Markdown
/gen-table --group prelim --format markdown
```

## Output Format

Tables follow MIS paper style:
- Rows = methods / models
- Columns = benchmark × metric (ASR, RSR, RR, HR)
- Best value per column in **bold**
- Includes ↑/↓ annotations where applicable

## Invocation

```python
python /mnt/hdd/xuran/vlm_safety_harness/scripts/generate_report.py \
  --results-dir results/ \
  [--group prelim|main|ablation|all] \
  [--format latex|markdown|both] \
  [--output FILE]
```
