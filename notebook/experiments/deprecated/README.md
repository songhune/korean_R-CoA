# Deprecated Scripts

This directory contains scripts that have been deprecated in favor of the unified runner.

## Deprecated Files

### run_all_evaluations.sh
- **Replaced by**: [run_evaluation.sh](../run_evaluation.sh)
- **Reason**: Unified with few-shot runner
- **Deprecated**: 2025-10-30
- **Status**: Kept for reference only

### exp6_run_fewshot.sh
- **Replaced by**: [run_evaluation.sh](../run_evaluation.sh) with `fewshot` mode
- **Reason**: Unified with zero-shot runner
- **Deprecated**: 2025-10-30
- **Status**: Kept for reference only

## Migration Guide

**Old usage:**
```bash
./run_all_evaluations.sh test
./exp6_run_fewshot.sh
```

**New usage:**
```bash
./run_evaluation.sh test
./run_evaluation.sh fewshot "1 3"
```

See [README_UNIFIED_RUNNER.md](../README_UNIFIED_RUNNER.md) for complete migration guide.
