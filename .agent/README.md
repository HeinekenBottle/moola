# .agent/

Purpose: agentic workflows and SOPs. No runtime secrets.

## Structure

- `workflows/` - Automated workflow definitions (YAML)
- `tasks/` - Task templates and execution scripts
- `sops/` - Standard Operating Procedures for ML operations
- `tools/` - Utility scripts and helper tools for agents

## Usage

This directory contains documentation and automation for AI agents orchestrating the Moola ML pipeline. All workflows are stateless and reference external data via environment variables.
