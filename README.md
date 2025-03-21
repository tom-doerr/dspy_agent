# DSPy Agent

An autonomous LLM agent built on Stanford's DSPy framework.

## Features

- Simple DSPy pipeline for LLM tasks
- Feedback-based optimization using deepseek-chat
- CLI interface for specifying tasks and rating criteria

## Installation

```bash
poetry install
```

## Usage

```bash
# Run the agent with a specific task
dspy-agent run "Summarize this article" --criterion "Conciseness and accuracy"

# Optimize the pipeline
dspy-agent optimize --task "Write a product description" --criterion "Persuasiveness"
```
