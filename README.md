# Employee Goal Generation

An AI-powered Python application that automates the generation and evaluation of employee goals using Large Language Models (LLMs).

## Overview

This system generates personalized employee goals using OpenAI's language models, evaluates them for quality, and supports batch processing for organizational efficiency. It includes synthetic data generation capabilities for testing and development.

## Features

- **AI Goal Generation**: Uses OpenAI GPT models for intelligent goal creation
- **Batch Processing**: Handles multiple employee goals simultaneously
- **Goal Evaluation**: AI-powered assessment of goal quality
- **Synthetic Data**: Generates realistic company and employee test data
- **Data Export**: Supports CSV and JSON output formats

## Requirements

- Python >= 3.13.3
- OpenAI API key
- Required Python packages (managed via `uv`)

## Installation

1. Clone the repository
2. Install dependencies:
# Install uv if you don't have it yet
curl -Ls https://astral.sh/uv/install.sh | bash

```bash
uv sync
```
# Activate virtual environment
```bash
source .venv/bin/activate
```
# Install dependencies
```bash
uv pip install -r pyproject.toml
```

## Configuration

1. Create a `.env` file in the project root:
```env
OPENAI_API_KEY=your_api_key_here
```

2. Adjust settings in `main.py`:
```python
NUM_EMPLOYEES = 50  # Number of employees
PROVIDER = "openai"  # LLM provider
MODEL = "gpt-4.1-nano"  # Model name
```

## Usage

Run the application:
```bash
python main.py
```

The program will:
1. Generate synthetic employee data
2. Create AI-powered goals for each employee
3. Evaluate the quality of generated goals
4. Save results in the `output_data` directory

## Project Structure

- `main.py` - Application entry point
- `scripts/` - Data generation utilities
- `task_configs/` - LLM configuration
- `task_endpoints/` - Goal generation and evaluation logic
- `output_data/` - Generated files

## Output Files

The system generates three main files in the `output_data` directory:
- `synthetic_employee_data_[N]_[PROVIDER]_[MODEL].csv` - Raw employee data
- `synthetic_employee_data_[N]_[PROVIDER]_[MODEL]_with_goals.json` - Data with generated goals
- `synthetic_employee_data_[N]_[PROVIDER]_[MODEL]_with_evaluated_goals.json` - Final data including goal evaluations

## License

MIT License
