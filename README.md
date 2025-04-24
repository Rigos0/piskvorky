<img width="601" alt="image" src="https://github.com/user-attachments/assets/7bcee6ac-3865-44cf-be6d-37c23972e446" />

## Setup Instructions (using uv)

These instructions assume you have Python and `git` installed.

1.  **Install `uv`:**
    If you don't have `uv` installed yet, follow the official installation instructions:
    [https://github.com/astral-sh/uv#installation](https://github.com/astral-sh/uv#installation)
    *(Common methods include `pip install uv`, `brew install uv`, `curl -LsSf https://astral.sh/uv/install.sh | sh`)*

2.  **Clone the repository:**
    ```bash
    git clone https://github.com/Rigos0/piskvorky.git
    cd piskvorky
    ```

3.  **Create and activate a virtual environment:**
    ```bash
    # Create the virtual environment
    uv venv

    # Activate the environment (choose the command for your shell)
    # Windows (Command Prompt):
    .venv\Scripts\activate.bat
    # Windows (PowerShell):
    .venv\Scripts\Activate.ps1
    # Linux/macOS (Bash/Zsh):
    source .venv/bin/activate
    ```
    You should see `(.venv)` at the beginning of your command prompt.

4.  **Install dependencies:**
    This step assumes your project's dependencies are defined in a `pyproject.toml` file according to PEP 621 standards.
    ```bash
    uv sync
    ```
    This command will install all necessary dependencies specified in your `pyproject.toml` into the active virtual environment.

## Usage
