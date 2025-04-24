# PyŠkvor (Piškvorky / Gomoku AI)
A Python implementation of the game Piškvorky (Gomoku). Features an AI player inspired by the AlphaGo algorithm.

<p align="center">
  <img width="500" alt="image" src="https://github.com/user-attachments/assets/7bcee6ac-3865-44cf-be6d-37c23972e446" />
</p>


This project originated as a Semestral Thesis for the Algoritmization 1 at MFF UK course in 2020.

## Technical Details

The core of the AI combines a neural network with Monte Carlo Tree Search (MCTS):

1.  **Neural Network (Policy Network):** Predicts promising moves from a given board state. It was trained via supervised learning using games played by established Gomoku engines.
2.  **Monte Carlo Tree Search (MCTS):** Intelligently explores the game tree, using the policy network's predictions to guide the search.

See pySkvor_dokumentace.docx for more information.


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

1.  **Running the Game:** Start the game by running the `main.py` file.
2.  **Setting Think Time:** After starting, the program will ask in the terminal how many seconds PyŠkvor should think. For ideal performance, it's recommended to enter **20 seconds or more**. For a quick game, go for **5 seconds**.
3.  **Game Board:** A window with the game board will appear. **You will be playing with the blue stones.**
4.  **First Move:** Wait for the computer to make its first move.
5.  **Your Move:** Make your move by clicking on any square on the game board.
6.  **Important:** **Do not click** on the window while the computer is calculating its move.
7.  **Continuing the Game:** Note that **the game does not automatically stop after one side wins.**
8.  **New Game:** To start a new game restart the script.
