# fianco-tournament

Fianco game simulator with an ai player
Rules from: http://www.di.fc.ul.pt/~jpn/gv/fianco.htm

Features:

- [x] Game simulator
- [x] Negamax with Alpha Beta Pruning
- [x] Iterative deeping with time limit
- [x] Trasposition table
- [x] Killer Moves
- [x] Dynamic Deeping when capturing
- [ ] Multi-threading

## Requirements

It is suggested to use a python virtual environment to run the project and install the dependencies in the requirements.txt file.

To build the project from scratch, you will need to install the rust toolchain and the maturin package. For more information on getting started with PyO3, visit the [PyO3 Getting Started Guide](https://pyo3.rs/v0.22.3/getting-started).

## How to build

To build the project and install it in the local env, run the following command:

```bash
maturin dev --release
```

# How to run

To run the project:

- Run the following command to start the game:

```bash
python gui.py
```
