# Particular

## Installation
### 1. Clone the project

```bash
git clone https://github.com/C4rb0n6/Particular.git
cd Particular
```
### 2. Setup Virtual Environment
```bash
python -m venv .venv
source .venv/bin/activate   # On Windows, use `.venv\Scripts\activate`
```
### 3. Install Dependencies
```bash
pip install -r requirements.txt
```
### Usage

* Simulation settings, such as particle count and screen size, are adjustable at the top of [sim.py](/sim.py).
* After configuring the initial settings, run [sim.py](/sim.py). You will see on-screen simulation controls for gravity, container size, and more.
