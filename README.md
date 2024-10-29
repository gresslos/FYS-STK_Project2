# FYS-STK_Project2

## Overview
- `task_a.py`                           :  OLS, GD-methods, and plots of results on data: **3. order polymial**
- `task b c.py`                         :  **Write here**
- `task b c tensorflow comparison.py`   :  **Write here**
- `task d.py`                           :  **Write here**
- `task_d_OP.py`                        :  **Write here**
- `task_e.py`                           :  **Write here**
- Additional_Plots/Linear_Regression    : Include plots from `task_a.py`, `task b c.py` and `task b c tensorflow comparison.py`
- Additional_Plots/Linear_Regression    : Include plots from `task_d_OP.py` and `task_e.py`

  

## Functions
### GD-Functions (found in `task_a.py`)
- **Input Variables**:
  - `X`, `y`, `beta0`: Design Matrix, data point, regression coefficent respectivly
  - `Niter`, `n_epochs`, `m`, `eta / eta0`, `lmb`, `tol`: Hyperparameters
  - bool-values: Defined under __main__ == "__main__" describe if want or not want given property
    - **Note**: To swich between analytical cost function gradient change `want_Autograd` (line 532).
- **Structure**: Detailed structure provided in `Project2.tex`.


Do similar as above OLP and JH for certain functions



## How to Run the Code
```bash
$ python3 <filename>