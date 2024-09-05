# QMWaveFunctionSolver
A tool to solve Quantum Mechanics based problems. The programm takes an input from `WaveFunction.py` and finds numerical solutions which then may be plotted in `WFAnimation.py`. The numerical solutions are computed by the SciPy based Runge-Kutta and the self-implemented Crank-Nicolson procedures for a time-independent potential. For comparison one can set up the analytical energy eigenfunctions so the solver evaluates an initial free wave packet by Gauss-Hermite integration. 
## Gauss-Hermite integration
$$
\int_{-\infty}^{\infty} \mathrm{d}\tilde{x} \, \mathrm{exp}{(-\tilde{x}^2)} f(\tilde{x}) \simeq \sum_{j=1}^{n} w_j f(\tilde{x}_j)
$$
<!-- mit Hermite Polynome (und Nullstellen)
$$
\begin{align*}
    \tilde{H}_{-1}(\tilde{x}) &= 0, \tilde{H}_{0}(\tilde{x}) = 1\\
    \tilde{H}_{j}(\tilde{x}) &= 2\tilde{x}\tilde{H}_{j-1}(\tilde{x})-2(j-1)\tilde{H}_{j-2}(\tilde{x}) \text{    für    } j=1,\dots,n
\end{align*}
$$
und Gewichte -->
$$
w_j = \frac{2^{n-1}n!\sqrt{\pi}}{n^2\tilde{H}_{n-1}(\tilde{x}_j)^2}
$$
<!-- Approximation des Wellenpakets -->
$$
\psi(x, t) \simeq N \sum_{j=1}^{n} w_j \exp ( -\text{i}\frac{(2\sigma_p \tilde{x}_j+p_0)^2}{2m\hbar}t ) \sum_{\alpha} \varphi_\alpha(x, 2\sigma_p \tilde{x}_j+p_0)
$$
## Runge-Kutta procedure
[SciPy API: `solve_ivp(..)`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html)
## Crank-Nicolson procedure
<!-- $$
\psi_j^{n+1} = ( \textbf{I} - \frac{\Delta t}{2\text{i}\hbar}\mathbf{H} )^{-1} ( \textbf{I} + \frac{\Delta t}{2\text{i}\hbar}\mathbf{H} ) \psi_j^n
$$
mit Einheitsmatrix $\textbf{I}$ und Hamiltonoperator -->
$$
\textbf{H} = -\frac{\hbar^2}{2m(\Delta x)^2}    
\begin{pmatrix}
    -2 & 1 & 0 & \cdots & 0\\
    1 & -2 & 1 & \cdots & 0\\
    0 & 1 & -2 & \cdots & 0\\
    \vdots & \vdots & \vdots & \ddots & \vdots\\
    0 & 0 & 0 & \cdots & -2
\end{pmatrix} +
\begin{pmatrix}
    V_1 & 0 & 0 & \cdots & 0\\
    0 & V_2 & 0 & \cdots & 0\\
    0 & 0 & V_3 & \cdots & 0\\
    \vdots & \vdots & \vdots & \ddots & \vdots\\
    0 & 0 & 0 & \cdots & V_J
\end{pmatrix}
$$