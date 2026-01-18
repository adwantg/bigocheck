# Author: gadwant
"""
Optional plotting utilities for bigocheck.

Requires matplotlib to be installed. If not available, functions raise ImportError.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .core import Analysis


def plot_analysis(
    analysis: "Analysis",
    *,
    title: str = "Complexity Analysis",
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """
    Plot benchmark measurements with the best-fit complexity curve.
    
    Args:
        analysis: Analysis object from benchmark_function.
        title: Plot title.
        save_path: If provided, save the plot to this path.
        show: If True, display the plot interactively.
    
    Raises:
        ImportError: If matplotlib is not installed.
    
    Example:
        >>> from bigocheck import benchmark_function
        >>> from bigocheck.plotting import plot_analysis
        >>> analysis = benchmark_function(lambda n: sum(range(n)), [100, 200, 400])
        >>> plot_analysis(analysis, show=False, save_path="plot.png")
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError(
            "matplotlib is required for plotting. Install with: pip install matplotlib"
        )
    
    from .core import complexity_basis
    
    # Extract data
    sizes = [m.size for m in analysis.measurements]
    times = [m.seconds for m in analysis.measurements]
    std_devs = [m.std_dev for m in analysis.measurements]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot measurements with error bars if std_dev available
    if any(s > 0 for s in std_devs):
        ax.errorbar(sizes, times, yerr=std_devs, fmt='o', capsize=5, 
                    label='Measurements', markersize=8, color='blue')
    else:
        ax.scatter(sizes, times, label='Measurements', s=80, color='blue', zorder=5)
    
    # Find and plot best fit curve
    best_fit = next((f for f in analysis.fits if f.label == analysis.best_label), None)
    if best_fit:
        basis_fn = complexity_basis().get(analysis.best_label)
        if basis_fn:
            # Generate smooth curve
            import numpy as np
            x_smooth = np.linspace(min(sizes), max(sizes), 100)
            y_smooth = [best_fit.scale * basis_fn(int(x)) for x in x_smooth]
            ax.plot(x_smooth, y_smooth, 'r-', linewidth=2, 
                    label=f'Best fit: {analysis.best_label}')
    
    ax.set_xlabel('Input Size (n)', fontsize=12)
    ax.set_ylabel('Time (seconds)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_all_fits(
    analysis: "Analysis",
    *,
    title: str = "All Complexity Fits",
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """
    Plot benchmark measurements with all fitted complexity curves.
    
    Args:
        analysis: Analysis object from benchmark_function.
        title: Plot title.
        save_path: If provided, save the plot to this path.
        show: If True, display the plot interactively.
    
    Raises:
        ImportError: If matplotlib is not installed.
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        raise ImportError(
            "matplotlib and numpy are required for plotting. "
            "Install with: pip install matplotlib numpy"
        )
    
    from .core import complexity_basis
    
    sizes = [m.size for m in analysis.measurements]
    times = [m.seconds for m in analysis.measurements]
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Plot measurements
    ax.scatter(sizes, times, label='Measurements', s=100, color='black', zorder=10)
    
    # Plot all fit curves
    x_smooth = np.linspace(min(sizes), max(sizes), 100)
    colors = plt.cm.tab10(np.linspace(0, 1, len(analysis.fits)))
    
    for fit, color in zip(analysis.fits, colors):
        basis_fn = complexity_basis().get(fit.label)
        if basis_fn and fit.scale > 0:
            y_smooth = [fit.scale * basis_fn(int(x)) for x in x_smooth]
            alpha = 1.0 if fit.label == analysis.best_label else 0.4
            linewidth = 3 if fit.label == analysis.best_label else 1.5
            ax.plot(x_smooth, y_smooth, color=color, linewidth=linewidth, 
                    alpha=alpha, label=f'{fit.label} (err={fit.error:.3f})')
    
    ax.set_xlabel('Input Size (n)', fontsize=12)
    ax.set_ylabel('Time (seconds)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()
