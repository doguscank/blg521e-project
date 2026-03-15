"""Training plot helpers."""

from __future__ import annotations


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "matplotlib is required for visualization. Install with: pip install '.[viz]'"
        ) from exc
    return plt


def plot_training_history(
    loss_history: list[float],
    avg_loss_history: list[float],
    title: str = "Evolution Loss History",
):
    """Create a line plot of min-loss and P50 average loss."""
    plt = _require_matplotlib()

    fig, ax = plt.subplots(figsize=(10, 4))
    x = list(range(len(loss_history)))
    ax.plot(x, loss_history, label="Min Loss", color="black", linewidth=1.8)
    ax.plot(x, avg_loss_history, label="P50 Avg Loss", color="gray", linewidth=1.4)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig, ax
