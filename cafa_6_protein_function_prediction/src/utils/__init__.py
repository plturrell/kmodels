"""Utility helpers (e.g., leaderboard sync, submissions) for CAFA 6."""

from .leaderboard import (  # noqa: E402
    LeaderboardEntry,
    authenticate,
    build_parser as build_leaderboard_parser,
    fetch_leaderboard,
    format_summary,
    main as leaderboard_main,
    read_local_score,
)
from .active_learning_cli import main as active_learning_main  # noqa: E402
from .evaluate_runs import main as evaluate_runs_main  # noqa: E402
from .interpretability_cli import main as interpretability_main  # noqa: E402
from .submission import (  # noqa: E402
    build_submission_parser,
    main as submission_main,
    submit_file,
)

__all__ = [
    "LeaderboardEntry",
    "active_learning_main",
    "authenticate",
    "build_leaderboard_parser",
    "build_submission_parser",
    "fetch_leaderboard",
    "format_summary",
    "leaderboard_main",
    "interpretability_main",
    "evaluate_runs_main",
    "read_local_score",
    "submission_main",
    "submit_file",
]
