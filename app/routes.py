from datetime import datetime
from typing import Optional, Dict, Any, List

import numpy as np
from flask import Blueprint, render_template, request, redirect, url_for

from vesta_ml.predictor import CyclePredictor

from app.db import insert_cycle, fetch_latest_cycles, count_cycles


main_bp = Blueprint("main", __name__)

_PREDICTOR: Optional[CyclePredictor] = None


def _get_predictor() -> CyclePredictor:
    global _PREDICTOR
    if _PREDICTOR is None:
        _PREDICTOR = CyclePredictor(verbose=False)
    return _PREDICTOR


def _parse_date(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%d")


def _format_date(date_str: str) -> str:
    """Format YYYY-MM-DD to a human-readable string like 'Mon, Jan 15, 2024'."""
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    return dt.strftime("%a, %b %-d, %Y")


def _calculate_cycle_stats(cycles: List[Dict[str, str]]) -> Optional[Dict[str, Any]]:
    """Calculate personalized cycle insights from recent cycles."""
    if len(cycles) < 2:
        return None
    
    # Parse dates and calculate cycle lengths and period durations
    cycle_lengths = []
    period_durations = []
    
    for i, cycle in enumerate(cycles):
        start = datetime.strptime(cycle["start_date"], "%Y-%m-%d")
        end = datetime.strptime(cycle["end_date"], "%Y-%m-%d")
        period_durations.append((end - start).days)
        
        if i < len(cycles) - 1:
            next_start = datetime.strptime(cycles[i + 1]["start_date"], "%Y-%m-%d")
            cycle_lengths.append((next_start - start).days)
    
    if not cycle_lengths:
        return None
    
    # Calculate statistics
    avg_cycle = np.mean(cycle_lengths)
    std_cycle = np.std(cycle_lengths)
    min_cycle = int(np.min(cycle_lengths))
    max_cycle = int(np.max(cycle_lengths))
    
    avg_period = np.mean(period_durations)
    min_period = int(np.min(period_durations))
    max_period = int(np.max(period_durations))
    
    # Determine consistency description
    if std_cycle < 2:
        consistency = "Very consistent"
    elif std_cycle < 4:
        consistency = "Fairly consistent"
    else:
        consistency = f"Varies by {int(std_cycle)}±{int(std_cycle)} days"
    
    # Determine recent trend (last 4 vs previous 4)
    if len(cycle_lengths) >= 8:
        recent_avg = np.mean(cycle_lengths[-4:])
        older_avg = np.mean(cycle_lengths[-8:-4])
        diff = recent_avg - older_avg
        
        if abs(diff) < 1:
            trend = "Stable pattern"
        elif diff > 1:
            trend = f"Cycles getting longer (+{diff:.1f} days)"
        else:
            trend = f"Cycles getting shorter ({diff:.1f} days)"
    else:
        trend = "Not enough data for trend (Need 8+ cycles)"
    
    return {
        "typical_cycle": f"{min_cycle}-{max_cycle} days",
        "typical_period": f"{min_period}-{max_period} days",
        "consistency": consistency,
        "trend": trend,
        "avg_cycle": f"{avg_cycle:.1f} days",
        "avg_period": f"{avg_period:.1f} days",
    }


@main_bp.route("/", methods=["GET", "POST"])
def index():
    error: Optional[str] = None
    prediction: Optional[Dict[str, Any]] = None
    cycles: List[Dict[str, str]] = []

    if request.method == "POST":
        start_date = request.form.get("start_date", "").strip()
        end_date = request.form.get("end_date", "").strip()

        try:
            start_dt = _parse_date(start_date)
            end_dt = _parse_date(end_date)
            if end_dt < start_dt:
                raise ValueError("End date must be on or after start date.")
            insert_cycle(start_date, end_date)
            # Redirect to prevent form resubmission on refresh
            return redirect(url_for("main.index"))
        except ValueError as exc:
            error = str(exc) or "Invalid date input."

    total_cycles = count_cycles()
    
    # Fetch cycles for prediction (oldest first, limit 6)
    cycles_for_prediction = fetch_latest_cycles(limit=6, oldest_first=True)
    
    # Fetch cycles for stats and display (oldest first for stats calculation)
    cycles_for_stats = fetch_latest_cycles(limit=12, oldest_first=True)

    if total_cycles >= 6 and len(cycles_for_prediction) >= 6:
        try:
            prediction = _get_predictor().predict_next_cycle(cycles_for_prediction)
            # Format prediction dates
            prediction["predicted_next_start_date"] = _format_date(
                prediction["predicted_next_start_date"]
            )
            prediction["date_range_lower"] = _format_date(prediction["date_range_lower"])
            prediction["date_range_upper"] = _format_date(prediction["date_range_upper"])
        except Exception as exc:
            error = f"Prediction failed: {exc}"
    elif error is None:
        error = (
            f"Add {6 - total_cycles} more cycle(s) to enable predictions. "
            "The model requires six completed cycles."
        )

    # Calculate personalized insights
    stats = _calculate_cycle_stats(cycles_for_stats)

    # Format cycles for display: most recent first with human-readable dates
    display_cycles = [
        {
            "start_date": _format_date(c["start_date"]),
            "end_date": _format_date(c["end_date"]),
        }
        for c in reversed(cycles_for_stats)
    ]

    return render_template(
        "index.html",
        prediction=prediction,
        error=error,
        cycles=display_cycles,
        total_cycles=total_cycles,
        stats=stats,
    )
