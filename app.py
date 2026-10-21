
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, render_template, request


# --------------------------------------------------
# 1. PROJECT CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "ipl_predictor.joblib"
MATCHES_PATH = BASE_DIR / "data" / "matches.csv"

app = Flask(__name__)

# Load the trained model once when Flask starts.
if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model file not found: {MODEL_PATH}"
    )

model = joblib.load(MODEL_PATH)


# --------------------------------------------------
# 2. TEAM AND MODEL CONFIGURATION
# --------------------------------------------------

TEAMS = [
    "Chennai Super Kings",
    "Delhi Capitals",
    "Gujarat Titans",
    "Kolkata Knight Riders",
    "Lucknow Super Giants",
    "Mumbai Indians",
    "Punjab Kings",
    "Rajasthan Royals",
    "Royal Challengers Bengaluru",
    "Sunrisers Hyderabad",
]

FEATURES = [
    "batting_team",
    "bowling_team",
    "venue",
    "target_runs",
    "current_score",
    "wickets_lost",
    "legal_balls_bowled",
    "balls_remaining",
    "runs_required",
    "current_run_rate",
    "required_run_rate",
]


# --------------------------------------------------
# 3. LOAD VENUES FROM THE DATASET
# --------------------------------------------------

def load_venues():
    try:
        data = pd.read_csv(MATCHES_PATH)

        if "venue" not in data.columns:
            raise ValueError(
                "The matches.csv file has no venue column."
            )

        venues = (
            data["venue"]
            .dropna()
            .astype(str)
            .str.strip()
        )

        venues = sorted(
            venue for venue in venues.unique() if venue
        )

        if venues:
            return venues

    except (FileNotFoundError, pd.errors.ParserError, ValueError) as exc:
        print(f"Venue loading warning: {exc}")

    # Fallback if the CSV cannot be read.
    return [
        "Wankhede Stadium",
        "M. A. Chidambaram Stadium",
    ]


VENUES = load_venues()


# --------------------------------------------------
# 4. COMMON PAGE RENDERING
# --------------------------------------------------

def render_home(prediction=None, error=None, form_data=None):
    return render_template(
        "index.html",
        teams=TEAMS,
        venues=VENUES,
        prediction=prediction,
        error=error,
        form_data=form_data or {},
    )


@app.route("/")
def home():
    return render_home()


# --------------------------------------------------
# 5. PREDICTION ROUTE
# --------------------------------------------------

@app.route("/predict", methods=["POST"])
def predict():
    form = request.form

    try:
        batting_team = form.get(
            "batting_team", ""
        ).strip()

        bowling_team = form.get(
            "bowling_team", ""
        ).strip()

        venue = form.get("venue", "").strip()

        # Validate dropdown selections.
        if batting_team not in TEAMS:
            raise ValueError("Please select a valid batting team.")

        if bowling_team not in TEAMS:
            raise ValueError("Please select a valid bowling team.")

        if batting_team == bowling_team:
            raise ValueError(
                "Batting team and bowling team must be different."
            )

        if venue not in VENUES:
            raise ValueError("Please select a valid venue.")

        # Read numerical inputs.
        try:
            target = int(form.get("target_runs", ""))
            score = int(form.get("current_score", ""))
            wickets = int(form.get("wickets_lost", ""))
        except (TypeError, ValueError):
            raise ValueError(
                "Target, current score, and wickets must be valid numbers."
            )

        overs_text = form.get(
            "overs_completed", ""
        ).strip()

        # Validate score and target.
        if target <= 0:
            raise ValueError("Target must be greater than zero.")

        if score < 0:
            raise ValueError("Current score cannot be negative.")

        if score >= target:
            raise ValueError(
                "The target has already been reached. "
                "This live-chase prediction is no longer needed."
            )

        if not 0 <= wickets <= 9:
            raise ValueError(
                "For an ongoing chase, wickets lost must be between 0 and 9."
            )

        # Parse cricket overs notation:
        # 12.3 means 12 overs and 3 balls, not 12.3 decimal overs.
        overs_parts = overs_text.split(".")

        if (
            len(overs_parts) > 2
            or not overs_parts[0].isdigit()
        ):
            raise ValueError(
                "Enter overs in cricket notation, such as 12 or 12.3."
            )

        whole_overs = int(overs_parts[0])

        partial_text = (
            overs_parts[1]
            if len(overs_parts) == 2
            else "0"
        )

        if (
            len(partial_text) > 1
            or not partial_text.isdigit()
        ):
            raise ValueError(
                "Enter overs in cricket notation, such as 12 or 12.3."
            )

        partial_balls = int(partial_text)

        if not 0 <= partial_balls <= 5:
            raise ValueError(
                "The number after the decimal must be between 0 and 5."
            )

        if whole_overs > 19 or (
            whole_overs == 19 and partial_balls > 5
        ):
            raise ValueError(
                "For a standard T20 chase, overs cannot exceed 19.5 "
                "while the innings is ongoing."
            )

        legal_balls = whole_overs * 6 + partial_balls

        if legal_balls >= 120:
            raise ValueError(
                "A standard 20-over chase has ended."
            )

        # Calculate live match features.
        balls_remaining = 120 - legal_balls
        runs_required = target - score

        current_run_rate = (
            score / (legal_balls / 6)
            if legal_balls > 0
            else 0.0
        )

        required_run_rate = (
            runs_required / (balls_remaining / 6)
        )

        # Build model input using the exact training columns.
        model_input = pd.DataFrame(
            [{
                "batting_team": batting_team,
                "bowling_team": bowling_team,
                "venue": venue,
                "target_runs": target,
                "current_score": score,
                "wickets_lost": wickets,
                "legal_balls_bowled": legal_balls,
                "balls_remaining": balls_remaining,
                "runs_required": runs_required,
                "current_run_rate": current_run_rate,
                "required_run_rate": required_run_rate,
            }],
            columns=FEATURES,
        )

        # Predict the probability that the chasing team wins.
        probability = float(
            model.predict_proba(model_input)[0][1]
        )

        result = {
            "batting_team": batting_team,
            "bowling_team": bowling_team,
            "venue": venue,
            "target": target,
            "score": score,
            "wickets": wickets,
            "overs": overs_text,
            "runs_required": runs_required,
            "balls_remaining": balls_remaining,
            "current_run_rate": round(current_run_rate, 2),
            "required_run_rate": round(required_run_rate, 2),
            "win_probability": round(probability * 100, 1),
            "opponent_probability": round(
                (1 - probability) * 100, 1
            ),
        }

        return render_home(
            prediction=result,
            form_data=form.to_dict(),
        )

    except ValueError as exc:
        return render_home(
            error=str(exc),
            form_data=form.to_dict(),
        )

    except Exception as exc:
        # Log the technical error in the terminal.
        app.logger.exception(
            "Unexpected error while making a prediction"
        )

        return render_home(
            error=(
                "Something went wrong while generating the prediction. "
                "Please check the terminal for details."
            ),
            form_data=form.to_dict(),
        )


# --------------------------------------------------
# 6. START THE FLASK SERVER
# --------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True)
