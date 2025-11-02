import argparse
import json
import logging
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import math
import webbrowser
import tempfile
from typing import Dict

from src.configs import MatchingEngineConfig
from src.matching_engine import MatchingEngine

def plot_candidates(candidates: Dict):
    """
    Plot interactive gauge charts for each candidate using Plotly.

    Args:
        candidates (list[dict]): List of candidate dicts with scores and resume info.
    """

    n_candidates = len(candidates)
    cols = 4  # one for each score type
    rows = n_candidates

    # Create subplot grid (each cell = gauge)
    fig = make_subplots(
        rows=rows,
        cols=cols,
        specs=[[{"type": "indicator"}] * cols for _ in range(rows)],
        subplot_titles=[
            "Skills", "Experience", "Title", "Total"
        ] * n_candidates,
        horizontal_spacing=0.08,
        vertical_spacing=0.1,
    )

    # Define consistent gauge style
    gauge_layout = dict(
        axis=dict(range=[0, 1]),
        bar=dict(thickness=0.3, color="royalblue"),
        bgcolor="white",
        borderwidth=1,
        bordercolor="gray",
    )

    # Colors per metric
    colors = {
        "skills_score": "dodgerblue",
        "experience_score": "mediumseagreen",
        "title_score": "darkorange",
        "total_score": "purple"
    }

    for i, cand in enumerate(candidates, start=1):
        name = f"{cand.get('first_name', '')} {cand.get('last_name', '')}".strip() or cand['email']
        email = cand['email']

        # Add gauges for each score
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=cand["skills_score"],
                title={
                    "text": f"<b>{name}</b>",
                    "font": {"size": 16}
                },
                gauge={**gauge_layout, "bar": {"color": colors["skills_score"]}},
                domain={"x": [0, 1], "y": [0, 1]},
            ),
            row=i,
            col=1,
        )

        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=cand["experience_score"],
                # title={"text": "Experience", "font": {"size": 14}},
                gauge={**gauge_layout, "bar": {"color": colors["experience_score"]}},
            ),
            row=i,
            col=2,
        )

        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=cand["title_score"],
                # title={"text": "Title", "font": {"size": 14}},
                gauge={**gauge_layout, "bar": {"color": colors["title_score"]}},
            ),
            row=i,
            col=3,
        )

        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=cand["total_score"],
                # title={"text": "Total", "font": {"size": 14}},
                gauge={**gauge_layout, "bar": {"color": colors["total_score"]}},
            ),
            row=i,
            col=4,
        )

    # Layout styling
    fig.update_layout(
        height=300 * n_candidates,
        width=1200,
        margin=dict(t=50, b=30),
        title=dict(
            text=f"Candidate Matching Scores ({n_candidates} results)",
            x=0.5,
            font=dict(size=22)
        ),
        template="plotly_white"
    )
    return fig
    # # Save to temp HTML with scrollable layout
    # html = f"""
    # <html>
    # <head>
    #     <style>
    #         body {{
    #             font-family: Arial, sans-serif;
    #             margin: 0;
    #             padding: 0;
    #             overflow-y: scroll;
    #         }}
    #         .container {{
    #             height: 90vh;
    #             overflow-y: scroll;
    #             padding: 20px;
    #             background-color: #f8f9fa;
    #         }}
    #     </style>
    # </head>
    # <body>
    #     <div class="container">
    #         {fig.to_html(include_plotlyjs='cdn', full_html=False)}
    #     </div>
    # </body>
    # </html>
    # """

    # temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
    # temp_file.write(html.encode("utf-8"))
    # temp_file.close()
    # webbrowser.open(f"file://{temp_file.name}")

    # print(f"✅ Dashboard opened in browser — showing {n_candidates} candidates.")


if __name__=='__main__':
    # --- Parse command-line arguments ---
    parser = argparse.ArgumentParser(
        description="Run the resume-job matching engine with custom weights and parameters."
    )

    parser.add_argument(
        "--jd",
        type=str,
        required=True,
        help="Path to the job description JSON file."
    )

    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="Number of top candidates to return (default: 10)."
    )

    parser.add_argument(
        "--skills_w",
        type=float,
        default=0.5,
        help="Weight for skills match (default: 0.5)."
    )

    parser.add_argument(
        "--experience_w",
        type=float,
        default=0.1,
        help="Weight for experience match (default: 0.2)."
    )

    parser.add_argument(
        "--title_w",
        type=float,
        default=0.4,
        help="Weight for title match (default: 0.3)."
    )

    args = parser.parse_args()

    # --- Validate weights ---
    total_w = args.skills_w + args.experience_w + args.title_w
    if abs(total_w - 1.0) > 1e-6:
        raise ValueError(
            f"Invalid weight configuration: sum of weights must be 1.0 (got {total_w})"
        )

    # --- Load job description ---
    with open(args.jd, "r") as f:
        job_description = json.load(f)

    # --- Initialize Matching Engine ---
    matching_engine_config = MatchingEngineConfig()
    matching_engine = MatchingEngine(matching_engine_config, verbose=logging.INFO)

    # --- Run Matching ---
    print(f"\n🔍 Running Matching Engine for JD: {args.jd}")
    print(f"   top_k={args.top_k}, skills_w={args.skills_w}, experience_w={args.experience_w}, title_w={args.title_w}")

    results = matching_engine.run(
        job_description,
        skill_w=args.skills_w,
        experience_w=args.experience_w,
        title_w=args.title_w,
        top_k=args.top_k
    )

    # --- Display Results ---
    print("\n🏆 Top Candidates:")
    for i, candidate in enumerate(results, 1):
        name = f"{candidate.get('first_name', '')} {candidate.get('last_name', '')}".strip()
        print(f"\n{i}. {name or candidate['email']}")
        print(f"   Email: {candidate['email']}")
        print(f"   Skills: {candidate['skills_score']:.3f}")
        print(f"   Experience: {candidate['experience_score']:.3f}")
        print(f"   Title: {candidate['title_score']:.3f}")
        print(f"   Total Score: {candidate['total_score']:.3f}")

    # plot 
    fig=plot_candidates(results)
    fig.show(renderer='browser')