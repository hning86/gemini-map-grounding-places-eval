import json
import pandas as pd
import os
import sys
from dotenv import load_dotenv

load_dotenv()
PROJECT_ID = os.getenv("PROJECT_ID", "ninghai-ccai")
LOCATION = os.getenv("LOCATION", "global")


def sanitize_md_cell(val):
    """Sanitize vertical bars | in string values to prevent breaking Markdown tables."""
    if val is None:
        return ""
    val_str = str(val)
    return val_str.replace("|", "\\|")


def analyze_results(input_file, output_report):
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.", file=sys.stderr)
        return

    data = []
    with open(input_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
            
    if not data:
        print("Error: No data found in the results file.", file=sys.stderr)
        return
        
    df = pd.DataFrame(data)
    
    # Filter successful calls
    df_success = df[df["success"] == True].copy()
    
    if df_success.empty:
        print("No successful runs to analyze in the results file.", file=sys.stderr)
        return
        
    # 1. Compute Grounded Rate: successful runs that have grounding chunks
    df_success["is_grounded"] = df_success["grounding_chunks"].apply(lambda x: len(x) > 0)
    
    # 2. Extract Place-level metrics from 'places_verified'
    place_records = []
    for idx, row in df_success.iterrows():
        places_verified = row["places_verified"]
        if not places_verified:
            continue
        for p in places_verified:
            # Try to match with a grounding chunk in the same run
            generated_pid = p.get("generated_place_id", "").strip()
            grounding_name = None
            if generated_pid:
                for chunk in row.get("grounding_chunks", []):
                    chunk_pid = chunk.get("place_id", "")
                    chunk_uri = chunk.get("uri", "")
                    if chunk_pid and (chunk_pid == generated_pid or chunk_pid.endswith("/" + generated_pid)):
                        grounding_name = chunk.get("title")
                        break
                    if chunk_uri and generated_pid in chunk_uri:
                        grounding_name = chunk.get("title")
                        break

            place_records.append({
                "model": row["model"],
                "effort": row["effort"],
                "query": row["query"],
                "iteration": row["iteration"],
                "title": p.get("title"),
                "generated_place_id": p.get("generated_place_id"),
                "retrieved_name": p.get("retrieved_name"),
                "grounding_name": grounding_name,
                "status": p.get("status"),
                "fuzzy_score": p.get("fuzzy_score", 0.0),
                "verified": p.get("verified", False),
                "latency_b_individual": p.get("latency", 0.0)
            })
            
    df_places = pd.DataFrame(place_records) if place_records else pd.DataFrame()
    
    # Summarize run-level metrics (Latencies, Grounded Rate)
    run_summary = df_success.groupby(["model", "effort"]).agg(
        avg_latency_a=("latency_stage_a", "mean"),
        avg_latency_b=("latency_stage_b", "mean"),
        avg_latency_combined=("latency_combined", "mean"),
        grounded_rate=("is_grounded", "mean"),
        total_runs=("success", "count")
    ).reset_index()
    
    # Summarize place-level metrics if we have place data
    if not df_places.empty:
        df_places["is_valid_id"] = df_places["status"] == "OK"
        df_places["is_invalid_id"] = df_places["status"].isin(["NOT_FOUND", "INVALID_REQUEST"])
        df_places["is_other_error"] = ~df_places["status"].isin(["OK", "NOT_FOUND", "INVALID_REQUEST"])
        
        place_summary = df_places.groupby(["model", "effort"]).agg(
            total_places=("title", "count"),
            verified_places=("verified", "sum"),
            valid_id_places=("is_valid_id", "sum"),
            invalid_id_places=("is_invalid_id", "sum"),
            other_error_places=("is_other_error", "sum"),
            avg_fuzzy_score=("fuzzy_score", "mean")
        ).reset_index()
        
        # Merge run and place summaries
        summary = run_summary.merge(place_summary, on=["model", "effort"], how="left")
        
        # Calculate rates
        summary["verification_rate"] = (summary["verified_places"] / summary["total_places"]) * 100
        summary["hallucination_rate"] = (1 - (summary["verified_places"] / summary["total_places"])) * 100
        summary["invalid_id_rate"] = (summary["invalid_id_places"] / summary["total_places"]) * 100
    else:
        summary = run_summary
        summary["total_places"] = 0
        summary["verified_places"] = 0
        summary["avg_fuzzy_score"] = 0.0
        summary["verification_rate"] = 0.0
        summary["hallucination_rate"] = 100.0
        summary["invalid_id_rate"] = 100.0
        
    summary["grounded_rate"] = summary["grounded_rate"] * 100
    
    # Format and Sort
    summary["effort"] = pd.Categorical(summary["effort"], categories=["low", "medium", "high"], ordered=True)
    summary = summary.sort_values(by=["model", "effort"]).reset_index(drop=True)
    
    # Make directory for reports
    if os.path.dirname(output_report):
        os.makedirs(os.path.dirname(output_report), exist_ok=True)
        
    with open(output_report, "w") as f:
        f.write("# Gemini 2-Stage Grounding & Places API Verification Report\n\n")
        f.write("This report details the evaluation metrics and verification results from the 2-stage Point-of-Interest (POI) discovery framework.\n")
        f.write("- **Stage A**: Gemini generation with structured output schema (including PlaceID) and Google Maps grounding enabled.\n")
        f.write("- **Stage B**: Live Verification of the generated PlaceID via the Google Places API (Details endpoint), including fuzzy name matching.\n\n")
        
        f.write("## 📊 Consolidated Metrics Summary\n\n")
        
        headers = [
            "Model", "Runs", "Avg Gemini Call (s)", "Avg Places API Call (s)", 
            "Grounded Rate (%)", "Total Places", "Verified Places", "Verification Rate (%)"
        ]
        
        table_rows = []
        for _, r in summary.iterrows():
            table_rows.append([
                f"**{r['model']}**",
                int(r['total_runs']),
                f"{r['avg_latency_a']:.2f}s",
                f"{r['avg_latency_b']:.2f}s",
                f"{r['grounded_rate']:.2f}%",
                int(r['total_places']),
                int(r['verified_places']),
                f"{r['verification_rate']:.2f}%"
            ])
            
        from tabulate import tabulate
        f.write(tabulate(table_rows, headers=headers, tablefmt="github"))
        f.write("\n\n")
        
        f.write("## 🔍 Place-Level Error & Matching Analysis\n\n")
        place_headers = [
            "Model", "Total Places", "Valid PlaceIDs", "Invalid/Malformed/Stale PlaceIDs", "Other API Errors", "Avg Fuzzy Name Match Score"
        ]
        place_rows = []
        for _, r in summary.iterrows():
            place_rows.append([
                f"**{r['model']}**",
                int(r['total_places']),
                int(r['verified_places']) if "verified_places" in r else 0,
                int(r['invalid_id_places']) if "invalid_id_places" in r else 0,
                int(r['other_error_places']) if "other_error_places" in r else 0,
                f"{r['avg_fuzzy_score']:.2f}"
            ])
        f.write(tabulate(place_rows, headers=place_headers, tablefmt="github"))
        f.write("\n\n")
        
        # 3. Add Place ID Verification Registry Table
        if not df_places.empty:
            f.write("## 📌 Place ID Verification Registry\n\n")
            f.write("The table below lists the unique Place IDs generated by each model, the name returned by Gemini, and the resolved name verified by the Places API:\n\n")
            
            df_unique_places = df_places[["model", "generated_place_id", "grounding_name", "title", "retrieved_name", "verified"]].drop_duplicates().copy()
            
            # Sort registry by model and name
            df_unique_places = df_unique_places.sort_values(by=["model", "title"]).reset_index(drop=True)
            
            registry_headers = ["Model", "Generated Place ID / CID", "Name returned by Gemini", "Name from Grounding Chunk", "Resolved Name (Places API)", "Verified?"]
            registry_rows = []
            for _, r in df_unique_places.iterrows():
                verified_str = "✅ Yes" if r["verified"] else "❌ No"
                grounding_name_str = r["grounding_name"] if pd.notna(r["grounding_name"]) and r["grounding_name"] else "N/A"
                if not r["verified"]:
                    retrieved_name_str = "-- INVALID ID --"
                else:
                    retrieved_name_str = r["retrieved_name"] if r["retrieved_name"] else "N/A"
                registry_rows.append([
                    sanitize_md_cell(r["model"]),
                    f"`{sanitize_md_cell(r['generated_place_id'])}`",
                    sanitize_md_cell(r["title"]),
                    sanitize_md_cell(grounding_name_str),
                    sanitize_md_cell(retrieved_name_str),
                    sanitize_md_cell(verified_str)
                ])
            f.write(tabulate(registry_rows, headers=registry_headers, tablefmt="github"))
            f.write("\n\n")
            
        f.write("## 📝 Execution Details\n\n")
        f.write(f"- **Total runs attempted:** {len(df)}\n")
        f.write(f"- **Total successful runs:** {len(df_success)}\n")
        f.write(f"- **Overall Run-Level Success Rate:** {len(df_success)/len(df):.2%}\n")
        f.write(f"- **GCP Project:** `{PROJECT_ID}`\n")
        f.write(f"- **Vertex AI Location:** `{LOCATION}`\n\n")
        
        f.write("### Metric Definitions\n")
        f.write("1. **Avg Gemini Call (s):** Average time taken by Gemini to run Google Maps grounding and output the structured JSON response containing the places and PlaceIDs.\n")
        f.write("2. **Avg Places API Call (s):** Average time taken to verify all PlaceIDs in parallel against the Google Places API Details endpoint.\n")
        f.write("3. **Grounded Rate (%):** The percentage of successful Gemini requests that contained any grounding metadata chunks returned from Google Maps.\n")
        f.write(r"4. **Verification Rate (%):** The percentage of model-generated places where the PlaceID was valid (Places API returned `OK`) AND the fuzzy string match score between the generated place name and the retrieved place name was $\ge 85\%$." + "\n")
        
    print(f"Analysis complete. Beautiful report compiled at: {output_report}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Analyze 2-Stage Grounding & Verification raw results.")
    parser.add_argument("--input", type=str, help="Path to raw JSONL results.")
    parser.add_argument("--output", type=str, help="Path to save markdown report.")
    parser.add_argument("--quick", action="store_true", help="Shortcut to analyze quick_test_results.jsonl.")
    
    args = parser.parse_args()
    
    if args.quick:
        input_file = os.path.join("results", "quick_test_results.jsonl")
        output_report = os.path.join("reports", "quick_test_report.md")
    else:
        input_file = args.input or os.path.join("results", "full_eval_results.jsonl")
        output_report = args.output or os.path.join("reports", "full_evaluation_report.md")
        
    analyze_results(input_file, output_report)
