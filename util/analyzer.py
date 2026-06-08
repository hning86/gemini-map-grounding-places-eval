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
            generated_cid = p.get("generated_cid", "").strip()
            grounding_name = None
            
            for chunk in row.get("grounding_chunks", []):
                maps_data = chunk.get("maps")
                web_data = chunk.get("web")
                
                chunk_pid = ""
                chunk_uri = ""
                chunk_title = ""
                
                if isinstance(maps_data, dict):
                    chunk_pid = maps_data.get("place_id", "")
                    chunk_uri = maps_data.get("uri", "")
                    chunk_title = maps_data.get("title", "")
                elif isinstance(web_data, dict):
                    chunk_uri = web_data.get("uri", "")
                    chunk_title = web_data.get("title", "")
                else:
                    # Fallback to old format
                    chunk_pid = chunk.get("place_id", "")
                    chunk_uri = chunk.get("uri", "")
                    chunk_title = chunk.get("title", "")
                    
                # Match by Place ID
                if generated_pid:
                    if chunk_pid and (chunk_pid == generated_pid or chunk_pid.endswith("/" + generated_pid)):
                        grounding_name = chunk_title
                        break
                    if chunk_uri and generated_pid in chunk_uri:
                        grounding_name = chunk_title
                        break
                # Match by CID
                if generated_cid:
                    if chunk_uri and generated_cid in chunk_uri:
                        grounding_name = chunk_title
                        break

            place_records.append({
                "model": row["model"],
                "effort": row["effort"],
                "query": row["query"],
                "iteration": row["iteration"],
                "title": p.get("title"),
                "generated_place_id": p.get("generated_place_id"),
                "generated_cid": p.get("generated_cid"),
                "grounded_place_id": p.get("grounded_place_id"),
                "status_place_id": p.get("status_place_id"),
                "status_grounded_place_id": p.get("status_grounded_place_id"),
                "status_cid": p.get("status_cid"),
                "retrieved_name_place_id": p.get("retrieved_name_place_id"),
                "retrieved_name_grounded_place_id": p.get("retrieved_name_grounded_place_id"),
                "retrieved_name_cid": p.get("retrieved_name_cid"),
                "grounding_name": grounding_name,
                "verified_place_id": p.get("verified_place_id", False),
                "verified_grounded_place_id": p.get("verified_grounded_place_id", False),
                "verified_cid": p.get("verified_cid", False),
                "matching_ids": p.get("matching_ids", False),
                "fuzzy_score_place_id": p.get("fuzzy_score_place_id", 0.0),
                "fuzzy_score_grounded_place_id": p.get("fuzzy_score_grounded_place_id", 0.0),
                "fuzzy_score_cid": p.get("fuzzy_score_cid", 0.0),
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
        df_places["is_valid_pid"] = df_places["status_place_id"] == "OK"
        df_places["is_valid_gpid"] = df_places["status_grounded_place_id"] == "OK"
        df_places["is_valid_cid"] = df_places["status_cid"] == "OK"
        
        place_summary = df_places.groupby(["model", "effort"]).agg(
            total_places=("title", "count"),
            verified_places=("verified", "sum"),
            verified_pids=("verified_place_id", "sum"),
            verified_gpids=("verified_grounded_place_id", "sum"),
            verified_cids=("verified_cid", "sum"),
            valid_pids=("is_valid_pid", "sum"),
            valid_gpids=("is_valid_gpid", "sum"),
            valid_cids=("is_valid_cid", "sum"),
            matching_ids_sum=("matching_ids", "sum"),
            avg_fuzzy_pid=("fuzzy_score_place_id", "mean"),
            avg_fuzzy_gpid=("fuzzy_score_grounded_place_id", "mean"),
            avg_fuzzy_cid=("fuzzy_score_cid", "mean")
        ).reset_index()
        
        # Merge run and place summaries
        summary = run_summary.merge(place_summary, on=["model", "effort"], how="left")
        
        # Calculate rates
        summary["pid_verification_rate"] = (summary["verified_pids"] / summary["total_places"]) * 100
        summary["gpid_verification_rate"] = (summary["verified_gpids"] / summary["total_places"]) * 100
        summary["cid_verification_rate"] = (summary["verified_cids"] / summary["total_places"]) * 100
    else:
        summary = run_summary
        summary["total_places"] = 0
        summary["verified_places"] = 0
        summary["verified_pids"] = 0
        summary["verified_gpids"] = 0
        summary["verified_cids"] = 0
        summary["valid_pids"] = 0
        summary["valid_gpids"] = 0
        summary["valid_cids"] = 0
        summary["matching_ids_sum"] = 0
        summary["avg_fuzzy_pid"] = 0.0
        summary["avg_fuzzy_gpid"] = 0.0
        summary["avg_fuzzy_cid"] = 0.0
        summary["pid_verification_rate"] = 0.0
        summary["gpid_verification_rate"] = 0.0
        summary["cid_verification_rate"] = 0.0
        
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
        f.write("- **Stage A**: Gemini generation with structured output schema (requiring both PlaceID and CID) and Google Maps grounding enabled.\n")
        f.write("- **Stage B**: Live Verification of the generated PlaceID, the grounded PlaceID (extracted from Maps grounding chunks matching the CID), and the CID.\n\n")
        
        f.write("## 📊 Consolidated Metrics Summary\n\n")
        
        headers = [
            "Model", "Runs", "Avg Gemini Call (s)", "Avg Places API Call (s)", 
            "Grounded Rate (%)", "Total Places", "Gen PlaceID Verif (%)", "Grounded PlaceID Verif (%)", "CID Verif (%)"
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
                f"{r['pid_verification_rate']:.2f}%" if "pid_verification_rate" in r else "0.00%",
                f"{r['gpid_verification_rate']:.2f}%" if "gpid_verification_rate" in r else "0.00%",
                f"{r['cid_verification_rate']:.2f}%" if "cid_verification_rate" in r else "0.00%"
            ])
            
        from tabulate import tabulate
        f.write(tabulate(table_rows, headers=headers, tablefmt="github"))
        f.write("\n\n")
        
        f.write("## 🔍 Place-Level Error & Matching Analysis\n\n")
        place_headers = [
            "Model", "Total Places", "Valid Gen PIDs", "Valid Grounded PIDs", "Valid CIDs", "Matching IDs", "Avg Fuzzy (Gen PID)", "Avg Fuzzy (Gr PID)", "Avg Fuzzy (CID)"
        ]
        place_rows = []
        for _, r in summary.iterrows():
            place_rows.append([
                f"**{r['model']}**",
                int(r['total_places']),
                int(r['valid_pids']) if "valid_pids" in r else 0,
                int(r['valid_gpids']) if "valid_gpids" in r else 0,
                int(r['valid_cids']) if "valid_cids" in r else 0,
                int(r['matching_ids_sum']) if "matching_ids_sum" in r else 0,
                f"{r['avg_fuzzy_pid']:.2f}" if "avg_fuzzy_pid" in r else "0.00",
                f"{r['avg_fuzzy_gpid']:.2f}" if "avg_fuzzy_gpid" in r else "0.00",
                f"{r['avg_fuzzy_cid']:.2f}" if "avg_fuzzy_cid" in r else "0.00"
            ])
        f.write(tabulate(place_rows, headers=place_headers, tablefmt="github"))
        f.write("\n\n")
        
        # 3. Add Place ID Verification Registry Table
        if not df_places.empty:
            f.write("## 📌 Place ID & CID Verification Registry\n\n")
            f.write("The table below lists the unique Place IDs and CIDs generated by each model, their respective validation status, and whether they successfully resolve to the same location:\n\n")
            
            df_unique_places = df_places[[
                "model", "generated_place_id", "generated_cid", "grounded_place_id",
                "grounding_name", "title", 
                "retrieved_name_place_id", "retrieved_name_grounded_place_id", "retrieved_name_cid",
                "verified_place_id", "verified_grounded_place_id", "verified_cid", "matching_ids", "verified"
            ]].drop_duplicates().copy()
            
            # Sort registry by model and name
            df_unique_places = df_unique_places.sort_values(by=["model", "title"]).reset_index(drop=True)
            
            registry_headers = [
                "Model", "Generated Place ID", "Grounded Place ID", "Generated CID", 
                "Gemini Name", "Grounding Name", "Resolved Name (API)", 
                "Gen PID Valid?", "Grounded PID Valid?", "CID Valid?", "Match?"
            ]
            registry_rows = []
            for _, r in df_unique_places.iterrows():
                resolved_name = r["retrieved_name_cid"] if r["retrieved_name_cid"] else (r["retrieved_name_grounded_place_id"] if r["retrieved_name_grounded_place_id"] else (r["retrieved_name_place_id"] if r["retrieved_name_place_id"] else "N/A"))
                pid_valid_str = "✅ Yes" if r["verified_place_id"] else "❌ No"
                gpid_valid_str = "✅ Yes" if r["verified_grounded_place_id"] else "❌ No"
                cid_valid_str = "✅ Yes" if r["verified_cid"] else "❌ No"
                match_str = "✅ Yes" if r["matching_ids"] else "❌ No"
                
                pid_raw = r["generated_place_id"]
                cid_raw = r["generated_cid"]
                gpid_raw = r["grounded_place_id"]
                
                pid_has_value = pd.notna(pid_raw) and str(pid_raw).strip() not in ["", "nan", "None", "N/A"]
                cid_has_value = pd.notna(cid_raw) and str(cid_raw).strip() not in ["", "nan", "None", "N/A"]
                gpid_has_value = pd.notna(gpid_raw) and str(gpid_raw).strip() not in ["", "nan", "None", "N/A"]
                
                pid_display = str(pid_raw).strip() if pid_has_value else "N/A"
                cid_display = str(cid_raw).strip() if cid_has_value else "N/A"
                gpid_display = str(gpid_raw).strip() if gpid_has_value else "N/A"
                
                if not r["verified_place_id"] and pid_has_value:
                    pid_display = f"{pid_display} (Invalid)"
                if not r["verified_grounded_place_id"] and gpid_has_value:
                    gpid_display = f"{gpid_display} (Invalid)"
                if not r["verified_cid"] and cid_has_value:
                    cid_display = f"{cid_display} (Invalid)"
                
                grounding_name_str = r["grounding_name"] if pd.notna(r["grounding_name"]) and r["grounding_name"] else "N/A"
                
                registry_rows.append([
                    sanitize_md_cell(r["model"]),
                    sanitize_md_cell(pid_display),
                    sanitize_md_cell(gpid_display),
                    sanitize_md_cell(cid_display),
                    sanitize_md_cell(r["title"]),
                    sanitize_md_cell(grounding_name_str),
                    sanitize_md_cell(resolved_name),
                    sanitize_md_cell(pid_valid_str),
                    sanitize_md_cell(gpid_valid_str),
                    sanitize_md_cell(cid_valid_str),
                    sanitize_md_cell(match_str)
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
        f.write("1. **Avg Gemini Call (s):** Average time taken by Gemini to run Google Maps grounding and output the structured JSON response containing the places, PlaceIDs, and CIDs.\n")
        f.write("2. **Avg Places API Call (s):** Average time taken to verify all PlaceIDs and CIDs in parallel against their respective Google APIs.\n")
        f.write("3. **Grounded Rate (%):** The percentage of successful Gemini requests that contained any grounding metadata chunks returned from Google Maps.\n")
        f.write("4. **Gen PlaceID Verif (%):** The percentage of model-generated places where the directly-generated `PlaceId` was valid AND the fuzzy match score was \\ge 85%.\n")
        f.write("5. **Grounded PlaceID Verif (%):** The percentage of model-generated places where the `grounded_place_id` (extracted from the grounding metadata) was valid AND the fuzzy match score was \\ge 85%.\n")
        f.write("6. **CID Verif (%):** The percentage of model-generated places where the numeric `CID` was valid AND the fuzzy match score was \\ge 85%.\n")
        
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
