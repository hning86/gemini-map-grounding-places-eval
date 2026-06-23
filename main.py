import os
import argparse
import sys
from util.evaluator import run_evaluation, DEFAULT_MODELS, DEFAULT_EFFORT, QUERIES
from util.analyzer import analyze_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Orchestrate 2-Stage Grounding & Verification Evaluation Pipeline.")
    parser.add_argument("--repetitions", "-r", type=int, help="Number of repetitions per query.")
    parser.add_argument("--quick", action="store_true", help="Shortcut to run a fast dry-run with 1 repetition of the first query.")
    parser.add_argument("--model", type=str, default=",".join(DEFAULT_MODELS), help="The target Gemini model(s) to evaluate (comma-separated).")
    parser.add_argument("--effort", type=str, default=DEFAULT_EFFORT, help="Thinking effort configuration (low/medium/high).")
    parser.add_argument("--workers", "-w", type=int, default=3, help="Number of concurrent workers for evaluation.")
    
    args = parser.parse_args()
    
    # Determine result and report files
    if args.quick:
        output_results = os.path.join("results", "quick_test_results.jsonl")
        output_report = os.path.join("reports", "quick_test_report.md")
        repetitions = 1
        eval_queries = [QUERIES[0]]
    else:
        output_results = os.path.join("results", "full_eval_results.jsonl")
        output_report = os.path.join("reports", "full_evaluation_report.md")
        repetitions = args.repetitions or 5
        eval_queries = QUERIES
        
    models_to_eval = [m.strip() for m in args.model.split(",")]
    
    # Clean up previous results if they exist to prevent combining old and new data
    base_res, ext_res = os.path.splitext(output_results)
    raw_results = base_res + "_raw" + ext_res
    files_to_clean = [
        output_results,
        output_results.replace(".jsonl", ".json"),
        raw_results,
        raw_results.replace(".jsonl", ".json")
    ]
    for file_path in files_to_clean:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception as e:
                print(f"Warning: Could not remove old file {file_path}: {e}")
                
    print("=" * 60)
    print("🚀 STARTING 2-STAGE MAPS GROUNDING & PLACES API VERIFICATION PIPELINE")
    print("=" * 60)
    
    try:
        run_evaluation(
            output_file=output_results,
            repetitions=repetitions,
            models=models_to_eval,
            effort=args.effort,
            queries=eval_queries,
            workers=args.workers
        )
    except Exception as e:
        print(f"❌ Pipeline failed during evaluation: {e}", file=sys.stderr)
        sys.exit(1)
        
    print("\n" + "=" * 60)
    print("📊 STARTING STATISTICAL ANALYSIS AND REPORT COMPILATION")
    print("=" * 60)
    
    try:
        analyze_results(
            input_file=output_results,
            output_report=output_report
        )
    except Exception as e:
        print(f"❌ Pipeline failed during analysis: {e}", file=sys.stderr)
        sys.exit(1)
        
    print("\n" + "=" * 60)
    print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"Detailed metrics report available at: {output_report}")
    print("=" * 60)
