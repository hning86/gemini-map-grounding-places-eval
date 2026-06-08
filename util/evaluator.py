import time
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from google import genai
from google.genai import types
from dotenv import load_dotenv
from .verifier import query_place_details, verify_places

load_dotenv()

# Configuration (Defaults)
DEFAULT_MODELS = ["gemini-3.1-flash-lite-preview", "gemini-3.1-flash-lite", "gemini-3.5-flash"]
DEFAULT_EFFORT = "low"
PROJECT_ID = os.getenv("PROJECT_ID")
LOCATION = os.getenv("LOCATION")
PLACES_API_KEY = os.getenv("PLACES_API_KEY")

if not PROJECT_ID or not LOCATION or not PLACES_API_KEY:
    raise ValueError("Missing required environment variables (PROJECT_ID, LOCATION, or PLACES_API_KEY) in .env")

QUERIES = [
    "Best street food spots and street food markets in Hanoi",
    "Best vegan restaurants in Berlin",
    "Top art museums and galleries in Paris",
    "Hidden specialty coffee shops in Tokyo",
    "Best rooftop bars with a view in Bangkok"
]

# JSON Schema for controlled generation (Stage A)
SCHEMA = {
    "type": "object",
    "properties": {
        "places": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "PlaceId":        {"type": "string"},
                    "CID":            {"type": "string"},
                    "title":          {"type": "string"},
                    "rating":         {"type": "string"},
                    "review_count":   {"type": "string"},
                    "text":           {"type": "string"},
                    "place_type":     {"type": "string"},
                    "opening_hours":  {"type": "string"},
                    "entry_price":    {"type": "string"},
                    "address":        {"type": "string"}
                },
                "required": ["PlaceId", "CID", "title", "rating", "review_count", "text"],
                "additionalProperties": False
            }
        }
    },
    "required": ["places"],
    "additionalProperties": False
}

SYSTEM_INSTRUCTION = """
You are a Point-of-Interest discovering agent.
Your parametric memory and training data regarding candidate places, addresses, ratings, and opening hours are considered OUTDATED and STALE.

CRITICAL RULES:
1. You are FORBIDDEN from listing any place purely from your training data.
2. You MUST use Google Maps search queries to discover places and retrieve their current details.
3. All ratings, review counts, and addresses in your final JSON response must match the Google Maps grounding results exactly.
"""

# Verification functions query_place_details and verify_places have been moved to verifier.py

def save_as_pretty_json(jsonl_file):
    base, ext = os.path.splitext(jsonl_file)
    pretty_json_file = base + ".json"
    
    if not os.path.exists(jsonl_file):
        return
        
    records = []
    with open(jsonl_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                if "response_text" in record and isinstance(record["response_text"], str):
                    try:
                        record["response_text"] = json.loads(record["response_text"])
                    except json.JSONDecodeError:
                        pass
                records.append(record)
            except json.JSONDecodeError:
                pass
                
    with open(pretty_json_file, "w") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    print(f"Pretty-printed results saved to {pretty_json_file}")

def run_evaluation(output_file, repetitions, models, effort, queries, workers=5):
    # Initialize Google GenAI client in Vertex AI mode
    client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)
    
    total_calls = len(models) * len(queries) * repetitions
    print(f"Starting 2-Stage Evaluation via Vertex AI ({LOCATION} region):")
    print(f"  Models: {models}")
    print(f"  Thinking Effort: {effort.upper()}")
    print(f"  Queries: {len(queries)} unique queries")
    print(f"  Repetitions: {repetitions} runs per config")
    print(f"  Total Stage A executions: {total_calls}")
    print(f"  Places API Key: {PLACES_API_KEY[:8]}...{PLACES_API_KEY[-4:] if len(PLACES_API_KEY) > 8 else ''}")
    print("-" * 50)
    
    tasks = []
    for model in models:
        for query in queries:
            for i in range(repetitions):
                tasks.append((model, query, i))
                
    pbar = tqdm(total=total_calls)
    lock = threading.Lock()
    success_count = 0
    completed_count = 0
    
    if os.path.dirname(output_file):
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
    with open(output_file, "w") as f:
        def worker(task):
            model, query, i = task
            record = {
                "model": model,
                "effort": effort,
                "query": query,
                "iteration": i,
                "timestamp": time.time(),
                "success": False,
                "latency_stage_a": 0.0,
                "latency_stage_b": 0.0,
                "latency_combined": 0.0,
                "response_text": None,
                "grounding_chunks": [],
                "places_verified": []
            }
            
            try:
                # --- STAGE A: GEMINI GENERATION WITH GOOGLE MAPS GROUNDING ---
                config = types.GenerateContentConfig(
                    tools=[types.Tool(google_maps=types.GoogleMaps())],
                    thinking_config=types.ThinkingConfig(
                        thinking_level=effort.upper()
                    ),
                    system_instruction=SYSTEM_INSTRUCTION,
                    response_mime_type="application/json",
                    response_schema=SCHEMA
                )
                
                start_a = time.time()
                response = client.models.generate_content(
                    model=model,
                    contents=query,
                    config=config
                )
                latency_a = time.time() - start_a
                record["latency_stage_a"] = latency_a
                record["response_text"] = response.text
                
                # Extract reference grounding chunks for reporting/analysis
                grounding_chunks = []
                if response.candidates and response.candidates[0].grounding_metadata:
                    metadata = response.candidates[0].grounding_metadata
                    if metadata.grounding_chunks:
                        grounding_chunks = [chunk.model_dump() for chunk in metadata.grounding_chunks]
                record["grounding_chunks"] = grounding_chunks
                
                # Parse output places and run Stage B
                try:
                    parsed_json = json.loads(response.text)
                    places = parsed_json.get("places", [])
                except Exception as je:
                    raise ValueError(f"Failed to parse generated JSON: {je}. Raw response: {response.text}")
                
                # --- STAGE B: GOOGLE PLACES API VERIFICATION ---
                start_b = time.time()
                verification_results = verify_places(places, PLACES_API_KEY, max_workers=5)
                latency_b = time.time() - start_b
                
                record["latency_stage_b"] = latency_b
                record["latency_combined"] = latency_a + latency_b
                record["places_verified"] = verification_results
                record["success"] = True
                
            except Exception as e:
                record["error"] = str(e)
                # Small backoff on error
                time.sleep(1.0)
                
            with lock:
                nonlocal success_count, completed_count
                completed_count += 1
                if record.get("success"):
                    success_count += 1
                f.write(json.dumps(record) + "\n")
                f.flush()
                success_rate = success_count / completed_count if completed_count > 0 else 0.0
                pbar.set_postfix(success_rate=f"{success_rate:.2%} ({success_count}/{completed_count})")
                pbar.update(1)
                
            # Inter-request rate limit spacing
            time.sleep(0.5)
            
        with ThreadPoolExecutor(max_workers=workers) as executor:
            executor.map(worker, tasks)
            
    pbar.close()
    print(f"\nEvaluation complete. Raw results saved to {output_file}")
    save_as_pretty_json(output_file)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate Gemini 2-Stage Grounding & Verification Framework.")
    parser.add_argument("--output", type=str, help="Path to save the output JSONL raw results.")
    parser.add_argument("--repetitions", "-r", type=int, help="Number of repetitions per query.")
    parser.add_argument("--quick", action="store_true", help="Shortcut to run a fast dry-run with 1 repetition of the first query.")
    parser.add_argument("--model", type=str, help="Override target model to evaluate.")
    parser.add_argument("--effort", type=str, default=DEFAULT_EFFORT, help="Override target thinking effort (low/medium/high).")
    parser.add_argument("--workers", "-w", type=int, default=3, help="Number of concurrent workers for Stage A.")
    
    args = parser.parse_args()
    
    if args.model:
        models_to_eval = [m.strip() for m in args.model.split(",")]
    else:
        models_to_eval = DEFAULT_MODELS
    
    if args.quick:
        output_file = args.output or os.path.join("results", "quick_test_results.jsonl")
        repetitions = 1
        eval_queries = [QUERIES[0]]
    else:
        output_file = args.output or os.path.join("results", "full_eval_results.jsonl")
        repetitions = args.repetitions or 5
        eval_queries = QUERIES
        
    run_evaluation(
        output_file=output_file,
        repetitions=repetitions,
        models=models_to_eval,
        effort=args.effort,
        queries=eval_queries,
        workers=args.workers
    )
