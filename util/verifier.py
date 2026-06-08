import time
import json
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor
from rapidfuzz import fuzz

def query_place_details(place_id, api_key):
    """Query Google Places API details for a specific Place ID or CID and measure latency."""
    clean_pid = place_id.strip()
    if clean_pid.lower().startswith("cid:"):
        clean_pid = clean_pid[4:].strip()
        
    start_time = time.time()
    
    # Check if Place ID is a numeric CID
    if clean_pid.isdigit():
        url = f"https://maps.googleapis.com/maps/api/place/details/json?cid={clean_pid}&key={api_key}"
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=5) as response:
                res_data = json.loads(response.read().decode('utf-8'))
                latency = time.time() - start_time
                
                status = res_data.get("status", "ERROR")
                if status == "OK" and "result" in res_data:
                    mapped_result = {
                        "status": "OK",
                        "result": {
                            "name": res_data["result"].get("name", ""),
                            "formatted_address": res_data["result"].get("formatted_address", "")
                        }
                    }
                    return mapped_result, latency
                else:
                    err_msg = res_data.get("error_message", f"Legacy API error status: {status}")
                    api_status = "NOT_FOUND" if status == "NOT_FOUND" else "INVALID_REQUEST"
                    return {"status": api_status, "error_message": err_msg}, latency
        except Exception as e:
            latency = time.time() - start_time
            return {"status": "ERROR", "error_message": str(e)}, latency
            
    # Otherwise, query the New Places API (v1)
    else:
        url = f"https://places.googleapis.com/v1/places/{clean_pid}"
        headers = {
            "X-Goog-Api-Key": api_key,
            "X-Goog-FieldMask": "id,displayName,formattedAddress"
        }
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=5) as response:
                res_data = json.loads(response.read().decode('utf-8'))
                latency = time.time() - start_time
                
                mapped_result = {
                    "status": "OK",
                    "result": {
                        "name": res_data.get("displayName", {}).get("text", ""),
                        "formatted_address": res_data.get("formattedAddress", "")
                    }
                }
                return mapped_result, latency
                
        except urllib.error.HTTPError as e:
            latency = time.time() - start_time
            try:
                err_body = json.loads(e.read().decode('utf-8'))
                err_detail = err_body.get("error", {})
                api_status = err_detail.get("status", "ERROR")
                api_message = err_detail.get("message", str(e))
                
                if api_status == "NOT_FOUND":
                    status = "NOT_FOUND"
                elif api_status == "INVALID_ARGUMENT":
                    status = "INVALID_REQUEST"
                else:
                    status = "ERROR"
                    
                return {"status": status, "error_message": api_message}, latency
            except Exception as parse_err:
                return {"status": "ERROR", "error_message": f"{e} (Failed to parse error body: {parse_err})"}, latency
                
        except Exception as e:
            latency = time.time() - start_time
            return {"status": "ERROR", "error_message": str(e)}, latency

def verify_places(places, api_key, max_workers=5):
    """Run Stage B verification for a list of places in parallel."""
    results = []
    
    def verify_single_place(place):
        title = place.get("title", "").strip()
        generated_pid = place.get("PlaceId", "").strip()
        
        result = {
            "title": title,
            "generated_place_id": generated_pid,
            "status": "MISSING_ID",
            "fuzzy_score": 0.0,
            "retrieved_name": None,
            "retrieved_address": None,
            "latency": 0.0,
            "verified": False
        }
        
        if not generated_pid:
            return result
            
        res_data, latency = query_place_details(generated_pid, api_key)
        result["latency"] = latency
        status = res_data.get("status", "ERROR")
        result["status"] = status
        
        if status == "OK" and "result" in res_data:
            retrieved_name = res_data["result"].get("name", "")
            retrieved_address = res_data["result"].get("formatted_address", "")
            result["retrieved_name"] = retrieved_name
            result["retrieved_address"] = retrieved_address
            
            # Fuzzy match generated title vs. retrieved name
            score = fuzz.token_sort_ratio(title.lower(), retrieved_name.lower())
            result["fuzzy_score"] = score
            
            # A place is verified if Places API returns OK and fuzzy name match is >= 85%
            if score >= 85.0:
                result["verified"] = True
        elif status == "NOT_FOUND":
            result["error_message"] = "The provided Place ID is invalid or stale according to Places API."
        elif "error_message" in res_data:
            result["error_message"] = res_data["error_message"]
            
        return result

    # Execute Place API details queries in parallel using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(verify_single_place, p) for p in places]
        for fut in futures:
            results.append(fut.result())
            
    return results
