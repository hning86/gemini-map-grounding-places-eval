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
                            "formatted_address": res_data["result"].get("formatted_address", ""),
                            "place_id": res_data["result"].get("place_id", "")
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
                        "formatted_address": res_data.get("formattedAddress", ""),
                        "place_id": res_data.get("id", "")
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
    """Run Stage B verification for a list of places in parallel, verifying both PlaceId and CID."""
    results = []
    
    def verify_single_place(place):
        title = place.get("title", "").strip()
        generated_pid = place.get("PlaceId", "").strip()
        generated_cid = place.get("CID", "").strip()
        
        result = {
            "title": title,
            "generated_place_id": generated_pid,
            "generated_cid": generated_cid,
            "status_place_id": "MISSING",
            "status_cid": "MISSING",
            "fuzzy_score_place_id": 0.0,
            "fuzzy_score_cid": 0.0,
            "retrieved_name_place_id": None,
            "retrieved_name_cid": None,
            "retrieved_address_place_id": None,
            "retrieved_address_cid": None,
            "cid_resolved_place_id": None,
            "latency": 0.0,
            "verified_place_id": False,
            "verified_cid": False,
            "verified": False,
            "matching_ids": False
        }
        
        start_time = time.time()
        
        # 1. Verify PlaceID
        if generated_pid:
            res_pid, lat_pid = query_place_details(generated_pid, api_key)
            status_pid = res_pid.get("status", "ERROR")
            result["status_place_id"] = status_pid
            if status_pid == "OK" and "result" in res_pid:
                retrieved_name = res_pid["result"].get("name", "")
                result["retrieved_name_place_id"] = retrieved_name
                result["retrieved_address_place_id"] = res_pid["result"].get("formatted_address", "")
                
                score = fuzz.token_sort_ratio(title.lower(), retrieved_name.lower())
                result["fuzzy_score_place_id"] = score
                if score >= 85.0:
                    result["verified_place_id"] = True
            elif status_pid == "NOT_FOUND":
                result["error_message_place_id"] = "Invalid or stale Place ID."
            elif "error_message" in res_pid:
                result["error_message_place_id"] = res_pid["error_message"]
        
        # 2. Verify CID
        if generated_cid:
            res_cid, lat_cid = query_place_details(generated_cid, api_key)
            status_cid = res_cid.get("status", "ERROR")
            result["status_cid"] = status_cid
            if status_cid == "OK" and "result" in res_cid:
                retrieved_name = res_cid["result"].get("name", "")
                result["retrieved_name_cid"] = retrieved_name
                result["retrieved_address_cid"] = res_cid["result"].get("formatted_address", "")
                result["cid_resolved_place_id"] = res_cid["result"].get("place_id", "")
                
                score = fuzz.token_sort_ratio(title.lower(), retrieved_name.lower())
                result["fuzzy_score_cid"] = score
                if score >= 85.0:
                    result["verified_cid"] = True
            elif status_cid == "NOT_FOUND":
                result["error_message_cid"] = "Invalid or stale CID."
            elif "error_message" in res_cid:
                result["error_message_cid"] = res_cid["error_message"]
                
        result["latency"] = time.time() - start_time
        
        # 3. Check matching and overall verification
        if result["verified_place_id"] and result["verified_cid"]:
            if result["cid_resolved_place_id"] and result["generated_place_id"]:
                if result["cid_resolved_place_id"] == result["generated_place_id"]:
                    result["matching_ids"] = True
            result["verified"] = True
        elif result["verified_cid"]:
            result["verified"] = True
        elif result["verified_place_id"]:
            result["verified"] = True
            
        return result

    # Execute Place API details queries in parallel using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(verify_single_place, p) for p in places]
        for fut in futures:
            results.append(fut.result())
            
    return results
