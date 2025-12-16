"""
Function to extract chemotherapy and radiotherapy plans from HGG patient JSON files.
Converts absolute timestamps to days relative to the first visit (day 0).
"""

import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional


def extract_treatment_plans(
    json_path: str,
    kill_rate_per_mg: Optional[float] = None
) -> Tuple[Dict[int, float], List[Tuple[int, int, float]]]:
    """
    Extract radiotherapy and chemotherapy plans from JSON file.
    
    Args:
        json_path: Path to the JSON file containing patient data
        kill_rate_per_mg: Optional conversion factor from chemo dose (mg) to kill_rate.
                         If None, will use dose value directly (may need adjustment).
    
    Returns:
        rt_plan: Dictionary mapping day (relative to first visit) to dose in Gy
        chemo_plan: List of tuples (start_day, end_day, kill_rate)
                   For consecutive days with same dose, they are grouped into ranges.
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Get first visit time (day 0)
    first_visit_time = datetime.fromisoformat(data['visits'][0]['time'])
    
    # Extract radiotherapy plan
    rt_plan = {}
    for rt_entry in data.get('radiotherapy', []):
        rt_time = datetime.fromisoformat(rt_entry['time'])
        days_from_first_visit = (rt_time - first_visit_time).days
        dose = rt_entry['dose']
        rt_plan[days_from_first_visit] = dose
    
    # Extract chemotherapy plan
    # Group consecutive days with same dose into ranges
    chemo_entries = []
    for chemo_entry in data.get('chemotherapy', []):
        chemo_time = datetime.fromisoformat(chemo_entry['time'])
        days_from_first_visit = (chemo_time - first_visit_time).days
        dose = chemo_entry['dose']
        chemo_entries.append((days_from_first_visit, dose))
    
    # Sort by day
    chemo_entries.sort(key=lambda x: x[0])
    
    # Group consecutive days with same dose
    chemo_plan = []
    if chemo_entries:
        current_start = chemo_entries[0][0]
        current_dose = chemo_entries[0][1]
        
        for i in range(1, len(chemo_entries)):
            day, dose = chemo_entries[i]
            
            # If dose changed or gap > 1 day, start new range
            if dose != current_dose or day != chemo_entries[i-1][0] + 1:
                # End previous range
                end_day = chemo_entries[i-1][0]
                # Convert dose to kill_rate (if conversion factor provided, else use dose/1000 as approximation)
                kill_rate = (current_dose / 1000.0) if kill_rate_per_mg is None else (current_dose * kill_rate_per_mg)
                chemo_plan.append((current_start, end_day, kill_rate))
                
                # Start new range
                current_start = day
                current_dose = dose
        
        # Add final range
        end_day = chemo_entries[-1][0]
        kill_rate = (current_dose / 1000.0) if kill_rate_per_mg is None else (current_dose * kill_rate_per_mg)
        chemo_plan.append((current_start, end_day, kill_rate))
    
    return rt_plan, chemo_plan
