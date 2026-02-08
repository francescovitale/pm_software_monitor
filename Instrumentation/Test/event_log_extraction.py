import pandas as pd
import os
import re
import pprint
import sys

from pm4py.objects.conversion.log import converter as log_converter
from pm4py.objects.log.util import dataframe_utils
from pm4py.objects.log.exporter.xes import exporter as xes_exporter

input_dir = "Input/ELE/"
output_dir = "Output/ELE/"

def parse_software_log_cases():
    
    software_logs = []
    
    log_pattern = re.compile(
        r'^(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d+)\s+-\s+.*?\s+-\s+.*?\s+-\s+(?P<activity>som\w+)'
    )
    
    for filename in os.listdir(input_dir):
        file_path = os.path.join(input_dir, filename)
        
        if not os.path.isfile(file_path):
            continue
        
        case_id = os.path.splitext(filename)[0]
        
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                match = log_pattern.search(line)
                if match:
                    software_logs.append({
                        "case:concept:name": case_id,
                        "concept:name": match.group("activity"),
                        "time:timestamp": pd.to_datetime(match.group("timestamp"))
                    })
    
    return software_logs
	
def build_event_log(software_logs):
    
    df = pd.DataFrame(software_logs)
    
    if df.empty:
        return None
    
    df = dataframe_utils.convert_timestamp_columns_in_df(df)
    
    event_log = log_converter.apply(
        df,
        variant=log_converter.Variants.TO_EVENT_LOG
    )
    
    return event_log
	
def save_log(event_log):
    
    if event_log is None:
        return None
    
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, "software_log.xes")
    xes_exporter.apply(event_log, output_path)
    
    return None

software_logs = parse_software_log_cases()
event_log = build_event_log(software_logs)
save_log(event_log)	
	

