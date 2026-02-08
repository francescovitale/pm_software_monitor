import pandas as pd
import os
import re
import pprint
import sys
from sklearn.model_selection import train_test_split
import random
import numpy as np
from copy import deepcopy

import pm4py
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.objects.log.util import dataframe_utils
from pm4py.algo.simulation.playout.petri_net import algorithm as simulator
from pm4py.algo.conformance.alignments.petri_net import algorithm as alignments
from pm4py.algo.evaluation.replay_fitness import algorithm as replay_fitness

input_dir = "Input/"
input_eventlogs_dir = input_dir + "EventLogs/"
input_petrinet_dir = input_dir + "PetriNet/"

output_dir = "Diagnoses/"

def read_event_logs():
    event_logs = {}


    for filename in os.listdir(input_eventlogs_dir):
        if filename.endswith(".xes") or filename.endswith(".xes.gz"):
            file_path = os.path.join(input_eventlogs_dir, filename)
            
            try:
                log = pm4py.read_xes(file_path)
                
                event_logs[filename.split(".xes")[0]] = log
                print(f"Successfully loaded: {filename}")
                
            except Exception as e:
                print(f"Failed to read {filename}: {e}")

    return event_logs
	
def build_event_log(cases):
	
	event_log = []
	
	for idx,entries in enumerate(cases):
		caseid = idx
		for idx_ts, entry in enumerate(entries):
			event_timestamp = timestamp_builder(idx_ts)
			event = [caseid, entry, event_timestamp]
			event_log.append(event)
			
	event_log = pd.DataFrame(event_log, columns=['CaseID', 'Event', 'Timestamp'])
	event_log.rename(columns={'Event': 'concept:name'}, inplace=True)
	event_log.rename(columns={'Timestamp': 'time:timestamp'}, inplace=True)
	event_log = dataframe_utils.convert_timestamp_columns_in_df(event_log)
	parameters = {log_converter.Variants.TO_EVENT_LOG.value.Parameters.CASE_ID_KEY: 'CaseID'}
	event_log = log_converter.apply(event_log, parameters=parameters, variant=log_converter.Variants.TO_EVENT_LOG)
		
	return event_log	
	
def timestamp_builder(number):
	
	ss = number
	mm, ss = divmod(ss, 60)
	hh, mm = divmod(mm, 60)
	ignore, hh = divmod(hh, 24)
	
	ss = ss%60
	mm = mm%60
	hh = hh%24
	
	return "1900-01-01T"+str(hh)+":"+str(mm)+":"+str(ss)	

def read_petri_net():

	petri_net = {}

	bpmn_model = pm4py.read_bpmn(input_petrinet_dir + "START_OF_MISSION.bpmn")
	petri_net["network"], petri_net["initial_marking"], petri_net["final_marking"] = pm4py.convert_to_petri_net(bpmn_model)

	return petri_net

def build_dataset(normal_event_logs, validation_split, test_split, petri_net):
    from pm4py.objects.log.obj import EventLog, Trace, Event
    from pm4py.objects.conversion.log import converter as log_converter
    import pandas as pd
    import numpy as np

    # --- Nested Helper Function ---
    def build_event_log(data):
        """
        Converts DataFrames, Lists of Traces, or Lists of Strings into PM4Py EventLog.
        """
        # 1. Handle Pandas DataFrame (Fix for your error)
        if isinstance(data, pd.DataFrame):
            if data.empty:
                return EventLog()
            # Convert DataFrame back to EventLog object
            return log_converter.apply(data, variant=log_converter.Variants.TO_EVENT_LOG)

        # 2. Handle Empty Lists
        if not data:
            return EventLog()

        # 3. Handle List of Traces (Legacy objects)
        if isinstance(data[0], Trace):
            log = EventLog()
            for trace in data:
                log.append(trace)
            return log
        
        # 4. Handle List of List of Strings (From anomaly simulation)
        elif isinstance(data[0], list) or isinstance(data[0], tuple):
            log = EventLog()
            for trace_data in data:
                trace = Trace()
                for activity_name in trace_data:
                    event = Event()
                    event["concept:name"] = activity_name
                    trace.append(event)
                log.append(trace)
            return log
            
        return EventLog()
    # ------------------------------

    anomaly_types = ["MA", "WOA", "UA"]

    normal_event_logs_training = {}
    normal_event_logs_validation = {}
    normal_event_logs_test = {}
    anomalous_event_logs = {}

    print("--- processing normal logs ---")
    
    for model_name, log_obj in normal_event_logs.items():
        print(f"Processing model: {model_name}")

        # LOGIC FIX: If input is DataFrame, we must split by Case ID, not by Row.
        if isinstance(log_obj, pd.DataFrame):
            # Assumes standard PM4Py naming 'case:concept:name'
            case_ids = log_obj["case:concept:name"].unique()
            
            # Split Case IDs
            train_ids_temp, test_ids = train_test_split(case_ids, test_size=test_split, shuffle=True)
            train_ids, val_ids = train_test_split(train_ids_temp, test_size=validation_split, shuffle=True)
            
            # Filter DataFrame based on selected Case IDs
            train_data = log_obj[log_obj["case:concept:name"].isin(train_ids)]
            val_data = log_obj[log_obj["case:concept:name"].isin(val_ids)]
            test_data = log_obj[log_obj["case:concept:name"].isin(test_ids)]

        # Fallback: If input is already an EventLog object (Legacy)
        else:
            train_temp, test_data = train_test_split(log_obj, test_size=test_split, shuffle=True)
            train_data, val_data = train_test_split(train_temp, test_size=validation_split, shuffle=True)

        print(f"  - Train size: {len(train_data) if isinstance(train_data, list) else train_data['case:concept:name'].nunique()}")
        print(f"  - Test size: {len(test_data) if isinstance(test_data, list) else test_data['case:concept:name'].nunique()}")

        # Convert everything to EventLog objects
        normal_event_logs_training[model_name] = build_event_log(train_data)
        normal_event_logs_validation[model_name] = build_event_log(val_data)
        normal_event_logs_test[model_name] = build_event_log(test_data)

    print("--- generating anomalies ---")
    
    net = petri_net["network"]
    im = petri_net["initial_marking"]
    
    all_anomalies_list = []
    
    for anomaly_type in anomaly_types:
        anomalous_event_logs[anomaly_type] = None
        
        simulated_traces = simulator.apply(
            net, 
            im, 
            variant=simulator.Variants.BASIC_PLAYOUT, 
            parameters={simulator.Variants.BASIC_PLAYOUT.value.Parameters.NO_TRACES: 50}
        )
        
        simulated_traces_list = []
        for trace in simulated_traces:
            events = [event["concept:name"] for event in trace]
            simulated_traces_list.append(events)
            
        try:
            anomalous_traces_data, _ = inject_anomalies_poisson(simulated_traces_list, anomaly_type)
        except NameError:
            print("  Warning: 'inject_anomalies_poisson' not defined.")
            anomalous_traces_data = []

        anomalous_event_logs[anomaly_type] = build_event_log(anomalous_traces_data)
        all_anomalies_list.extend(anomalous_traces_data)

    anomalous_event_logs["ALL"] = build_event_log(all_anomalies_list)
        
    return normal_event_logs_training, normal_event_logs_validation, normal_event_logs_test, anomalous_event_logs

def inject_anomalies_poisson(traces, anomaly_type, lam=3.0):
    assert anomaly_type in {"MA", "WOA", "UA"}, "Invalid anomaly type."
    traces_copy = deepcopy(traces)

    unknown_pool = ["X", "Y", "Z", "Q", "R"]
    anomalous_traces = []
    anomaly_counts = []

    for trace in traces_copy:
        t = trace.copy()
        k = np.random.poisson(lam)
        k = max(0, k)
        anomaly_counts.append(k)

        for _ in range(k):
            if anomaly_type == "MA":
                if len(t) > 1:
                    idx = random.randint(1, len(t) - 2) if len(t) > 2 else 0
                    del t[idx]

            elif anomaly_type == "WOA":
                if len(t) > 2:
                    i, j = random.sample(range(len(t)), 2)
                    t[i], t[j] = t[j], t[i]

            elif anomaly_type == "UA":
                idx = random.randint(0, len(t))
                new_act = random.choice(unknown_pool)
                t.insert(idx, new_act)

        anomalous_traces.append(t)

    return anomalous_traces, anomaly_counts

def compute_diagnoses(event_logs, petri_net):

	diagnoses = {}
	
	for model in event_logs:
		diagnoses[model] = None
		event_log_activities = set(get_event_log_activities(event_logs[model]))
		petri_net_activities = set(get_petri_net_activities(petri_net))
		all_activities = sorted(list(petri_net_activities.union(event_log_activities)))
		diagnoses[model] = generate_ab_diagnoses(event_logs[model], petri_net, all_activities)

	return diagnoses
	
def get_event_log_activities(event_log):
	
	activities = []
	for trace in event_log:
		for event in trace:
			if event["concept:name"] not in activities:
				activities.append(event["concept:name"])	
					
	activites = list(set(activities))

	return activities

def get_petri_net_activities(petri_net):
	activities = []
	transitions = list(petri_net["network"]._PetriNet__get_transitions())

	for transition in transitions:
		transition = transition._Transition__get_label()
		if transition != None:
			activities.append(transition)

	return activities

def generate_ab_diagnoses(log, petri_net, activities):

	ab_diagnoses = []

	for trace in log:
		fitness, aligned_traces = compute_fitness(petri_net, [trace])
		temp = [list(aligned_trace.values())[0] for aligned_trace in aligned_traces]
		aligned_traces = temp

		misaligned_activities = compute_misaligned_activities([trace], aligned_traces)

		row = {activity: 0 for activity in activities} 
		for activity, count in misaligned_activities.items():
			row[activity] = count

		row["Fitness"] = fitness
		ab_diagnoses.append(row)

	ab_diagnoses = pd.DataFrame(ab_diagnoses, columns=list(activities) + ["Fitness"])

	return ab_diagnoses

def compute_fitness(petri_net, event_log):

	log_fitness = 0.0
	aligned_traces = None
	parameters = {}
	parameters[log_converter.Variants.TO_EVENT_LOG.value.Parameters.CASE_ID_KEY] = 'CaseID'
	
	aligned_traces = alignments.apply_log(event_log, petri_net["network"], petri_net["initial_marking"], petri_net["final_marking"], parameters=parameters, variant=alignments.Variants.VERSION_STATE_EQUATION_A_STAR)
	log_fitness = replay_fitness.evaluate(aligned_traces, variant=replay_fitness.Variants.ALIGNMENT_BASED)["log_fitness"]
	

	return log_fitness, aligned_traces
	
def compute_misaligned_activities(event_log, aligned_traces):
	
	misaligned_activities = {}
	events = {}
	
	for aligned_trace in aligned_traces:
		for move in aligned_trace:
			log_behavior = move[0]
			model_behavior = move[1]
			if log_behavior != model_behavior:
				
				if log_behavior != None and log_behavior != ">>":
					try:
						events[log_behavior] = events[log_behavior]+1
					except:
						events[log_behavior] = 0
						events[log_behavior] = events[log_behavior]+1
				elif model_behavior != None and model_behavior != ">>":
					
					try:
						events[model_behavior] = events[model_behavior] + 1
					except:
						events[model_behavior] = 0
						events[model_behavior] = events[model_behavior]+1
	while bool(events):
		popped_event = events.popitem()
		if popped_event[1] > 0:
			misaligned_activities[popped_event[0]] = popped_event[1]

	return misaligned_activities

def save_diagnoses(normal_event_logs_training_diagnoses, normal_event_logs_validation_diagnoses, normal_event_logs_test_diagnoses, anomalous_event_logs_diagnoses, rep):

	os.makedirs(output_dir + str(rep), exist_ok=True)

	for model in normal_event_logs_training_diagnoses:
		os.makedirs(output_dir + str(rep) + "/" + model, exist_ok=True)
		normal_event_logs_training_diagnoses[model].to_csv(output_dir + str(rep) + "/" + model + "/TR.csv", index=False)
		normal_event_logs_validation_diagnoses[model].to_csv(output_dir + str(rep) + "/" + model + "/VAL.csv", index=False)
		normal_event_logs_test_diagnoses[model].to_csv(output_dir + str(rep) + "/" + model + "/TST.csv", index=False)
		for anomaly_type in anomalous_event_logs_diagnoses:
			anomalous_event_logs_diagnoses[anomaly_type].to_csv(output_dir + str(rep) + "/" + model + "/" + anomaly_type + ".csv", index=False)

	return None

try:
	n_reps = int(sys.argv[1])
	validation_split = float(sys.argv[2])
	test_split = float(sys.argv[3])
except:
	print("Enter the right number of input arguments.")
	sys.exit()

normal_event_logs = read_event_logs()
petri_net = read_petri_net()

for i in range(0, n_reps):
	normal_event_logs_training, normal_event_logs_validation, normal_event_logs_test, anomalous_event_logs = build_dataset(normal_event_logs, validation_split, test_split, petri_net)
	normal_event_logs_training_diagnoses = compute_diagnoses(normal_event_logs_training, petri_net)
	normal_event_logs_validation_diagnoses = compute_diagnoses(normal_event_logs_validation, petri_net)
	normal_event_logs_test_diagnoses = compute_diagnoses(normal_event_logs_test, petri_net)
	anomalous_event_logs_diagnoses = compute_diagnoses(anomalous_event_logs, petri_net)
	save_diagnoses(normal_event_logs_training_diagnoses, normal_event_logs_validation_diagnoses, normal_event_logs_test_diagnoses, anomalous_event_logs_diagnoses, i)
	
	

