from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from tensorflow.keras.layers import Dense,Input,Concatenate
from tensorflow.keras.models import Model,load_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras import backend as K
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from sklearn.decomposition import SparsePCA
from sklearn.cluster import DBSCAN
from sklearn.metrics import roc_curve, roc_auc_score
import pickle
import random
import os
import sys
import math
import random
import pandas as pd
import numpy as np
from itertools import permutations

import warnings
warnings.filterwarnings("ignore", category=FutureWarning) 

input_dir = "Input/"

output_dir = "Output/"
output_metrics_dir = output_dir + "Metrics/"
output_model_dir = output_dir + "Model/"


dbscan_hyperparameters = {
	"eps": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
	"min_samples": [1, 2, 3, 4, 5, 6]
}

ae_hyperparameters = {
	"n_components": [2, 4, 8, 16],
	"n_hidden_neurons": [32, 64, 128, 256],
	"optimizer": ["adam", "rmsprop", "SGD"],
	"batch_size": [8, 16, 32, 64],
	"epochs": [100, 250, 500]
}

ft_hyperparameters = {
	"thresholding": ["min", "mean", "median", "max"]
}


def read_data(anomaly_type, ad_method, model, rep_number):

	training_set = None
	validation_set = None
	normal_test_set = None
	anomalous_test_set = None

	for file in os.listdir(input_dir + rep_number + "/" + model):
		if file.split(".csv")[0] == "TR":
			training_set = pd.read_csv(input_dir + rep_number + "/" + model + "/" + file)
		elif file.split(".csv")[0] == "VAL":
			validation_set = pd.read_csv(input_dir + rep_number + "/" + model + "/" + file)	
		elif file.split(".csv")[0] == "TST":
			normal_test_set = pd.read_csv(input_dir + rep_number + "/" + model + "/" + file)
		elif file.split(".csv")[0] == anomaly_type:
			anomalous_test_set = pd.read_csv(input_dir + rep_number + "/" + model + "/" + file)
			
	if ad_method != "ft":
		for df_name in ["training_set", "validation_set", "normal_test_set", "anomalous_test_set"]:
			df = locals()[df_name]
			if "Fitness" in df.columns:
				df = df.drop(columns=["Fitness"])
			locals()[df_name] = df

		all_columns = set(training_set.columns) | set(validation_set.columns) | set(normal_test_set.columns) | set(anomalous_test_set.columns)

		def ensure_columns(df, all_columns):
			for col in all_columns:
				if col not in df.columns:
					df[col] = 0
			return df[list(all_columns)]

		training_set = ensure_columns(training_set, all_columns)
		validation_set = ensure_columns(validation_set, all_columns)
		normal_test_set = ensure_columns(normal_test_set, all_columns)
		anomalous_test_set = ensure_columns(anomalous_test_set, all_columns)
	else:
		training_set = training_set[["Fitness"]]
		validation_set = validation_set[["Fitness"]]
		normal_test_set = normal_test_set[["Fitness"]]
		anomalous_test_set = anomalous_test_set[["Fitness"]]
	
	normal_test_set["Label"] = ["N"]*len(normal_test_set)
	anomalous_test_set["Label"] = ["A"]*len(anomalous_test_set)
	test_set = pd.concat([normal_test_set, anomalous_test_set], axis=0, ignore_index=True)

	return training_set, validation_set, test_set

def normalize_dataset(dataset, reuse_parameters, normalization_parameters_in, normalization_technique):
	
	normalized_dataset = dataset.copy()
	normalization_parameters = {}
	if reuse_parameters == 0:
		if normalization_technique == "zscore":
			for column in normalized_dataset:
				column_values = normalized_dataset[column].values
				if np.any(column_values) == True:
					column_values_mean = np.mean(column_values)
					column_values_std = np.std(column_values)
					if column_values_std != 0:
						column_values = (column_values - column_values_mean)/column_values_std
				else:
					column_values_mean = 0
					column_values_std = 0
				normalized_dataset[column] = column_values
				normalization_parameters[column+"_mean"] = column_values_mean
				normalization_parameters[column+"_std"] = column_values_std
		elif normalization_technique == "min-max":
			column_intervals = get_intervals(dataset)
			for column in normalized_dataset:
				column_data = normalized_dataset[column].tolist()
				intervals = column_intervals[column]
				if intervals[0] != intervals[1]:
					for idx,sample in enumerate(column_data):
						column_data[idx] = (sample-intervals[0])/(intervals[1]-intervals[0])
					
				normalized_dataset[column] = column_data

			for column in column_intervals:
				normalization_parameters[column+"_min"] = column_intervals[column][0]
				normalization_parameters[column+"_max"] = column_intervals[column][1]			
	else:
		if normalization_technique == "zscore":
			for label in normalized_dataset:
				mean = normalization_parameters_in[label+"_mean"]
				std = normalization_parameters_in[label+"_std"]
				parameter_values = normalized_dataset[label].values
				if std != 0:
					parameter_values = (parameter_values - float(mean))/float(std)
				normalized_dataset[label] = parameter_values
		elif normalization_technique == "min-max":
			for label in normalized_dataset:
				min = normalization_parameters_in[label+"_min"]
				max = normalization_parameters_in[label+"_max"]
				parameter_values = normalized_dataset[label].values
				if min != max:
					for idx,sample in enumerate(parameter_values):
						parameter_values[idx] = (sample-min)/(max-min)
				normalized_dataset[label] = parameter_values			
	
	return normalized_dataset, normalization_parameters	

def get_intervals(timeseries):

	intervals = {}
	
	columns = list(timeseries.columns)
	for column in columns:
		intervals[column] = [9999999999, -9999999999]
	for column in timeseries:
		temp_max = timeseries[column].max()
		temp_min = timeseries[column].min()
		if intervals[column][0] > temp_min:
			intervals[column][0] = temp_min
		if intervals[column][1] < temp_max:
			intervals[column][1] = temp_max

	return intervals	

def train_ae(training_set, validation_set, n_components, n_hidden_neurons, optimizer, batch_size, epochs):
	model = None
	threshold=0.0
	training_set_np = training_set.to_numpy()
	training_set_np = training_set_np.astype('float32')
	validation_set_np = validation_set.to_numpy()
	validation_set_np = validation_set_np.astype('float32')
	model = autoencoder(n_hidden_neurons, n_components, len(list(training_set.columns)), optimizer)
	model.fit(training_set_np,training_set_np, epochs=epochs, batch_size=batch_size, shuffle=True, verbose=1)
	reconstructed_validation_set_np = model.predict(validation_set_np)
	threshold = mean_squared_error(validation_set_np, reconstructed_validation_set_np)
	return model, threshold

def autoencoder(hidden_neurons, latent_code_dimension, input_dimension, optimizer):
	input_layer = Input(shape=(input_dimension,))
	encoder = Dense(hidden_neurons,activation="relu")(input_layer)
	code = Dense(latent_code_dimension)(encoder)
	decoder = Dense(hidden_neurons,activation="relu")(code)
	output_layer = Dense(input_dimension,activation="linear")(decoder)
	model = Model(inputs=[input_layer],outputs=[output_layer])
	model.compile(optimizer=optimizer,loss="mse")
	return model	
	
def train_dbscan(training_set, validation_set, eps, min_samples):

	model = {}
	
	temp_training_set = training_set.copy()
	temp_validation_set = validation_set.copy()

	model = DBSCAN(eps=eps, min_samples=min_samples).fit(temp_training_set)
	cluster_labels = model.labels_
	temp_training_set["Cluster"] = cluster_labels
	used = set();
	clusters = [x for x in cluster_labels if x not in used and (used.add(x) or True)]

	instances_sets = {}
	centroids = {}
			
	for cluster in clusters:
		instances_sets[cluster] = []
		centroids[cluster] = []

	temp = temp_training_set
	for index, row in temp.iterrows():
		instances_sets[int(row["Cluster"])].append(row.values.tolist())

	n_features_per_instance = len(instances_sets[list(instances_sets.keys())[0]][0])-1
			
	for instances_set_label in instances_sets:
		instances = instances_sets[instances_set_label]
		for idx, instance in enumerate(instances):
			instances[idx] = instance[0:n_features_per_instance]
		for i in range(0,n_features_per_instance):
			values = []
			for instance in instances:
				values.append(instance[i])
			centroids[instances_set_label].append(np.mean(values))
		
	model = centroids
	
	clusters = []
	for index, instance in temp_validation_set.iterrows():
		min_value = float('inf')
		min_centroid = -1
		for centroid in centroids:
			centroid_coordinates = np.array([float(i) for i in centroids[centroid]])
			dist = np.linalg.norm(instance.values-centroid_coordinates)
			if dist<min_value:
				min_value = dist
				min_centroid = centroid
		clusters.append(min_centroid)
		
	temp_validation_set["Cluster"] = clusters
	distances = []
	for index, instance in temp_validation_set.iterrows():
		if instance["Cluster"] != -1:
			instance = np.array([float(i) for i in instance])
			instance_cluster = int(instance[-1])
			centroid_coordinates = np.array([float(i) for i in model[instance_cluster]])
			instance = np.delete(instance, len(instance)-1)
			distances.append(np.linalg.norm(instance-centroid_coordinates))
	
	try:
		threshold = max(distances)
	except:
		threshold = 0.0

	return model, threshold

def train_ft(training_set, validation_set, thresholding):
	model = None
	threshold = 0.0
	training_set_np = training_set.to_numpy()
	training_set_np = training_set_np.astype('float32')
	validation_set_np = validation_set.to_numpy()
	validation_set_np = validation_set_np.astype('float32')
	if thresholding == "min":
		threshold = min(validation_set_np)
	elif threshold == "mean":
		threshold = sum(validation_set_np)/len(validation_set_np)
	elif threshold == "median":
		threshold = np.median(validation_set_np)
	elif threshold == "max":
		threshold = max(validation_set_np)
		
	return model, threshold	

def classify_diagnoses(model, threshold, test_set, ad_method):

	anomaly_scores = []
	predicted_labels = []
	
	test_set_clean = test_set.copy()
	for col in test_set_clean.columns:
		if test_set_clean[col].dtype.kind in "biufc":  # numeric columns
			test_set_clean[col] = test_set_clean[col].fillna(test_set_clean[col].mean())
		else:
			test_set_clean[col] = test_set_clean[col].fillna("missing")

	test_labels = list(test_set_clean["Label"])
	test_set_no_labels_np = test_set_clean.drop(["Label"], axis=1).to_numpy()
	test_set_no_labels_np = np.nan_to_num(test_set_no_labels_np, nan=0.0, posinf=0.0, neginf=0.0)
	
	if ad_method == "ae":
		reconstructed_test_set_np = model.predict(test_set_no_labels_np, verbose=0)
		for idx,elem in enumerate(reconstructed_test_set_np):
			error = mean_squared_error(test_set_no_labels_np[idx], reconstructed_test_set_np[idx])
			anomaly_scores.append(error)
			if error > threshold:
				predicted_labels.append("A")
			else:
				predicted_labels.append("N")

	elif ad_method == "dbscan":
		for idx, elem in enumerate(test_set_no_labels_np):
			min_value = float('inf')
			min_centroid = -1
			for centroid in model:
				centroid_coordinates = np.array([float(i) for i in model[centroid]])
				dist = np.linalg.norm(elem-centroid_coordinates)
				if dist<min_value:
					min_value = dist
					min_centroid = centroid
			anomaly_scores.append(dist)	
			if dist > threshold:
				predicted_labels.append("A")
			else:
				predicted_labels.append("N")
				
	elif ad_method == "ft":
		for idx, row in test_set.iterrows():
			anomaly_scores.append(row["Fitness"])	
			if row["Fitness"] >= threshold:
				predicted_labels.append("N")
			else:
				predicted_labels.append("A")			

	return predicted_labels, test_labels, anomaly_scores

def save_data(encoded_el):

	for type in encoded_el:
		if type == "Training":
			encoded_el[type].to_csv(output_data_dir + "TrainingData.csv", index=False)
		else:
			encoded_el[type].to_csv(output_data_dir + "TestData.csv", index=False)

	return None

def write_model(model, threshold, fd_method):

	if fd_method == "ae":
		model.save(output_model_dir + "ae.keras")
		file = open(output_model_dir + "threshold.txt", "w")
		file.write(str(threshold))
		file.close()

	return None
	
def write_metrics(performance_metrics, ad_method, best_parameter_set):
	global output_metrics_dir  # use globally defined variable

	# Ensure the directory exists
	os.makedirs(output_metrics_dir, exist_ok=True)

	output_path = os.path.join(output_metrics_dir, "Metrics.txt")
	with open(output_path, "w") as file:
		file.write("Accuracy: " + str(performance_metrics.get("accuracy", "N/A")) + "\n")
		file.write("Precision: " + str(performance_metrics.get("precision", "N/A")) + "\n")
		file.write("Recall: " + str(performance_metrics.get("recall", "N/A")) + "\n")
		file.write("F1: " + str(performance_metrics.get("f1", "N/A")) + "\n")
		file.write("AUC: " + str(performance_metrics.get("auc", "N/A")) + "\n")
		file.write("TN: " + str(performance_metrics.get("tn", "N/A")) + "\n")
		file.write("TP: " + str(performance_metrics.get("tp", "N/A")) + "\n")
		file.write("FN: " + str(performance_metrics.get("fn", "N/A")) + "\n")
		file.write("FP: " + str(performance_metrics.get("fp", "N/A")) + "\n")

		# ✅ Safe check for ROC data
		if (
			"roc_curve" in performance_metrics
			and isinstance(performance_metrics["roc_curve"], dict)
			and "fpr" in performance_metrics["roc_curve"]
			and len(performance_metrics["roc_curve"]["fpr"]) > 0
		):
			roc_data = performance_metrics["roc_curve"]

			# ✅ Invert thresholds if ad_method == "ft"
			thresholds = roc_data["thresholds"]
			if ad_method == "ft":
				thresholds = -1 * np.array(thresholds)

			file.write("\nROC Curve:\n")
			for f, t, thr in zip(roc_data["fpr"], roc_data["tpr"], thresholds):
				file.write(f"FPR: {f:.4f}, TPR: {t:.4f}, Threshold: {thr:.4f}\n")
				
	output_path = os.path.join(output_metrics_dir, "Parameter_set.txt")			
	
	with open(output_path, "w") as file:
		for parameter in best_parameter_set:
			file.write(parameter + ": " + str(best_parameter_set[parameter]) + "\n")

	print(f"✅ Metrics written to {output_path}")

def evaluate_performance_metrics(test_labels, predicted_labels, anomaly_scores, ad_method):
	performance_metrics = {}
	tp = tn = fp = fn = 0

	for idx, label in enumerate(test_labels):
		if predicted_labels[idx] == "N" and predicted_labels[idx] == test_labels[idx]:
			tn += 1
		elif predicted_labels[idx] == "A" and predicted_labels[idx] == test_labels[idx]:
			tp += 1
		elif predicted_labels[idx] == "N" and predicted_labels[idx] != test_labels[idx]:
			fn += 1
		elif predicted_labels[idx] == "A" and predicted_labels[idx] != test_labels[idx]:
			fp += 1

	performance_metrics["tp"] = tp
	performance_metrics["tn"] = tn
	performance_metrics["fp"] = fp
	performance_metrics["fn"] = fn

	try:
		performance_metrics["accuracy"] = (tp + tn) / (tp + tn + fp + fn)
	except ZeroDivisionError:
		print("Accuracy could not be computed because the denominator was 0")
		performance_metrics["accuracy"] = "undefined"

	try:
		performance_metrics["precision"] = tp / (tp + fp)
	except ZeroDivisionError:
		print("Precision could not be computed because the denominator was 0")
		performance_metrics["precision"] = "undefined"

	try:
		performance_metrics["recall"] = tp / (tp + fn)
	except ZeroDivisionError:
		print("Recall could not be computed because the denominator was 0")
		performance_metrics["recall"] = "undefined"

	try:
		performance_metrics["f1"] = 2 * tp / (2 * tp + fp + fn)
	except ZeroDivisionError:
		print("F1 could not be computed because the denominator was 0")
		performance_metrics["f1"] = "undefined"

	try:
		y_true = [1 if lbl == "A" else 0 for lbl in test_labels]

		if ad_method == "ft":
			effective_scores = [-s for s in anomaly_scores]
		else:
			effective_scores = anomaly_scores

		fpr, tpr, thresholds = roc_curve(y_true, effective_scores)
		auc_score = roc_auc_score(y_true, effective_scores)

		performance_metrics["auc"] = auc_score
		performance_metrics["roc_curve"] = {"fpr": fpr, "tpr": tpr, "thresholds": thresholds}

	except Exception as e:
		print(f"Could not compute ROC/AUC: {e}")
		performance_metrics["auc"] = "undefined"
		performance_metrics["roc_curve"] = {"fpr": [], "tpr": [], "thresholds": []}

	print("The evaluated performance metrics are the following:")
	for k, v in performance_metrics.items():
		if k not in ["roc_curve"]:
			print(f"{k.capitalize()}: {v}")

	return performance_metrics
	
try:
	ad_method = sys.argv[1]
	anomaly_type = sys.argv[2]
	model = sys.argv[3]
	rep_number = sys.argv[4]
except:
	print("Input the right number of input arguments.")
	sys.exit()


training_set, validation_set, test_set = read_data(anomaly_type, ad_method, model, rep_number)
best_performing_model = None
best_threshold = 0.0
best_f1 = 0.0
best_performance = None
best_parameter_set = {}

if ad_method == "ae":
	best_parameter_set["n_components"] = None
	best_parameter_set["n_hidden_neurons"] = None
	best_parameter_set["optimizer"] = None
	best_parameter_set["batch_size"] = None
	best_parameter_set["epochs"] = None
	
	for n_components in ae_hyperparameters["n_components"]:
		for n_hidden_neurons in ae_hyperparameters["n_hidden_neurons"]:
			for optimizer in ae_hyperparameters["optimizer"]:
				for batch_size in ae_hyperparameters["batch_size"]:
					for epochs in ae_hyperparameters["epochs"]:
						try:
							model, threshold = train_ae(training_set, validation_set, n_components, n_hidden_neurons, optimizer, batch_size, epochs)
							predicted_labels, test_labels, anomaly_scores = classify_diagnoses(model, threshold, test_set, ad_method)
							performance_metrics = evaluate_performance_metrics(test_labels, predicted_labels, anomaly_scores, ad_method)
							if performance_metrics["f1"] > best_f1:
								best_f1 = performance_metrics["f1"]
								best_performance = performance_metrics.copy()
								best_performing_model = model
								best_threshold = threshold
								
								best_parameter_set["n_components"] = n_components
								best_parameter_set["n_hidden_neurons"] = n_hidden_neurons
								best_parameter_set["optimizer"] = optimizer
								best_parameter_set["batch_size"] = batch_size
								best_parameter_set["epochs"] = epochs
						except:
							continue
							
if ad_method == "dbscan":
	best_parameter_set["eps"] = None
	best_parameter_set["min_samples"] = None

	for eps in dbscan_hyperparameters["eps"]:
		for min_samples in dbscan_hyperparameters["min_samples"]:
			try:
				model, threshold = train_dbscan(training_set, validation_set, eps, min_samples)
				predicted_labels, test_labels, anomaly_scores = classify_diagnoses(model, threshold, test_set, ad_method)
				performance_metrics = evaluate_performance_metrics(test_labels, predicted_labels, anomaly_scores, ad_method)
				if performance_metrics["f1"] > best_f1:
					best_f1 = performance_metrics["f1"]
					best_performance = performance_metrics.copy()
					best_performing_model = model
					best_threshold = threshold
					
					best_parameter_set["eps"] = eps
					best_parameter_set["min_samples"] = min_samples
			except:
				continue
		

if ad_method == "ft":
	best_parameter_set["thresholding"] = None

	for thresholding in ft_hyperparameters["thresholding"]:
		try:
			model, threshold = train_ft(training_set, validation_set, thresholding)
			predicted_labels, test_labels, anomaly_scores = classify_diagnoses(model, threshold, test_set, ad_method)
			performance_metrics = evaluate_performance_metrics(test_labels, predicted_labels, anomaly_scores, ad_method)
			if performance_metrics["f1"] > best_f1:
				best_f1 = performance_metrics["f1"]
				best_performance = performance_metrics.copy()
				best_performing_model = model
				best_parameter_set["thresholding"] = thresholding
		except:
			continue

write_metrics(best_performance, ad_method, best_parameter_set)











