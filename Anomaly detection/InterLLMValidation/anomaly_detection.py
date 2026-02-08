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
from sklearn.metrics import auc
import pickle
import random
import os
import sys
import math
import random
import pandas as pd
import numpy as np
from itertools import permutations
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerBase

import warnings
warnings.filterwarnings("ignore", category=FutureWarning) 

input_dir = "Input/"

output_dir = "Output/"
output_metrics_dir = output_dir + "Metrics/"
output_plot_dir = output_dir + "Plot/"

training_model = "claude-sonnet-4.5"
all_models = ["claude-sonnet-4.5", "gemini-3.0-pro", "gemini-3.0-flash", "claude-haiku-4.5"]

ae_hyperparameters = {
	"n_components": 2,
	"n_hidden_neurons": 64,
	"optimizer": "adam",
	"batch_size": 8,
	"epochs": 500
}


def read_data(rep_number):
	
	training_set = None
	validation_set = None
	
	test_sets_normal = {}
	test_sets_anomalous = {}
	final_test_sets = {}

	path_train = input_dir + rep_number + "/" + training_model + "/"
	if os.path.exists(path_train):
		for file in os.listdir(path_train):
			if file.startswith("TR"):
				training_set = pd.read_csv(path_train + file)
			elif file.startswith("VAL"):
				validation_set = pd.read_csv(path_train + file)

	for model in all_models:
		path_test = input_dir + rep_number + "/" + model + "/"
		if os.path.exists(path_test):
			for file in os.listdir(path_test):
				if file.startswith("TST"):
					test_sets_normal[model] = pd.read_csv(path_test + file)
				elif file.startswith("ALL"):
					test_sets_anomalous[model] = pd.read_csv(path_test + file)

	if training_set is not None and "Fitness" in training_set.columns:
		training_set = training_set.drop(columns=["Fitness"])
	if validation_set is not None and "Fitness" in validation_set.columns:
		validation_set = validation_set.drop(columns=["Fitness"])
	
	for model in test_sets_normal:
		if "Fitness" in test_sets_normal[model].columns:
			test_sets_normal[model] = test_sets_normal[model].drop(columns=["Fitness"])
	for model in test_sets_anomalous:
		if "Fitness" in test_sets_anomalous[model].columns:
			test_sets_anomalous[model] = test_sets_anomalous[model].drop(columns=["Fitness"])

	all_columns = set()
	if training_set is not None: all_columns.update(training_set.columns)
	if validation_set is not None: all_columns.update(validation_set.columns)
	for df in test_sets_normal.values(): all_columns.update(df.columns)
	for df in test_sets_anomalous.values(): all_columns.update(df.columns)

	def ensure_columns(df, all_columns):
		if df is None: return None
		missing_cols = list(all_columns - set(df.columns))
		if missing_cols:
			zeros_df = pd.DataFrame(0, index=df.index, columns=missing_cols)
			df = pd.concat([df, zeros_df], axis=1)
		return df[sorted(list(all_columns))]

	training_set = ensure_columns(training_set, all_columns)
	validation_set = ensure_columns(validation_set, all_columns)

	for model in all_models:
		norm = ensure_columns(test_sets_normal.get(model), all_columns)
		anom = ensure_columns(test_sets_anomalous.get(model), all_columns)
		
		if norm is not None and anom is not None:
			norm["Label"] = "N"
			anom["Label"] = "A"
			
			final_test_sets[model] = pd.concat([norm, anom], axis=0, ignore_index=True)

	return training_set, validation_set, final_test_sets

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
	
def classify_diagnoses(model, threshold, test_set):

	anomaly_scores = []
	predicted_labels = []
	
	test_set_clean = test_set.copy()
	for col in test_set_clean.columns:
		if test_set_clean[col].dtype.kind in "biufc": 
			test_set_clean[col] = test_set_clean[col].fillna(test_set_clean[col].mean())
		else:
			test_set_clean[col] = test_set_clean[col].fillna("missing")

	test_labels = list(test_set_clean["Label"])
	test_set_no_labels_np = test_set_clean.drop(["Label"], axis=1).to_numpy()
	test_set_no_labels_np = np.nan_to_num(test_set_no_labels_np, nan=0.0, posinf=0.0, neginf=0.0)
	
	reconstructed_test_set_np = model.predict(test_set_no_labels_np, verbose=0)
	for idx,elem in enumerate(reconstructed_test_set_np):
		error = mean_squared_error(test_set_no_labels_np[idx], reconstructed_test_set_np[idx])
		anomaly_scores.append(error)
		if error > threshold:
			predicted_labels.append("A")
		else:
			predicted_labels.append("N")

	return predicted_labels, test_labels, anomaly_scores

def evaluate_performance_metrics(test_labels, predicted_labels, anomaly_scores):
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

		effective_scores = anomaly_scores

		fpr, tpr, thresholds = roc_curve(y_true, effective_scores)
		auc_score = roc_auc_score(y_true, effective_scores)

		performance_metrics["auc"] = auc_score
		performance_metrics["roc_curve"] = {"fpr": fpr, "tpr": tpr, "thresholds": thresholds}

	except Exception as e:
		print(f"Could not compute ROC/AUC: {e}")
		performance_metrics["auc"] = "undefined"
		performance_metrics["roc_curve"] = {"fpr": [], "tpr": [], "thresholds": []}

	'''
	print("The evaluated performance metrics are the following:")
	for k, v in performance_metrics.items():
		if k not in ["roc_curve"]:
			print(f"{k.capitalize()}: {v}")
	'''		

	return performance_metrics

def plot_mean_roc_curves(aggregated_results, mean_fpr):
	plt.rcParams["font.family"] = "Times New Roman"

	COLORS = {
		'gemini-3.0-pro': '#00c6ff',
		'gemini-3.0-flash': '#ffcc67',
		'claude-haiku-4.5': '#ffa6a1',
		'claude-sonnet-4.5': '#b39ddb'
	}
	DEFAULT_COLOR = '#b0bec5'
	
	FS_LABEL = 16
	FS_TICK = 16
	FS_LEGEND = 9

	class HandlerOverlayedLines(HandlerBase):
		def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
			outline, inner = orig_handle
			yc = ydescent + height / 2.0
			x0 = xdescent
			x1 = xdescent + width

			line_outline = Line2D(
				[x0, x1], [yc, yc],
				linestyle=outline.get_linestyle(),
				linewidth=outline.get_linewidth(),
				color=outline.get_color(),
				solid_capstyle='round',
				transform=trans
			)

			line_inner = Line2D(
				[x0, x1], [yc, yc],
				linestyle=inner.get_linestyle(),
				linewidth=inner.get_linewidth(),
				color=inner.get_color(),
				solid_capstyle='round',
				transform=trans
			)

			return [line_outline, line_inner]   
	
	fig, ax = plt.subplots(figsize=(5, 3.5))
	
	model_stats = []
	
	for model_name, data in aggregated_results.items():
		tprs = data["tprs"]
		aucs = data["aucs"]
		
		if not tprs:
			continue
			
		mean_tpr = np.mean(tprs, axis=0)
		mean_tpr[-1] = 1.0
		std_tpr = np.std(tprs, axis=0)
		
		mean_auc = auc(mean_fpr, mean_tpr)
		std_auc = np.std(aucs)
		
		model_stats.append({
			"name": model_name,
			"mean_tpr": mean_tpr,
			"std_tpr": std_tpr,
			"mean_auc": mean_auc,
			"std_auc": std_auc,
			"color": COLORS.get(model_name, DEFAULT_COLOR)
		})

	model_stats.sort(key=lambda x: x["mean_auc"], reverse=True)

	tuple_handles = []
	tuple_labels = []

	for stat in model_stats:
		
		tprs_upper = np.minimum(stat["mean_tpr"] + stat["std_tpr"], 1)
		tprs_lower = np.maximum(stat["mean_tpr"] - stat["std_tpr"], 0)
		
		ax.fill_between(
			mean_fpr, 
			tprs_lower, 
			tprs_upper, 
			color=stat["color"], 
			alpha=0.2, 
			zorder=1 
		)

		line_outline, = ax.plot(
			mean_fpr, stat["mean_tpr"],
			linewidth=8,
			color=stat["color"],
			solid_capstyle='round',
			zorder=2
		)
		
		line_inner, = ax.plot(
			mean_fpr, stat["mean_tpr"],
			linewidth=2,
			color='black',
			solid_capstyle='round',
			zorder=3
		)
		
		label_str = f"{stat['name']} (AUC = {stat['mean_auc']*100:.2f} $\pm$ {stat['std_auc']*100:.2f}%)"
		
		tuple_handles.append((line_outline, line_inner))
		tuple_labels.append(label_str)

	ax.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=2, zorder=1)
	
	handler_map = {tuple: HandlerOverlayedLines()}
	
	ax.legend(
		tuple_handles,
		tuple_labels,
		loc='lower right',
		frameon=True,
		prop={'weight': 'bold', 'size': FS_LEGEND},
		handler_map=handler_map
	)

	ax.set_xlabel('FPR', fontweight='bold', fontsize=FS_LABEL)
	ax.set_ylabel('TPR', fontweight='bold', fontsize=FS_LABEL)
	
	ticks = np.linspace(0, 1, 6)
	tick_labels = [f"{int(x*100)}%" for x in ticks]
	
	ax.set_xticks(ticks)
	ax.set_xticklabels(tick_labels, fontweight='bold', fontsize=FS_TICK)
	
	ax.set_yticks(ticks)
	ax.set_yticklabels(tick_labels, fontweight='bold', fontsize=FS_TICK)
	
	ax.set_xlim([-0.01, 1.01])
	ax.set_ylim([-0.01, 1.05])
	
	ax.grid(True, linestyle='--', alpha=0.6)
	
	plt.tight_layout()
	
	plt.savefig(output_plot_dir + "ROC_CURVE.pdf", bbox_inches='tight', dpi=300)
	plt.show()
	
def save_auc_metrics(aggregated_results):

    
    stats = []
    for model_name, data in aggregated_results.items():
        aucs = data["aucs"]
        if aucs:
            mean_auc = np.mean(aucs)
            std_auc = np.std(aucs)
            stats.append((model_name, mean_auc, std_auc))

    stats.sort(key=lambda x: x[1], reverse=True)

    with open(output_metrics_dir + "AUC.txt", "w") as f:
        # Write Header
        f.write(f"{'Model':<30} | {'Mean AUC':<10} | {'Std Dev':<10}\n")
        f.write("-" * 60 + "\n")
        
        # Write Rows
        for name, mean, std in stats:
            f.write(f"{name:<30} | {mean:.4f}     | {std:.4f}\n")
            
    print("Done.")
	
repetitions = ["0", "1", "2", "3", "4"]
aggregated_results = {model: {"tprs": [], "aucs": []} for model in all_models}
mean_fpr = np.linspace(0, 1, 100)

print("Starting evaluation across repetitions...")	
	
for rep in repetitions:
	print(f"\n--- Processing Repetition {rep} ---")
	
	training_set, validation_set, test_sets = read_data(rep)
	
	ae_model, threshold = train_ae(training_set, validation_set, ae_hyperparameters["n_components"], ae_hyperparameters["n_hidden_neurons"], ae_hyperparameters["optimizer"], ae_hyperparameters["batch_size"], ae_hyperparameters["epochs"])
	
	for model in all_models:
		predicted_labels, test_labels, anomaly_scores = classify_diagnoses(
			ae_model, threshold, test_sets[model]
		)
			
		metrics = evaluate_performance_metrics(test_labels, predicted_labels, anomaly_scores)
			
		fpr = metrics["roc_curve"]["fpr"]
		tpr = metrics["roc_curve"]["tpr"]
		roc_auc = metrics["auc"]
			
		if len(fpr) > 0 and isinstance(roc_auc, (int, float)):
			interp_tpr = np.interp(mean_fpr, fpr, tpr)
			interp_tpr[0] = 0.0
				
			aggregated_results[model]["tprs"].append(interp_tpr)
			aggregated_results[model]["aucs"].append(roc_auc)
				
			print(f"  {model} (Rep {rep}): AUC = {roc_auc:.4f}")

plot_mean_roc_curves(aggregated_results, mean_fpr)
save_auc_metrics(aggregated_results)
	
	









