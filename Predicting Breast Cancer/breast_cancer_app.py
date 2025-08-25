# ============================================================
# 🖥️ Breast Cancer Prediction Desktop App - Full Version
# ============================================================

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
from datetime import datetime
import numpy as np

# -----------------------------
# Load model, scaler, features
# -----------------------------
svm_model = joblib.load('svm_breast_cancer_model_top.pkl')
scaler = joblib.load('scaler_top_features.pkl')
top_features = joblib.load('top_features.pkl')
X_train_top = joblib.load('X_train_top_features.pkl')

# SHAP explainer with small background sample for speed
explainer = shap.KernelExplainer(svm_model.predict_proba, X_train_top.sample(50, random_state=42))

# -----------------------------
# Functions
# -----------------------------
def predict_cancer():
    try:
        input_data = {feature: sliders[feature].get() for feature in top_features}
        input_df = pd.DataFrame([input_data], columns=top_features)

        # Scale input
        input_scaled = scaler.transform(input_df)

        # Make prediction
        prediction = svm_model.predict(input_scaled)[0]
        prediction_proba = svm_model.predict_proba(input_scaled)[0][1]
        result_color = "red" if prediction == 1 else "green"
        result_text = f"Prediction: {'Malignant' if prediction==1 else 'Benign'}\nProbability: {prediction_proba:.2f}"
        result_label.config(text=result_text, fg=result_color)

        # Log prediction with timestamp
        log_file = "patient_predictions.csv"
        log_entry = input_df.copy()
        log_entry['Prediction'] = 'Malignant' if prediction==1 else 'Benign'
        log_entry['Probability'] = prediction_proba
        log_entry['Timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if os.path.exists(log_file):
            log_entry.to_csv(log_file, mode='a', header=False, index=False)
        else:
            log_entry.to_csv(log_file, index=False)

        # SHAP explanation
        global fig
        shap_values = explainer.shap_values(input_scaled)[1]  # class 1 (malignant)
        feature_order = np.argsort(np.abs(shap_values[0]))[::-1]
        sorted_features = np.array(top_features)[feature_order]
        sorted_shap_values = shap_values[0][feature_order]

        fig, ax = plt.subplots(figsize=(8,6))
        colors = ['red' if val > 0 else 'green' for val in sorted_shap_values]
        ax.barh(sorted_features, sorted_shap_values, color=colors)
        ax.set_xlabel("SHAP Value")
        ax.set_title("Feature Contributions to Prediction")
        plt.tight_layout()

        # Display SHAP in SHAP tab
        for widget in shap_tab.winfo_children():
            widget.destroy()
        canvas = FigureCanvasTkAgg(fig, master=shap_tab)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

        # Add export button
        tk.Button(shap_tab, text="Export SHAP Plot", command=export_shap_plot, bg="green", fg="white").pack(pady=10)

    except Exception as e:
        messagebox.showerror("Error", f"Invalid input or error: {e}")

def export_shap_plot():
    try:
        if 'fig' not in globals():
            messagebox.showwarning("No Plot", "Please generate a SHAP explanation first.")
            return
        export_file = f"shap_plot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        fig.savefig(export_file, dpi=300, bbox_inches='tight')
        messagebox.showinfo("Exported", f"SHAP plot saved as {export_file}")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to export SHAP plot: {e}")

def view_log():
    log_file = "patient_predictions.csv"
    if not os.path.exists(log_file):
        messagebox.showinfo("No Log", "No predictions have been saved yet.")
        return
    log_df = pd.read_csv(log_file)
    for widget in log_tab.winfo_children():
        widget.destroy()
    txt = scrolledtext.ScrolledText(log_tab, wrap=tk.WORD)
    txt.pack(fill="both", expand=True)
    txt.insert(tk.END, log_df.to_string(index=False))
    txt.configure(state='disabled')

# -----------------------------
# Main Window
# -----------------------------
root = tk.Tk()
root.title("Breast Cancer Prediction App (Full Version)")
root.geometry("950x1400")

# Notebook for tabs
notebook = ttk.Notebook(root)
notebook.pack(fill="both", expand=True)

prediction_tab = tk.Frame(notebook)
shap_tab = tk.Frame(notebook)
log_tab = tk.Frame(notebook)

notebook.add(prediction_tab, text="Prediction")
notebook.add(shap_tab, text="SHAP Explanation")
notebook.add(log_tab, text="Past Predictions")

# -----------------------------
# Prediction Tab
# -----------------------------
tk.Label(prediction_tab, text="Adjust patient measurements using sliders:", font=("Arial", 14)).pack(pady=10)

sliders = {}
feature_groups = {
    "Mean Features": [f for f in top_features if "_mean" in f],
    "Worst Features": [f for f in top_features if "_worst" in f],
    "Standard Error Features": [f for f in top_features if "_se" in f]
}

for group_name, features in feature_groups.items():
    group_frame = tk.LabelFrame(prediction_tab, text=group_name, padx=5, pady=5)
    group_frame.pack(pady=5, fill="x")
    for feature in features:
        frame = tk.Frame(group_frame)
        frame.pack(pady=2, fill="x")
        tk.Label(frame, text=feature, width=25, anchor="w").pack(side="left")
        min_val = float(X_train_top[feature].min())
        max_val = float(X_train_top[feature].max())
        slider = tk.Scale(frame, from_=min_val, to=max_val, orient=tk.HORIZONTAL, resolution=0.01, length=400)
        slider.set((min_val + max_val)/2)
        slider.pack(side="right", fill="x", expand=True)
        sliders[feature] = slider

tk.Button(prediction_tab, text="Predict", command=predict_cancer, bg="blue", fg="white").pack(pady=10)
tk.Button(prediction_tab, text="View Past Predictions", command=view_log, bg="orange", fg="black").pack(pady=5)
result_label = tk.Label(prediction_tab, text="", font=("Arial", 14))
result_label.pack(pady=10)

# -----------------------------
# Run App
# -----------------------------
root.mainloop()
