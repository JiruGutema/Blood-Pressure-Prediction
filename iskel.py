import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt

class BloodPressurePredictorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Advanced Blood Pressure Prediction")
        self.root.geometry("600x600")

        # Styling
        style = ttk.Style()
        style.configure("TLabel", font=("Helvetica", 12), padding=10)
        style.configure("TButton", font=("Helvetica", 12), padding=5)

        # Generate synthetic dataset
        self.create_dataset()
        self.train_model()

        # Prediction history
        self.prediction_history = []

        # Input fields
        self.create_widgets()

    def create_dataset(self):
        # Generate synthetic dataset
        data = {
            "Age": np.random.randint(20, 70, 200),
            "BMI": np.random.uniform(18.5, 35.0, 200),
            "Heart_Rate": np.random.randint(60, 100, 200),
            "Weight": np.random.uniform(50, 100, 200),
            "Cholesterol": np.random.randint(150, 300, 200),
            "Physical_Activity": np.random.choice(["Low", "Moderate", "High"], 200),
            "Blood_Pressure": np.random.randint(90, 180, 200),
        }
        self.df = pd.DataFrame(data)

        # Encode categorical data
        self.df["Physical_Activity"] = self.df["Physical_Activity"].map({"Low": 0, "Moderate": 1, "High": 2})

        # Feature and target separation
        self.X = self.df[["Age", "BMI", "Heart_Rate", "Weight", "Cholesterol", "Physical_Activity"]]
        self.y = self.df["Blood_Pressure"]

        # Feature scaling
        self.scaler = StandardScaler()
        self.X_scaled = self.scaler.fit_transform(self.X)  # Standardization (mean = 0, std = 1)

    def train_model(self):
        # Splitting the dataset
        X_train, X_test, y_train, y_test = train_test_split(self.X_scaled, self.y, test_size=0.2, random_state=42)

        # Train Random Forest Regressor (Regression model)
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)

        # Cross-validation
        cv_scores = cross_val_score(self.model, self.X_scaled, self.y, cv=5, scoring='r2')
        self.cv_mean = np.mean(cv_scores)  # Mean of R-squared scores from cross-validation
        self.cv_std = np.std(cv_scores)    # Standard deviation of R-squared scores

        # Predict and evaluate
        y_pred = self.model.predict(X_test)
        self.mse = mean_squared_error(y_test, y_pred)  # Mean Squared Error (MSE)
        self.mae = mean_absolute_error(y_test, y_pred)  # Mean Absolute Error (MAE)
        self.r2 = r2_score(y_test, y_pred)              # R-squared score

        # Calculate variance and standard deviation
        self.variance = np.var(y_test)  # Variance of the actual blood pressure values
        self.std_dev = np.std(y_test)    # Standard deviation of the actual blood pressure values

        # Coefficient of Variation (CV) = (Standard Deviation / Mean) * 100
        self.coeff_of_variation = (self.std_dev / np.mean(y_test)) * 100

    def create_widgets(self):
        frame = ttk.Frame(self.root, padding="20")
        frame.pack(expand=True, fill='both')

        # Title Label
        ttk.Label(frame, text="Blood Pressure Prediction", font=("Helvetica", 16)).grid(row=0, column=0, columnspan=2, pady=10)

        # Input fields (Group them)
        input_frame = ttk.LabelFrame(frame, text="Input Data", padding="10")
        input_frame.grid(row=1, column=0, columnspan=2, pady=10, sticky='ew')

        self.create_input_field(input_frame, "Age", 0)
        self.create_input_field(input_frame, "BMI", 1)
        self.create_input_field(input_frame, "Heart Rate", 2)
        self.create_input_field(input_frame, "Weight", 3)
        self.create_input_field(input_frame, "Cholesterol", 4)
        self.create_dropdown(input_frame, "Physical Activity", ["Low", "Moderate", "High"], 5)

        # Buttons (Group them)
        button_frame = ttk.Frame(frame)
        button_frame.grid(row=2, column=0, columnspan=2, pady=10)

        ttk.Button(button_frame, text="Predict", command=self.predict_blood_pressure).grid(row=0, column=0, padx=5)
        ttk.Button(button_frame, text="Show Metrics", command=self.show_metrics).grid(row=0, column=1, padx=5)
        ttk.Button(button_frame, text="Show Feature Importance", command=self.show_feature_importance).grid(row=0, column=2, padx=5)

        # Save/Load Model Buttons
        model_frame = ttk.LabelFrame(frame, text="Model Management", padding="10")
        model_frame.grid(row=3, column=0, columnspan=2, pady=10, sticky='ew')

        ttk.Button(model_frame, text="Save Model", command=self.save_model).grid(row=0, column=0, padx=5)
        ttk.Button(model_frame, text="Load Model", command=self.load_model).grid(row=0, column=1, padx=5)

        # Export History Button
        ttk.Button(frame, text="Export History", command=self.export_history).grid(row=4, column=0, columnspan=2, pady=10)

        # Help Button
        ttk.Button(frame, text="Help", command=self.show_help).grid(row=5, column=0, columnspan=2, pady=5)

    def create_input_field(self, frame, label_text, row):
        ttk.Label(frame, text=f"{label_text}:").grid(row=row, column=0, sticky=tk.W, pady=5)
        entry = ttk.Entry(frame)
        entry.grid(row=row, column=1, pady=5)
        setattr(self, f"{label_text.lower().replace(' ', '_')}_entry", entry)

    def create_dropdown(self, frame, label_text, options, row):
        ttk.Label(frame, text=f"{label_text}:").grid(row=row, column=0, sticky=tk.W, pady=5)
        dropdown = ttk.Combobox(frame, values=options, state="readonly")
        dropdown.grid(row=row, column=1, pady=5)
        dropdown.set(options[0])
        setattr(self, f"{label_text.lower().replace(' ', '_')}_dropdown", dropdown)

    def predict_blood_pressure(self):
        try:
            inputs = {
                "Age": float(self.age_entry.get()),
                "BMI": float(self.bmi_entry.get()),
                "Heart_Rate": float(self.heart_rate_entry.get()),
                "Weight": float(self.weight_entry.get()),
                "Cholesterol": float(self.cholesterol_entry.get()),
                "Physical_Activity": self.physical_activity_dropdown.get(),
            }
            inputs["Physical_Activity"] = {"Low": 0, "Moderate": 1, "High": 2}[inputs["Physical_Activity"]]

            new_data = pd.DataFrame([inputs])
            new_data_scaled = self.scaler.transform(new_data)

            predicted_bp = self.model.predict(new_data_scaled)
            self.prediction_history.append(predicted_bp[0])
            messagebox.showinfo("Prediction", f"Predicted Blood Pressure: {predicted_bp[0]:.2f}")

        except ValueError as ve:
            messagebox.showerror("Input Error", f"Invalid input: {ve}")
        except Exception as e:
            messagebox.showerror("Error", f"An unexpected error occurred: {e}")

    def save_model(self):
        try:
            with open("blood_pressure_model.pkl", "wb") as file:
                pickle.dump(self.model, file)
            messagebox.showinfo("Info", "Model saved successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save model: {e}")

    def load_model(self):
        try:
            with open("blood_pressure_model.pkl", "rb") as file:
                self.model = pickle.load(file)
            messagebox.showinfo("Info", "Model loaded successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {e}")

    def show_metrics(self):
        metrics = (f"Mean Squared Error: {self.mse:.2f}\n"
                   f"Mean Absolute Error: {self.mae:.2f}\n"
                   f"R-squared Score: {self.r2:.2f}\n"
                   f"Cross-Validation R2: {self.cv_mean:.2f} ± {self.cv_std:.2f}\n"
                   f"Variance: {self.variance:.2f}\n"
                   f"Standard Deviation: {self.std_dev:.2f}\n"
                   f"Coefficient of Variation: {self.coeff_of_variation:.2f}%")
        messagebox.showinfo("Model Metrics", metrics)

    def show_feature_importance(self):
        importances = self.model.feature_importances_
        feature_names = ["Age", "BMI", "Heart Rate", "Weight", "Cholesterol", "Physical Activity"]
        plt.figure(figsize=(10, 6))
        plt.barh(feature_names, importances, color="skyblue")
        plt.xlabel("Feature Importance")
        plt.ylabel("Features")
        plt.title("Feature Importance in Blood Pressure Prediction")
        plt.show()

    def export_history(self):
        if self.prediction_history:
            history_df = pd.DataFrame({"Prediction": self.prediction_history})
            history_df.to_csv("prediction_history.csv", index=False)
            messagebox.showinfo("Export", "Prediction history exported to prediction_history.csv")
        else:
            messagebox.showwarning("Warning", "No prediction history to export.")

    def show_help(self):
        help_text = ("This application predicts blood pressure based on various health metrics.\n\n"
                     "Input the following data:\n"
                     "- Age: Your age in years.\n"
                     "- BMI: Body Mass Index.\n"
                     "- Heart Rate: Beats per minute.\n"
                     "- Weight: Your weight in kg.\n"
                     "- Cholesterol: Cholesterol level.\n"
                     "- Physical Activity: Select your level of physical activity.\n\n"
                     "Click 'Predict' to see your predicted blood pressure.")
        messagebox.showinfo("Help", help_text)

if __name__ == "__main__":
    root = tk.Tk()
    app = BloodPressurePredictorApp(root)
    root.mainloop()
