


# Blood Pressure Predictor App

## Overview

The **Blood Pressure Predictor App** is a user-friendly application designed to predict blood pressure based on user-provided health metrics. Utilizing a Random Forest regression model, this app allows users to input their age, BMI, heart rate, weight, cholesterol level, and physical activity level to receive an estimated blood pressure reading. The application also features model management, metrics evaluation, and visualization of feature importance.

## Features

- **User Input**: Enter health metrics to predict blood pressure.
- **Model Training**: Uses a Random Forest regressor for accurate predictions.
- **Metrics Display**: View performance metrics such as Mean Squared Error, R-squared, and Coefficient of Variation.
- **Feature Importance Visualization**: Understand which features most influence the predictions.
- **Model Management**: Save and load trained models for future use.
- **Export Prediction History**: Save your predictions to a CSV file.
- **Help Section**: Guidance on how to use the app effectively.

## Requirements

Before running the application, ensure you have the following installed:

- Python 3.x
- pip (Python package installer)

## Installation

1. **Clone the Repository** (or download the files):
   ```bash
   git clone <repository-url>
   cd BloodPressurePredictorApp
   ```

2. **Create a Virtual Environment**:
   ```bash
   python -m venv venv
   ```

3. **Activate the Virtual Environment**:
   - On Windows:
     ```bash
     venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     source venv/bin/activate
     ```

4. **Install Required Packages**:
   ```bash
   pip install numpy pandas scikit-learn matplotlib
   ```

## Running the Application

After setting up the environment and installing the required packages, you can run the application with:

```bash
python app.py
```

Replace `app.py` with the actual name of your main Python file if it differs.

## Usage

1. Enter your health metrics in the input fields:
   - **Age**: Age in years.
   - **BMI**: Body Mass Index.
   - **Heart Rate**: Beats per minute.
   - **Weight**: Weight in kilograms.
   - **Cholesterol**: Cholesterol level.
   - **Physical Activity**: Choose your activity level (Low, Moderate, High).

2. Click the **Predict** button to receive your estimated blood pressure.

3. Use the **Show Metrics** button to view the performance metrics of the model.

4. Click **Show Feature Importance** to visualize the importance of each feature in the prediction.

5. Save or load models using the **Save Model** and **Load Model** buttons.

6. Export prediction history to a CSV file using the **Export History** button.

7. Access the **Help** section for guidance on using the app.

## Contributing

Contributions are welcome! If you have suggestions for improvements or new features, feel free to create a pull request or open an issue.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments

- Libraries used: `NumPy, Pandas, Scikit-learn, Matplotlib, Tkinter.`


### Instructions for Use

1. **Replace `<repository-url>`** with the actual URL of your GitHub repository or the source from which the application can be cloned.

2. **Adjust any specific instructions** based on your application structure or additional features.

