🧠 Early Stroke Detection Using AI
📌 Project Overview

Early detection of stroke—especially wake-up strokes occurring during sleep—is critical for reducing mortality and long-term disability.
This project presents an AI-based early stroke detection system that analyzes physiological and hemodynamic signals to predict stroke risk at an early stage.

The system is designed with a machine learning / deep learning pipeline that can be extended to real-time wearable devices for continuous monitoring and emergency alerting.

🎯 Objectives

Detect early signs of stroke using physiological signals

Analyze patterns during sleep (wake-up stroke focus)

Build a scalable AI/ML model for stroke risk prediction

Classify patients into Low, Medium, and High risk categories

Enable future real-time integration with wearable systems

🧠 Key Features

End-to-end ML pipeline: preprocessing → training → evaluation

Risk score prediction based on health parameters

Modular and clean project structure

Easily extendable to Graph Neural Networks (GNN) and real-time data

Research-oriented and internship-ready codebase

🛠️ Tech Stack

Programming Language: Python

Libraries & Tools:

NumPy, Pandas

Scikit-learn

PyTorch

Matplotlib, Seaborn

Concepts Used:

Data Preprocessing

Feature Engineering

Machine Learning / Deep Learning

Model Evaluation

📂 Project Structure
early-stroke-detection/
│
├── data/
│   ├── raw/            # Raw dataset (not uploaded)
│   └── processed/      # Cleaned & processed data
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   └── 03_model_training.ipynb
│
├── src/
│   ├── data_loader.py
│   ├── preprocess.py
│   ├── model.py
│   ├── train.py
│   └── evaluate.py
│
├── models/
│   └── stroke_model.pth
│
├── results/
│   └── metrics & plots
│
├── README.md
├── requirements.txt
└── .gitignore

📊 Dataset Description

The dataset consists of physiological and hemodynamic signals, including:

Heart Rate (HR)

Heart Rate Variability (HRV)

Blood Pressure (Systolic & Diastolic)

SpO₂

Sleep-related parameters

Demographic/clinical attributes (where available)

📌 Dataset is excluded from the repository due to size and privacy constraints.

⚙️ Methodology

Data Cleaning & Preprocessing

Handling missing values

Normalization & feature preparation

Feature Engineering

Extracting relevant health indicators

Model Training

Supervised learning / Deep learning model

Evaluation

Accuracy and performance metrics

Risk Classification

Output mapped to Low / Medium / High stroke risk

📈 Results

The model demonstrates promising performance in distinguishing stroke vs non-stroke patterns

Achieves effective early risk prediction suitable for preventive healthcare applications

📌 Detailed metrics and plots are available in the results/ folder.

🚀 How to Run the Project
1️⃣ Install Dependencies
pip install -r requirements.txt

2️⃣ Preprocess Data
python src/preprocess.py

3️⃣ Train the Model
python src/train.py

4️⃣ Evaluate the Model
python src/evaluate.py

🔮 Future Enhancements

Integration with wearable IoT devices

Real-time stroke risk monitoring

Mobile application dashboard

Graph Neural Network (GNN)-based modeling

Clinical validation with larger datasets
