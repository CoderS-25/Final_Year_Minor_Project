# 🔬 MedCluster AI: Automated Tumor Detection Dashboard

![Project Status](https://img.shields.io/badge/Status-100%25_Complete-brightgreen)
![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-Framework-FF4B4B.svg)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-Machine_Learning-F7931E.svg)

## 📌 Project Overview
This project applies **Unsupervised Machine Learning** to the field of clinical oncology. The objective is to build an AI engine capable of automatically analyzing the physical geometry of breast tumor cells and grouping them into distinct medical profiles (Benign vs. Malignant) **without relying on human-provided diagnostic labels**. 

By utilizing clustering algorithms and dimensionality reduction, this project mathematically proves that Artificial Intelligence can naturally discover the boundary between cancerous and non-cancerous cells purely based on raw physical measurements.

## 🗄️ Dataset
* **Source:** UCI Machine Learning Repository
* **Dataset Name:** Wisconsin Breast Cancer Diagnostic Dataset
* **Features:** 30 numerical features computed from digitized images of a fine needle aspirate (FNA) of a breast mass. Features describe characteristics of the cell nuclei present in the image (e.g., Radius, Texture, Perimeter, Area, Smoothness, Compactness, Concavity).

## ⚙️ Algorithms & Methodology
This project implements a multi-algorithm engine to compare clustering efficacy:
1. **K-Means Clustering:** Centroid-based clustering optimized via the Elbow Method (WCSS).
2. **Agglomerative (Hierarchical) Clustering:** Bottom-up tree-based clustering.
3. **DBSCAN:** Density-based spatial clustering to identify core clusters and anomalies.
4. **Principal Component Analysis (PCA):** Mathematical dimensionality reduction to compress 30 medical features down to 3 principal components for interactive 3D visualization.

## 🚀 Features of the Interactive Dashboard
The project is deployed locally via a **Streamlit Web Dashboard**, which includes:
* **3D Tumor Map:** Interactive, rotating 3D scatter plot of the PCA-reduced tumor space.
* **Algorithm Comparison:** Live swapping between K-Means, Hierarchical, and DBSCAN engines to evaluate mathematical differences.
* **Elbow Method Proof:** Dynamic rendering of WCSS and Silhouette scores proving `k=2` is mathematically optimal.
* **Cluster Profiling:** Interactive Radar Charts comparing the distinct physical traits of Malignant vs. Benign groups.
* **Live Patient Simulation:** A custom data entry tool allowing doctors/users to manually input specific cell measurements and watch the AI instantly diagnose the patient by matching them to the nearest geometric centroid.

## 💻 How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone https://github.com/YOUR_GITHUB_USERNAME/Medical_AI_Project.git
   cd Medical_AI_Project
   ```

2. **Install dependencies:**
   ```bash
   pip install streamlit pandas numpy plotly scikit-learn matplotlib scipy
   ```

3. **Launch the Dashboard:**
   ```bash
   streamlit run app.py
   ```
   *The dashboard will automatically open in your default web browser at `http://localhost:8501`.*

## 👨‍💻 Developed By
* **SOURAV V P**
* **JOSHUA FAITHSON RONY**

*Built for University Minor Project Submission*
