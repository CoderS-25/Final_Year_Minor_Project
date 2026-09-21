# Automated Malignant Tumor Detection using Unsupervised Clustering

## 1. Project Overview
This project applies **Machine Learning (Artificial Intelligence)** to the medical field of oncology. The goal is to build an Unsupervised Learning model that can automatically analyze the physical measurements of cells from a breast biopsy and group them into distinct profiles (Benign vs. Malignant) without any human doctor telling the AI which one is which beforehand.

## 2. The Dataset
* **Source:** UCI Machine Learning Repository (University of California, Irvine)
* **Name:** Wisconsin Breast Cancer Diagnostic Dataset
* **What it contains:** The dataset contains 569 patient records. Instead of images, it contains 30 numerical columns representing the physical shapes of the cell nuclei extracted from the tumors.

### Understanding the 30 Columns
The 30 columns are built from **10 base measurements** of the cells:
1. **Radius / Perimeter / Area:** The physical size of the cell.
2. **Texture:** The variance in grayscale (is it grainy or smooth?).
3. **Smoothness:** How perfectly round vs. jagged the edges are.
4. **Compactness:** How dense the cell shape is.
5. **Concavity / Concave Points:** The severity and number of "dents" on the cell edge. (Cancer cells are often highly irregular and dented).
6. **Symmetry:** How symmetrical the cell is.
7. **Fractal Dimension:** The complex "spikiness" of the cell's outer border.

For each of these 10 measurements, the dataset records 3 numbers per patient:
* **Mean:** The average measurement across all cells in the biopsy.
* **Error (Standard Error):** How much the cells are mutating and varying from each other.
* **Worst:** The average of the 3 most extreme/ugly cells found in the biopsy.
*(10 base features × 3 metrics = 30 columns)*

## 3. The Methodology (The Algorithms)
Because this is an **Unsupervised Learning** project, we completely hide the actual cancer diagnosis from the AI. We rely on three primary steps:

1. **Data Scaling (StandardScaler):** We must standardize the data first. Cell 'area' might be a massive number (like 1000), while 'smoothness' is a tiny decimal (like 0.05). Scaling ensures the AI treats all physical measurements fairly.
2. **The Elbow Method:** We run the algorithm testing different numbers of clusters (k=1 to k=10) and calculate the error rate. The point where the graph bends like an elbow tells us the mathematically perfect number of clusters to group the patients into (which naturally reveals itself as k=2).
3. **K-Means Clustering:** The algorithm mathematically plots all 569 tumors in a 30-dimensional space and groups the ones that are physically similar. 

## 4. Why This Project is Impressive
Usually, medical AI models use *Supervised Learning* (where a doctor feeds the AI 10,000 pictures explicitly labeled "Cancer" and 10,000 labeled "Safe"). 

By using *Unsupervised Learning*, this project proves that Machine Learning algorithms are smart enough to naturally discover the boundary between cancerous and non-cancerous cells purely based on raw mathematical geometry, completely unassisted by humans.

## 5. Current Progress (30% Phase)
✅ **Data Acquisition:** The UCI dataset is securely downloaded and stored.
✅ **Exploratory Data Analysis (EDA):** Distributions of the tumor features are graphed to prove data integrity.
✅ **Preprocessing:** Data is successfully normalized and scaled.
✅ **Initial Modeling:** The Elbow Method is calculated, plotted, and proves the optimal mathematical separation is k=2.
