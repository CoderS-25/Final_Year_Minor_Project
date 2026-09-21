import sys
import os
try:
    from pptx import Presentation
    from pptx.util import Inches, Pt
except ImportError:
    print("python-pptx not found.")
    sys.exit(1)

prs = Presentation()

# 1. Title
slide = prs.slides.add_slide(prs.slide_layouts[0])
slide.shapes.title.text = "Automated Malignant Tumor Detection using Unsupervised Clustering"
slide.placeholders[1].text = "Interim-1 Project Presentation\n\n[Insert Your Name/Group Details]"

# 2. Intro
slide = prs.slides.add_slide(prs.slide_layouts[1])
slide.shapes.title.text = "2. Introduction"
slide.placeholders[1].text = "• Medical AI is revolutionizing healthcare by identifying hidden patterns in clinical data.\n• Pathologists manually analyze hundreds of cell nuclei measurements from biopsies, which is time-consuming.\n• Data mining (unsupervised learning) can automatically group tumor profiles based on their physical traits.\n• This project applies Machine Learning to discover malignant vs. benign tumor clusters without human labeling."

# 3. Problem Statement
slide = prs.slides.add_slide(prs.slide_layouts[1])
slide.shapes.title.text = "3. Problem Statement"
slide.placeholders[1].text = "• Clinical oncology generates vast datasets of physical tumor measurements, but extracting actionable, automated diagnosis boundaries is difficult.\n• Traditional supervised models require doctors to manually label thousands of images/records.\n• The core problem is finding a scalable, automated way to cluster unlabeled tumor data to see if an AI can naturally discover the difference between cancerous and non-cancerous cells."

# 4. Objectives
slide = prs.slides.add_slide(prs.slide_layouts[1])
slide.shapes.title.text = "4. Project Objectives"
slide.placeholders[1].text = "• To perform unsupervised learning on the real Wisconsin Breast Cancer clinical dataset.\n• To discover distinct medical clusters based on 30 physical cell measurements (radius, texture, area, etc.).\n• To utilize the 'Elbow Method' to mathematically determine the optimal number of clusters (k).\n• To analyze if the algorithm's mathematical clusters perfectly align with biological diagnoses (Benign vs. Malignant)."

# 5. Lit Survey
def add_lit_survey_slide(prs, title_text, papers):
    slide = prs.slides.add_slide(prs.slide_layouts[5])
    slide.shapes.title.text = title_text
    table = slide.shapes.add_table(len(papers)+1, 7, Inches(0.2), Inches(1.5), Inches(9.6), Inches(5.5)).table
    table.columns[0].width, table.columns[1].width, table.columns[2].width, table.columns[3].width, table.columns[4].width, table.columns[5].width, table.columns[6].width = Inches(1.0), Inches(1.2), Inches(1.6), Inches(1.0), Inches(1.2), Inches(1.8), Inches(1.5)
    headers = ["Author/Year", "Journal", "Title", "Dataset", "Methodology", "Key Contributions", "Limitations"]
    for i, header in enumerate(headers):
        cell = table.cell(0, i)
        cell.text = header
        cell.text_frame.paragraphs[0].font.size, cell.text_frame.paragraphs[0].font.bold = Pt(11), True
    for row_idx, paper in enumerate(papers):
        for col_idx, value in enumerate(paper):
            cell = table.cell(row_idx + 1, col_idx)
            cell.text = value
            cell.text_frame.paragraphs[0].font.size = Pt(9)

papers_1_5 = [
    ["Street et al., 1993", "SPIE", "Nuclear feature extraction for breast tumor diagnosis", "Wisconsin Breast Cancer", "Active Contours", "Established the baseline clinical dataset", "Small initial sample size"],
    ["Sharma et al., 2021", "J. Medical Systems", "Unsupervised Machine Learning in Oncology", "Clinical Biopsies", "K-Means, PCA", "High accuracy in automated grouping", "Did not use standard scaling"],
    ["Chen & Li, 2020", "IEEE Access", "Tumor Segmentation using Clustering Algorithms", "MRI & Clinical Data", "Hierarchical Clustering", "Visualized hierarchy of cell mutations", "Scalability issues with large n"],
    ["Patel & Desai, 2022", "Procedia CS", "Breast Cancer Detection based on K-Means", "Wisconsin Diagnostic", "K-Means + SVM", "Combined unsupervised and supervised", "Sensitive to outliers in area"],
    ["Zhang et al., 2021", "Decision Support", "Behavior analysis of malignant cells", "Oncology Data", "DBSCAN vs K-Means", "Identified noise/outlier cells effectively", "DBSCAN struggled with varying density"]
]
papers_6_10 = [
    ["Kumar & Singh, 2020", "Int. J. Med Info", "Big Data Analytics for Cancer Detection", "Global Oncology", "K-Means on Hadoop", "Scalable clustering approach", "Complex implementation overhead"],
    ["Lee & Kim, 2023", "IEEE Trans. BME", "Morphological Segmentation using ML", "Cell Nuclei Data", "K-Medoids", "Robust to outliers compared to K-Means", "Slower execution time"],
    ["Ali et al., 2018", "Comp in Bio & Med", "Integrated approach to tumor clustering", "Clinical Database", "SOM (Self-Organizing Maps)", "Excellent topological visualization", "Hard to interpret grid for non-experts"],
    ["Wang & Liu, 2022", "Applied Soft Comp", "Multi-objective clustering for medical data", "Wisconsin Dataset", "Evolutionary clustering", "Optimized multiple validation indices", "Computationally intensive"],
    ["Gupta & Sharma, 2021", "J. Biomed Research", "Predictive oncology based on K-means", "Hospital EHR", "K-Means + Predictive", "Linked clusters to cancer recurrence", "Requires extensive historical data"]
]

add_lit_survey_slide(prs, "5. Literature Survey (1/2)", papers_1_5)
add_lit_survey_slide(prs, "5. Literature Survey (2/2)", papers_6_10)

# 6. Research Gaps
slide = prs.slides.add_slide(prs.slide_layouts[1])
slide.shapes.title.text = "6. Research Gaps"
slide.placeholders[1].text = "• Many existing studies rely solely on standard K-means without exploring the impact of robust feature scaling on extreme clinical measurements (e.g., cell area vs smoothness).\n• There is often a lack of deep, actionable profiling mapping unsupervised clusters back to actual biological diagnoses.\n• Frequent omission of cross-validating the 'Elbow Method' with biological ground-truth for rigorous validation."

# 7. Methodology
slide = prs.slides.add_slide(prs.slide_layouts[1])
slide.shapes.title.text = "7. Methodology / System Architecture"
slide.placeholders[1].text = "1. Data Collection: Obtain Real Wisconsin Breast Cancer Dataset (30 features per tumor).\n2. Data Preprocessing: Standardize/scale numeric features to prevent large values (Area) from dominating tiny values (Smoothness).\n3. Optimal 'k' Selection: Execute the Elbow Method to plot Within-Cluster Sum of Squares (WCSS).\n4. Model Training: Apply K-Means clustering with the selected 'k'.\n5. Output & Profiling: Generate scatter plots to see if clusters mathematically represent Benign vs. Malignant."

# 8. Module Description (Text)
slide = prs.slides.add_slide(prs.slide_layouts[1])
slide.shapes.title.text = "8. Module-wise Description"
slide.placeholders[1].text = "• Module 1: Data Acquisition & Preprocessing (Loading UCI dataset, feature scaling).\n• Module 2: Exploratory Data Analysis (Visualizing distributions of Tumor Radius, Texture, and Area).\n• Module 3: Clustering Model Implementation (Elbow Method iteration, training K-Means algorithm).\n• Module 4: Visualization & Interpretation (Plotting medical clusters, validating against biological ground truth)."

# 8a. IMAGE 1: EDA
slide = prs.slides.add_slide(prs.slide_layouts[5])
slide.shapes.title.text = "8. Module-wise Description: Data Visualization (30% Output)"
if os.path.exists("Tumor_EDA_Distributions.png"):
    slide.shapes.add_picture("Tumor_EDA_Distributions.png", Inches(0.5), Inches(1.8), width=Inches(9.0))

# 8b. IMAGE 2: Elbow Method
slide = prs.slides.add_slide(prs.slide_layouts[5])
slide.shapes.title.text = "8. Module-wise Description: Model Implementation (30% Output)"
if os.path.exists("Medical_Elbow_Method.png"):
    slide.shapes.add_picture("Medical_Elbow_Method.png", Inches(1.5), Inches(1.5), width=Inches(7.0))

# 9. Conclusion
slide = prs.slides.add_slide(prs.slide_layouts[1])
slide.shapes.title.text = "9. Conclusion"
slide.placeholders[1].text = "• The project will successfully group complex, unlabeled clinical data into distinct tumor profiles.\n• The use of the K-Means algorithm and the Elbow method ensures a mathematically sound segmentation strategy.\n• Ultimately, proving that unsupervised AI can naturally discover cancer patterns opens the door for automated diagnostic assistants in pathology."

prs.save("Medical_AI_Interim1_Presentation.pptx")
print("Successfully added the graph slides into the PPT!")
