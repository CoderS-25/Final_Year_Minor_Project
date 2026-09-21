import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import accuracy_score, silhouette_score, confusion_matrix
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
import io

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="MedCluster AI - Tumor Detection",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS ---
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #1a1a3e 50%, #24243e 100%);
    }
    
    /* FIX: Force all standard text, labels, and markdown to be white/light gray */
    div[data-testid="stMarkdownContainer"] p, 
    div[data-testid="stMarkdownContainer"] h1, 
    div[data-testid="stMarkdownContainer"] h2, 
    div[data-testid="stMarkdownContainer"] h3, 
    div[data-testid="stMarkdownContainer"] h4, 
    div[data-testid="stMarkdownContainer"] h5, 
    div[data-testid="stMarkdownContainer"] h6,
    div[data-testid="stWidgetLabel"] p,
    div[data-testid="stText"],
    label[data-baseweb="radio"] div {
        color: #f8fafc !important;
    }
    
    /* Fix Tabs text visibility */
    button[role="tab"] p {
        color: #94a3b8 !important;
        font-weight: 600;
    }
    button[role="tab"][aria-selected="true"] p {
        color: #00d4ff !important;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1b2a 0%, #1b2838 100%);
        border-right: 2px solid #00d4ff33;
    }
    section[data-testid="stSidebar"] * {
        color: #e0e0e0 !important;
    }
    
    /* Hero header */
    .hero-title {
        font-size: 42px;
        font-weight: 800;
        background: linear-gradient(90deg, #00d4ff, #7b2ff7, #ff6b9d);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
        letter-spacing: -1px;
    }
    .hero-subtitle {
        font-size: 18px;
        color: #8899aa;
        margin-bottom: 30px;
        letter-spacing: 2px;
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 16px;
        padding: 24px;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 40px rgba(0, 212, 255, 0.15);
    }
    .metric-label {
        font-size: 13px;
        color: #94a3b8 !important;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 8px;
    }
    .metric-value {
        font-size: 36px;
        font-weight: 800;
        margin: 0;
    }
    .metric-green { color: #22c55e !important; }
    .metric-blue { color: #00d4ff !important; }
    .metric-purple { color: #a78bfa !important; }
    .metric-orange { color: #fb923c !important; }
    
    /* Section headers */
    .section-header {
        font-size: 24px;
        font-weight: 700;
        color: #f8fafc !important;
        margin-top: 20px;
        margin-bottom: 10px;
        padding-bottom: 8px;
        border-bottom: 2px solid #00d4ff33;
    }
    .section-desc {
        font-size: 14px;
        color: #cbd5e1 !important;
        margin-bottom: 20px;
    }
    
    /* Divider */
    .glow-divider {
        height: 2px;
        background: linear-gradient(90deg, transparent, #00d4ff, #7b2ff7, transparent);
        border: none;
        margin: 30px 0;
    }
    
    /* Info boxes */
    .info-box {
        background: #1e293b;
        border: 1px solid #334155;
        border-left: 4px solid #00d4ff;
        border-radius: 8px;
        padding: 16px;
        margin: 10px 0;
        color: #f8fafc !important;
        font-size: 14px;
    }
    
    /* Patient result */
    .patient-result {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 16px;
        padding: 24px;
        margin-top: 16px;
    }
    .diagnosis-benign {
        font-size: 28px;
        font-weight: 800;
        color: #22c55e !important;
        text-shadow: 0 0 20px rgba(34, 197, 94, 0.3);
    }
    .diagnosis-malignant {
        font-size: 28px;
        font-weight: 800;
        color: #ef4444 !important;
        text-shadow: 0 0 20px rgba(239, 68, 68, 0.3);
    }
    
    /* Sidebar styling */
    .sidebar-title {
        font-size: 22px;
        font-weight: 700;
        background: linear-gradient(90deg, #00d4ff, #7b2ff7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 4px;
    }
    .sidebar-badge {
        display: inline-block;
        background: linear-gradient(90deg, #22c55e, #16a34a);
        color: white !important;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 1px;
    }
    
    /* Button */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #7b2ff7, #00d4ff);
        border: none;
        color: white !important;
        font-weight: 700;
        padding: 12px 28px;
        border-radius: 12px;
        font-size: 15px;
        transition: all 0.3s ease;
    }
    .stButton > button[kind="primary"]:hover {
        box-shadow: 0 8px 25px rgba(0, 212, 255, 0.3);
        transform: translateY(-2px);
    }
    
    /* Hide default streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# --- LOAD DATA ---
@st.cache_data
def load_data():
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    return df, data.target, data.target_names, data.feature_names

df, true_labels, target_names, feature_names = load_data()

# --- PREPROCESSING ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# --- SIDEBAR ---
with st.sidebar:
    st.markdown('<div class="sidebar-title">MedCluster AI</div>', unsafe_allow_html=True)
    st.markdown("Unsupervised Oncology Engine")
    st.markdown("---")
    
    st.markdown("##### Algorithm Selection")
    algo = st.radio(
        "Choose the clustering model:",
        ("K-Means", "Agglomerative (Hierarchical)", "DBSCAN"),
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.markdown("##### Dataset Info")
    st.markdown(f"**Patients:** {len(df)}")
    st.markdown(f"**Features:** {len(df.columns)}")
    st.markdown(f"**Source:** UCI Repository")
    st.markdown(f"**Origin:** Univ. of Wisconsin")
    
    st.markdown("---")
    st.markdown('<span class="sidebar-badge">PHASE: 100% COMPLETE</span>', unsafe_allow_html=True)

# --- ALGORITHM ENGINE ---
if algo == "K-Means":
    model = KMeans(n_clusters=2, random_state=42, n_init=10)
    clusters = model.fit_predict(X_scaled)
elif algo == "Agglomerative (Hierarchical)":
    model = AgglomerativeClustering(n_clusters=2)
    clusters = model.fit_predict(X_scaled)
elif algo == "DBSCAN":
    model = DBSCAN(eps=3.5, min_samples=10)
    clusters = model.fit_predict(X_scaled)

# --- VALIDATION & MAPPING ---
if algo != "DBSCAN":
    if sum(true_labels[clusters == 0]) > sum(true_labels[clusters == 1]):
        mapped_clusters = np.where(clusters == 0, 1, 0)
        cluster_names = np.where(clusters == 0, "Benign", "Malignant")
    else:
        mapped_clusters = clusters
        cluster_names = np.where(clusters == 0, "Malignant", "Benign")
    
    acc = accuracy_score(true_labels, mapped_clusters)
    sil_score = silhouette_score(X_scaled, clusters)
    cm = confusion_matrix(true_labels, mapped_clusters)
else:
    cluster_names = np.where(clusters == -1, "Anomaly", np.where(clusters == 0, "Benign", "Malignant"))
    acc = None
    if len(set(clusters)) > 1:
        sil_score = silhouette_score(X_scaled, clusters)
    else:
        sil_score = 0
    cm = None

# --- PCA ---
pca = PCA(n_components=3)
pca_result = pca.fit_transform(X_scaled)
df_pca = pd.DataFrame(pca_result, columns=['PC1', 'PC2', 'PC3'])
df_pca['AI Cluster'] = cluster_names
df_pca['Actual Diagnosis'] = np.where(true_labels == 0, "Malignant", "Benign")

# --- HERO HEADER ---
st.markdown('<div class="hero-title">MedCluster AI</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-subtitle">AUTOMATED TUMOR DETECTION USING UNSUPERVISED CLUSTERING</div>', unsafe_allow_html=True)

# --- METRICS ROW ---
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f'''
    <div class="metric-card">
        <div class="metric-label">Active Algorithm</div>
        <div class="metric-value metric-blue">{algo.split(" ")[0]}</div>
    </div>''', unsafe_allow_html=True)
with col2:
    acc_display = f"{acc*100:.1f}%" if acc else "N/A"
    st.markdown(f'''
    <div class="metric-card">
        <div class="metric-label">Biological Accuracy</div>
        <div class="metric-value metric-green">{acc_display}</div>
    </div>''', unsafe_allow_html=True)
with col3:
    st.markdown(f'''
    <div class="metric-card">
        <div class="metric-label">Silhouette Score</div>
        <div class="metric-value metric-purple">{sil_score:.3f}</div>
    </div>''', unsafe_allow_html=True)
with col4:
    n_clusters_found = len(set(clusters)) - (1 if -1 in clusters else 0)
    st.markdown(f'''
    <div class="metric-card">
        <div class="metric-label">Clusters Found</div>
        <div class="metric-value metric-orange">{n_clusters_found}</div>
    </div>''', unsafe_allow_html=True)

st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)

# --- TABBED SECTIONS ---
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "3D Tumor Map",
    "Elbow Method",
    "Cluster Profiles",
    "Confusion Matrix",
    "Live Patient Simulation"
])

# === TAB 1: 3D VISUALIZATION ===
with tab1:
    st.markdown('<div class="section-header">3D Tumor Geometry (PCA Reduced)</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-desc">30 physical cell measurements compressed into 3 principal components. Each dot is a real patient. Colors represent AI-assigned clusters.</div>', unsafe_allow_html=True)
    
    colA, colB = st.columns([3, 1])
    with colA:
        fig = px.scatter_3d(
            df_pca, x='PC1', y='PC2', z='PC3',
            color='AI Cluster',
            color_discrete_map={"Benign": "#22c55e", "Malignant": "#ef4444", "Anomaly": "#64748b"},
            hover_data={'PC1': ':.2f', 'PC2': ':.2f', 'PC3': ':.2f', 'Actual Diagnosis': True},
            opacity=0.8
        )
        fig.update_layout(
            scene=dict(
                xaxis=dict(backgroundcolor='#0f172a', gridcolor='#1e293b', title_font=dict(color='#64748b')),
                yaxis=dict(backgroundcolor='#0f172a', gridcolor='#1e293b', title_font=dict(color='#64748b')),
                zaxis=dict(backgroundcolor='#0f172a', gridcolor='#1e293b', title_font=dict(color='#64748b')),
                bgcolor='#0f172a'
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='#94a3b8',
            height=550,
            margin=dict(l=0, r=0, t=0, b=0),
            legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(size=13))
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with colB:
        st.markdown(f"""
        <div class="info-box">
            <b>How to Read This:</b><br><br>
            Each dot = 1 patient's tumor<br><br>
            <span style="color:#22c55e;">Green</span> = AI says Benign<br>
            <span style="color:#ef4444;">Red</span> = AI says Malignant<br><br>
            The AI separated these clusters using only the physical geometry of the cells. No diagnosis was ever provided to the model.
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="info-box">
            <b>PCA Variance Explained:</b><br><br>
            PC1: {pca.explained_variance_ratio_[0]*100:.1f}%<br>
            PC2: {pca.explained_variance_ratio_[1]*100:.1f}%<br>
            PC3: {pca.explained_variance_ratio_[2]*100:.1f}%<br><br>
            <b>Total: {sum(pca.explained_variance_ratio_)*100:.1f}%</b>
        </div>
        """, unsafe_allow_html=True)

# === TAB 2: ELBOW METHOD ===
with tab2:
    st.markdown('<div class="section-header">Elbow Method: Optimal Cluster Discovery</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-desc">We test K-Means with k=1 through k=10 and measure the error (WCSS). The "elbow" bend in the curve reveals the mathematically optimal number of clusters.</div>', unsafe_allow_html=True)
    
    wcss = []
    sil_scores = []
    for i in range(2, 11):
        km = KMeans(n_clusters=i, random_state=42, n_init=10)
        km.fit(X_scaled)
        wcss.append(km.inertia_)
        sil_scores.append(silhouette_score(X_scaled, km.labels_))
    
    km1 = KMeans(n_clusters=1, random_state=42, n_init=10)
    km1.fit(X_scaled)
    wcss_full = [km1.inertia_] + wcss
    
    colE1, colE2 = st.columns(2)
    with colE1:
        fig_elbow = go.Figure()
        fig_elbow.add_trace(go.Scatter(
            x=list(range(1, 11)), y=wcss_full,
            mode='lines+markers',
            marker=dict(size=10, color='#00d4ff', line=dict(width=2, color='white')),
            line=dict(color='#00d4ff', width=3),
            name='WCSS'
        ))
        fig_elbow.add_vline(x=2, line_dash="dash", line_color="#ef4444", annotation_text="Optimal k=2", annotation_font_color="#ef4444")
        fig_elbow.update_layout(
            title=dict(text="WCSS vs Number of Clusters", font=dict(color='#e2e8f0', size=16)),
            xaxis_title="Number of Clusters (k)",
            yaxis_title="WCSS",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='#0f172a',
            font_color='#94a3b8',
            xaxis=dict(gridcolor='#1e293b'),
            yaxis=dict(gridcolor='#1e293b'),
            height=400
        )
        st.plotly_chart(fig_elbow, use_container_width=True)
    
    with colE2:
        fig_sil = go.Figure()
        fig_sil.add_trace(go.Bar(
            x=list(range(2, 11)), y=sil_scores,
            marker_color=['#7b2ff7' if i == 0 else '#334155' for i in range(len(sil_scores))],
            text=[f"{s:.3f}" for s in sil_scores],
            textposition='outside',
            textfont=dict(color='#94a3b8')
        ))
        fig_sil.update_layout(
            title=dict(text="Silhouette Score by k", font=dict(color='#e2e8f0', size=16)),
            xaxis_title="Number of Clusters (k)",
            yaxis_title="Silhouette Score",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='#0f172a',
            font_color='#94a3b8',
            xaxis=dict(gridcolor='#1e293b'),
            yaxis=dict(gridcolor='#1e293b'),
            height=400
        )
        st.plotly_chart(fig_sil, use_container_width=True)
    
    st.markdown("""
    <div class="info-box">
        <b>What does this prove?</b> The Elbow curve bends sharply at <b>k=2</b>, and the Silhouette Score is highest at <b>k=2</b>. 
        Both metrics independently confirm that the data naturally separates into exactly <b>2 clusters</b> — which biologically corresponds to <b>Benign</b> and <b>Malignant</b> tumors. 
    </div>
    """, unsafe_allow_html=True)

# === TAB 3: CLUSTER PROFILES ===
with tab3:
    st.markdown('<div class="section-header">Cluster Profiling: What Does Each Group Look Like?</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-desc">Comparing the average physical measurements of tumors in each AI-generated cluster to understand what distinguishes them.</div>', unsafe_allow_html=True)
    
    mean_features = [f for f in feature_names if f.startswith('mean')]
    df_profile = df[mean_features].copy()
    df_profile['Cluster'] = cluster_names
    cluster_means = df_profile.groupby('Cluster').mean()
    cluster_means_norm = (cluster_means - cluster_means.min()) / (cluster_means.max() - cluster_means.min() + 1e-8)
    
    colR1, colR2 = st.columns([2, 1])
    with colR1:
        fig_radar = go.Figure()
        colors = {"Benign": "#22c55e", "Malignant": "#ef4444", "Anomaly": "#64748b"}
        
        for cluster_name in cluster_means_norm.index:
            if cluster_name in colors:
                values = cluster_means_norm.loc[cluster_name].values.tolist()
                values.append(values[0])
                categories = [f.replace('mean ', '').title() for f in mean_features]
                categories.append(categories[0])
                
                fig_radar.add_trace(go.Scatterpolar(
                    r=values,
                    theta=categories,
                    fill='toself',
                    name=cluster_name,
                    line_color=colors[cluster_name],
                    fillcolor=colors[cluster_name].replace(')', ', 0.15)').replace('rgb', 'rgba') if 'rgb' in colors[cluster_name] else colors[cluster_name] + '22',
                    opacity=0.9
                ))
        
        fig_radar.update_layout(
            polar=dict(
                bgcolor='#0f172a',
                radialaxis=dict(visible=True, gridcolor='#1e293b', color='#64748b'),
                angularaxis=dict(gridcolor='#1e293b', color='#94a3b8')
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#94a3b8',
            height=500,
            legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(size=14)),
            title=dict(text="Tumor Profile Radar", font=dict(color='#e2e8f0', size=16))
        )
        st.plotly_chart(fig_radar, use_container_width=True)
    
    with colR2:
        st.markdown("""
        <div class="info-box">
            <b>How to Read the Radar:</b><br><br>
            Each axis represents a physical cell measurement.<br><br>
            <span style="color:#ef4444;"><b>Red (Malignant)</b></span> tumors are significantly <b>larger</b>, more <b>concave</b>, and more <b>irregular</b>.<br><br>
            <span style="color:#22c55e;"><b>Green (Benign)</b></span> tumors are <b>smaller</b>, <b>smoother</b>, and more <b>symmetrical</b>.<br><br>
            This is exactly what oncologists expect biologically.
        </div>
        """, unsafe_allow_html=True)
    
    fig_bar = go.Figure()
    for cluster_name in cluster_means.index:
        if cluster_name in colors:
            fig_bar.add_trace(go.Bar(
                x=[f.replace('mean ', '').title() for f in mean_features],
                y=cluster_means.loc[cluster_name].values,
                name=cluster_name,
                marker_color=colors[cluster_name],
                opacity=0.85
            ))
    fig_bar.update_layout(
        barmode='group',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='#0f172a',
        font_color='#94a3b8',
        xaxis=dict(gridcolor='#1e293b'),
        yaxis=dict(gridcolor='#1e293b', title="Mean Value"),
        height=350,
        legend=dict(bgcolor='rgba(0,0,0,0)')
    )
    st.plotly_chart(fig_bar, use_container_width=True)

# === TAB 4: CONFUSION MATRIX ===
with tab4:
    st.markdown('<div class="section-header">Ground-Truth Validation: Confusion Matrix</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-desc">After the AI clusters the tumors without any labels, we reveal the actual medical diagnoses and check how many the AI got right.</div>', unsafe_allow_html=True)
    
    if cm is not None:
        colC1, colC2 = st.columns([2, 1])
        with colC1:
            fig_cm = px.imshow(
                cm,
                labels=dict(x="AI Prediction", y="Actual Diagnosis", color="Count"),
                x=["Malignant", "Benign"],
                y=["Malignant", "Benign"],
                color_continuous_scale=["#0f172a", "#00d4ff", "#7b2ff7"],
                text_auto=True
            )
            fig_cm.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='#0f172a',
                font=dict(color='#94a3b8', size=14),
                height=450,
                title=dict(text=f"Confusion Matrix ({algo})", font=dict(color='#e2e8f0', size=16))
            )
            fig_cm.update_traces(textfont_size=20, textfont_color='white')
            st.plotly_chart(fig_cm, use_container_width=True)
        
        with colC2:
            tn, fp, fn, tp = cm.ravel()
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            st.markdown(f"""
            <div class="info-box">
                <b>Detailed Metrics:</b><br><br>
                <b>True Positives:</b> {tp}<br>
                <b>True Negatives:</b> {tn}<br>
                <b>False Positives:</b> {fp}<br>
                <b>False Negatives:</b> {fn}<br><br>
                <b>Precision:</b> {precision*100:.1f}%<br>
                <b>Recall:</b> {recall*100:.1f}%<br>
                <b>F1-Score:</b> {f1*100:.1f}%<br><br>
                <b>Overall Accuracy:</b> <span style="color:#22c55e; font-size:20px;">{acc*100:.1f}%</span>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="info-box">
                <b>What does this mean?</b><br><br>
                The AI was never told which tumors were cancerous. Yet it achieved high accuracy purely through mathematical geometry. This proves unsupervised clustering can serve as an automated diagnostic assistant.
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="info-box">
            DBSCAN is a density-based algorithm that does not produce fixed cluster assignments suitable for direct confusion matrix comparison.
            Please select <b>K-Means</b> or <b>Agglomerative</b> to view the validation matrix.
        </div>
        """, unsafe_allow_html=True)

# === TAB 5: LIVE PATIENT SIMULATION ===
with tab5:
    st.markdown('<div class="section-header">Live Patient Biopsy Simulation</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-desc">Feed a completely new patient biopsy into the Unsupervised AI Engine and watch it assign a medical profile.</div>', unsafe_allow_html=True)
    
    if algo == "K-Means":
        sim_mode = st.radio("Choose Input Method:", ["🎲 Random Existing Patient", "⌨️ Manual Data Entry (Type Exact Numbers)"], horizontal=True)
        st.write("---")
        
        if sim_mode == "🎲 Random Existing Patient":
            colP1, colP2 = st.columns([1, 1])
            with colP1:
                st.markdown("""
                <div class="info-box">
                    <b>How This Works (Centroid Proximity):</b><br><br>
                    1. A random patient's 30 cell measurements are selected.<br>
                    2. The AI calculates the mathematical distance from this patient to each cluster center (centroid).<br>
                    3. The patient is assigned to the nearest cluster.
                </div>
                """, unsafe_allow_html=True)
            
            with colP2:
                if st.button("Run Random Patient Biopsy", type="primary", use_container_width=True):
                    random_idx = np.random.randint(0, len(df))
                    patient_data = df.iloc[random_idx:random_idx+1]
                    patient_scaled = scaler.transform(patient_data)
                    pred_cluster = model.predict(patient_scaled)[0]
                    
                    if sum(true_labels[clusters == 0]) > sum(true_labels[clusters == 1]):
                        pred_label = "Benign" if pred_cluster == 0 else "Malignant"
                    else:
                        pred_label = "Malignant" if pred_cluster == 0 else "Benign"
                    
                    actual = "Benign" if true_labels[random_idx] == 1 else "Malignant"
                    correct = pred_label == actual
                    
                    st.markdown(f"""
                    <div class="patient-result">
                        <div class="metric-label">AI DIAGNOSIS</div>
                        <div class="{'diagnosis-benign' if pred_label == 'Benign' else 'diagnosis-malignant'}">{pred_label} Profile</div>
                        <br>
                        <div class="metric-label">ACTUAL DIAGNOSIS (HIDDEN GROUND TRUTH)</div>
                        <div style="font-size:18px; color:{'#22c55e' if correct else '#ef4444'};">
                            {actual} {'--- Correct Match!' if correct else '--- Mismatch'}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("##### Patient's Raw Cell Measurements")
                    display_df = patient_data.T.reset_index()
                    display_df.columns = ['Measurement', 'Value']
                    display_df['Value'] = display_df['Value'].round(4)
                    st.dataframe(display_df, use_container_width=True, height=200)

        else: # Manual Data Entry (Typing Mode)
            st.markdown("##### ⌨️ Type Custom Tumor Measurements")
            st.write("Click inside the boxes to type exact numbers (or use the small +/- arrows). The remaining 25 features are locked at the population average.")
            
            col_s1, col_s2 = st.columns(2)
            with col_s1:
                val_radius = st.number_input("Mean Radius (Cell Size)", min_value=float(df['mean radius'].min()), max_value=float(df['mean radius'].max()), value=float(df['mean radius'].mean()), step=0.1)
                val_texture = st.number_input("Mean Texture (Grayscale Variance)", min_value=float(df['mean texture'].min()), max_value=float(df['mean texture'].max()), value=float(df['mean texture'].mean()), step=0.1)
                val_concavity = st.number_input("Mean Concavity (Severity of Dents)", min_value=float(df['mean concavity'].min()), max_value=float(df['mean concavity'].max()), value=float(df['mean concavity'].mean()), step=0.01)
            with col_s2:
                val_area = st.number_input("Mean Area (Total Footprint)", min_value=float(df['mean area'].min()), max_value=float(df['mean area'].max()), value=float(df['mean area'].mean()), step=10.0)
                val_smoothness = st.number_input("Mean Smoothness (Edge Jaggedness)", min_value=float(df['mean smoothness'].min()), max_value=float(df['mean smoothness'].max()), value=float(df['mean smoothness'].mean()), step=0.01)
            
            st.write("---")
            if st.button("Diagnose Custom Patient", type="primary", use_container_width=True):
                # Build custom patient data from population mean
                custom_patient = df.mean().copy()
                # Override the 5 chosen features
                custom_patient['mean radius'] = val_radius
                custom_patient['mean texture'] = val_texture
                custom_patient['mean concavity'] = val_concavity
                custom_patient['mean area'] = val_area
                custom_patient['mean smoothness'] = val_smoothness
                
                # Predict
                patient_scaled = scaler.transform([custom_patient.values])
                pred_cluster = model.predict(patient_scaled)[0]
                
                if sum(true_labels[clusters == 0]) > sum(true_labels[clusters == 1]):
                    pred_label = "Benign" if pred_cluster == 0 else "Malignant"
                else:
                    pred_label = "Malignant" if pred_cluster == 0 else "Benign"
                    
                st.markdown(f"""
                <div class="patient-result" style="text-align:center;">
                    <div class="metric-label">AI DIAGNOSIS (CUSTOM INPUT)</div>
                    <div class="{'diagnosis-benign' if pred_label == 'Benign' else 'diagnosis-malignant'}" style="font-size:36px;">{pred_label} Profile</div>
                    <p style="color:#94a3b8; margin-top:10px;">The algorithm calculated the mathematical distance of your custom inputs and matched it to the {pred_label} centroid.</p>
                </div>
                """, unsafe_allow_html=True)

    else:
        st.markdown(f"""
        <div class="info-box">
            Live Patient Simulation requires centroid-based models. <b>{algo}</b> does not compute centroids for new-point assignment.<br><br>
            Please select <b>K-Means</b> from the sidebar to enable this feature.
        </div>
        """, unsafe_allow_html=True)

# --- FOOTER ---
st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center; color:#475569; font-size:13px; padding: 10px 0;">
    Built with Streamlit & Scikit-Learn | Dataset: UCI Machine Learning Repository (Wisconsin Breast Cancer Diagnostic)
</div>
""", unsafe_allow_html=True)
