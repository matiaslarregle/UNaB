import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="Segmentación de Pacientes",
    page_icon="🏥",
    layout="wide"
)

# ── COLORES ───────────────────────────────────────────────────────────────────
CLUSTER_COLORS = {
    'Bajo Riesgo':     '#4ade80',
    'Riesgo Moderado': '#60a5fa',
    'Alto Riesgo':     '#f87171',
}
CLUSTER_ORDER = ['Bajo Riesgo', 'Riesgo Moderado', 'Alto Riesgo']
RISK_COLORS   = {
    'Low': '#4ade80', 'Medium': '#facc15',
    'High': '#fb923c', 'Critical': '#f87171'
}
RISK_ORDER = ['Low', 'Medium', 'High', 'Critical']

# ── CARGA ─────────────────────────────────────────────────────────────────────
@st.cache_data
def load():
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), "patient_segmentation_final.csv"))

    cluster_map = {0: 'Riesgo Moderado', 1: 'Bajo Riesgo', 2: 'Alto Riesgo'}
    if 'cluster' in df.columns and df['cluster'].dtype in [int, float]:
        df['cluster'] = df['cluster'].map(cluster_map)

    if 'BMI_ord' not in df.columns:
        df['BMI_ord'] = pd.cut(
            df['BMI'],
            bins=[0, 24.9, 29.9, 34.9, 39.9, 100],
            labels=[0, 1, 2, 3, 4]
        ).astype(float)

    return df

df = load()

# ── NORMALIZACIÓN ─────────────────────────────────────────────────────────────
def norm(series):
    mn, mx = series.min(), series.max()
    return (series - mn) / (mx - mn) if mx > mn else pd.Series(0.0, index=series.index)

df['age_n']  = norm(df['Age'])
df['bmi_n']  = norm(df['BMI_ord'])
df['cc_n']   = norm(df['Num_Chronic_Conditions'])
df['vis_n']  = 1 - norm(df['Annual_Visits'])       # invertida: menos visitas = más riesgo
df['prev_n'] = df['Preventive_Care_Flag'].map(
                   {'No': 1.0, 'Sí': 0.0, 0: 1.0, 1: 0.0}
               ).fillna(0.0)                        # sin atención preventiva = más riesgo

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏥 Segmentación")
    st.markdown("---")

    st.markdown("### Pesos del Risk Score")
    st.caption("Todas las variables están normalizadas (0–1). Los pesos son comparables entre sí.")

    w_age  = st.slider("Edad",                          0.0, 5.0, 1.0, 0.1)
    w_bmi  = st.slider("IMC (BMI)",                     0.0, 5.0, 2.0, 0.1)
    w_cc   = st.slider("Condiciones crónicas",          0.0, 5.0, 2.0, 0.1)
    w_vis  = st.slider("Pocas visitas anuales ↑riesgo", 0.0, 5.0, 1.0, 0.1)
    w_prev = st.slider("Sin atención preventiva",       0.0, 5.0, 1.0, 0.1)

    st.markdown("### Umbrales de riesgo")
    t_low  = st.slider("Límite Low / Medium",    0.0,       5.0,  2.0, 0.1)
    t_mid  = st.slider("Límite Medium / High",   t_low,    10.0,  4.0, 0.1)
    t_high = st.slider("Límite High / Critical", t_mid,    15.0,  6.0, 0.1)

    st.markdown("### Filtros")
    cluster_filter = st.multiselect(
        "Clusters", options=CLUSTER_ORDER, default=CLUSTER_ORDER
    )
    risk_filter = st.multiselect(
        "Niveles de riesgo", options=RISK_ORDER, default=RISK_ORDER
    )

# ── RISK SCORE DINÁMICO ───────────────────────────────────────────────────────
df['Risk_Score'] = (
    df['age_n']  * w_age  +
    df['bmi_n']  * w_bmi  +
    df['cc_n']   * w_cc   +
    df['vis_n']  * w_vis  +
    df['prev_n'] * w_prev
)

df['Risk_Level'] = pd.cut(
    df['Risk_Score'],
    bins=[-np.inf, t_low, t_mid, t_high, np.inf],
    labels=['Low', 'Medium', 'High', 'Critical']
).astype(str)

df_f = df[
    df['cluster'].isin(cluster_filter) &
    df['Risk_Level'].isin(risk_filter)
].copy()

# ── HEADER ────────────────────────────────────────────────────────────────────
st.title("🏥 Segmentación de Pacientes")
st.caption(
    "Ajustá los pesos del Risk Score desde el sidebar y explorá cómo cambia "
    "la distribución de riesgo dentro de cada cluster. "
    "Todas las variables están normalizadas entre 0 y 1, por lo que "
    "los pesos son directamente comparables entre sí."
)
st.markdown("---")

# ── MÉTRICAS ──────────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
total = len(df_f)
high  = len(df_f[df_f['Risk_Level'].isin(['High', 'Critical'])])

c1.metric("Pacientes", f"{total:,}")
c2.metric("Alto / Crítico", f"{high:,}",
          delta=f"{high/total*100:.1f}% del total" if total > 0 else "—")
c3.metric("Risk Score promedio",
          f"{df_f['Risk_Score'].mean():.2f}" if total > 0 else "—")
c4.metric("Clusters activos", len(cluster_filter))

st.markdown("---")

# ── DISTRIBUCIÓN RISK LEVEL POR CLUSTER ──────────────────────────────────────
st.markdown("## Distribución de Risk Level por cluster")

if total == 0:
    st.warning("No hay pacientes con los filtros seleccionados.")
else:
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    pivot = (df_f
             .groupby(['cluster', 'Risk_Level'])
             .size()
             .reset_index(name='count'))

    x     = np.arange(len(cluster_filter))
    width = 0.2

    for k, level in enumerate(RISK_ORDER):
        if level not in risk_filter:
            continue
        vals = [
            pivot[(pivot['cluster'] == cl) &
                  (pivot['Risk_Level'] == level)]['count'].sum()
            for cl in cluster_filter
        ]
        ax.bar(x + k * width, vals, width,
               label=level, color=RISK_COLORS[level], alpha=0.85)

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(cluster_filter, fontsize=11)
    ax.set_ylabel('Pacientes', fontsize=11)
    ax.set_title('Risk Level por cluster', fontsize=13, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(frameon=False, fontsize=10)
    ax.grid(True, axis='y', alpha=0.25, linestyle='--')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

st.markdown("---")

# ── DISTRIBUCIONES DE VARIABLES ───────────────────────────────────────────────
st.markdown("## Distribución de variables por cluster")

num_vars = ['Age', 'Num_Chronic_Conditions', 'Annual_Visits',
            'Avg_Billing_Amount', 'BMI_ord', 'Days_Since_Last_Visit', 'Risk_Score']
cat_vars = ['Gender', 'Insurance_Type', 'Primary_Condition', 'Preventive_Care_Flag']

tab_num, tab_cat = st.tabs(["Numéricas 📊", "Categóricas 📋"])

with tab_num:
    var_num = st.selectbox("Variable numérica", num_vars)

    if total == 0:
        st.warning("Sin datos con los filtros actuales.")
    else:
        fig, ax = plt.subplots(figsize=(10, 4))
        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')

        sns.violinplot(
            data=df_f,
            x='cluster', y=var_num,
            order=[c for c in CLUSTER_ORDER if c in cluster_filter],
            palette=CLUSTER_COLORS,
            inner='quartile',
            linewidth=1,
            ax=ax
        )

        ax.set_title(f'Distribución de {var_num} por cluster',
                     fontsize=13, fontweight='bold')
        ax.set_xlabel('')
        ax.set_ylabel(var_num, fontsize=11)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, axis='y', alpha=0.25, linestyle='--')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

with tab_cat:
    var_cat = st.selectbox("Variable categórica", cat_vars)

    if total == 0:
        st.warning("Sin datos con los filtros actuales.")
    else:
        active = [c for c in CLUSTER_ORDER if c in cluster_filter]
        fig, axes = plt.subplots(1, len(active),
                                  figsize=(5 * len(active), 4),
                                  sharey=False)
        if len(active) == 1:
            axes = [axes]

        fig.patch.set_facecolor('white')

        for ax, cluster_name in zip(axes, active):
            ax.set_facecolor('white')
            subset = df_f[df_f['cluster'] == cluster_name]
            counts = subset[var_cat].value_counts()
            color  = CLUSTER_COLORS[cluster_name]

            bars = ax.barh(counts.index, counts.values, color=color, alpha=0.85)
            ax.set_title(cluster_name, color=color, fontsize=11, fontweight='bold')
            ax.set_xlabel('Pacientes', fontsize=9)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.grid(True, axis='x', alpha=0.25, linestyle='--')
            ax.tick_params(labelsize=9)

            for bar, val in zip(bars, counts.values):
                ax.text(val + 0.5, bar.get_y() + bar.get_height() / 2,
                        str(val), va='center', fontsize=8, color='#374151')

        fig.suptitle(f'Distribución de {var_cat} por cluster',
                     fontsize=13, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

st.markdown("---")

# ── TABLA DE PACIENTES ────────────────────────────────────────────────────────
st.markdown("## Pacientes")

cols_show = ['PatientID', 'Age', 'Gender', 'Insurance_Type',
             'Num_Chronic_Conditions', 'Annual_Visits',
             'Avg_Billing_Amount', 'Primary_Condition',
             'Preventive_Care_Flag', 'cluster', 'Risk_Level', 'Risk_Score']

df_show = (df_f[[c for c in cols_show if c in df_f.columns]]
           .copy()
           .sort_values('Risk_Score', ascending=False))

df_show['Risk_Score'] = df_show['Risk_Score'].round(3)
df_show = df_show.rename(columns={'cluster': 'Cluster'})

st.caption(f"{len(df_show):,} pacientes · ordenados por Risk Score descendente")

st.dataframe(
    df_show,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Risk_Score": st.column_config.NumberColumn("Risk Score", format="%.3f"),
        "Avg_Billing_Amount": st.column_config.NumberColumn("Avg Billing", format="$%.0f"),
    }
)

st.markdown("---")
st.caption("TP Segmentación de Pacientes · K-Means K=3 · Variables numéricas (V1)")
