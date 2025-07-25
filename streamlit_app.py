import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(
    page_title="Antimicrobial Resistance Impact Dashboard",
    page_icon="🧬",
    layout="wide"
)

st.title("🧬 Antimicrobial Resistance Impact Dashboard")
st.markdown("**Goal:** Help clinicians choose between antibiotics by estimating impact on future resistance")

# Load the data
@st.cache_data
def load_data():
    data_c1 = {
        'A.1': [1, 0.5, -1],
        'A.2': [-1, 0.5, -1], 
        'A.3': [2, -2, 1.5]
    }
    data_c2 = {
        'A.1': [0.75, -2, 1],
        'A.2': [0.75, -2, -1],
        'A.3': [-3, -0.5, 1]  
    }
    data_c3 = {
        'A.1': [1, -0.5, -1],
        'A.2': [1, 2, -2],
        'A.3': [1, -2, 0.5]
    }
    df_c1 = pd.DataFrame(data_c1, index=['B.1', 'B.2', 'B.3'])
    df_c2 = pd.DataFrame(data_c2, index=['B.1', 'B.2', 'B.3'])
    df_c3 = pd.DataFrame(data_c3, index=['B.1', 'B.2', 'B.3'])
    return {'C.1': df_c1, 'C.2': df_c2, 'C.3': df_c3}

data = load_data()

def style_dataframe(df):
    def color_negative_red(val):
        if isinstance(val, (int, float)):
            if val > 0:
                return 'background-color: lightgreen'
            elif val < 0:
                return 'background-color: lightcoral'
            else:
                return 'background-color: lightyellow'
        return ''
    return df.style.applymap(color_negative_red)

def plot_centered_bar_chart(df, x_col, y_col, title):
    fig, ax = plt.subplots(figsize=(6, 3))
    max_val = df[x_col].abs().max()
    colors = df[x_col].apply(lambda x: 'green' if x > 0 else 'red' if x < 0 else 'gold')
    ax.barh(df[y_col], df[x_col], color=colors)
    ax.axvline(0, color='black', linewidth=1)
    ax.set_xlim(-max_val*1.1, max_val*1.1)
    ax.set_xlabel("Impact (A vs. B)")
    ax.set_title(title)
    st.pyplot(fig)

# Selections
st.subheader("Selection Controls")
col1, col2, col3 = st.columns(3)
with col1:
    drug_a = st.selectbox("Select Drug A", [''] + ['A.1', 'A.2', 'A.3'], index=0)
with col2:
    drug_b = st.selectbox("Select Drug B", [''] + ['B.1', 'B.2', 'B.3'], index=0)
with col3:
    drug_c = st.selectbox("Select Drug C", [''] + ['C.1', 'C.2', 'C.3'], index=0)
st.markdown("---")

def create_overview_table():
    st.subheader("📊 Complete Data Overview")
    tab1, tab2, tab3 = st.tabs(["C.1 Resistance", "C.2 Resistance", "C.3 Resistance"])
    with tab1:
        st.write("**Impact on C.1 Resistance:**")
        st.dataframe(style_dataframe(data['C.1']), use_container_width=True)
    with tab2:
        st.write("**Impact on C.2 Resistance:**")
        st.dataframe(style_dataframe(data['C.2']), use_container_width=True)
    with tab3:
        st.write("**Impact on C.3 Resistance:**")
        st.dataframe(style_dataframe(data['C.3']), use_container_width=True)

def create_2d_analysis(selected_a):
    st.subheader(f"📈 Analysis: {selected_a} vs All Drug B Options")
    summary_data = []
    for c_drug in ['C.1', 'C.2', 'C.3']:
        for b_drug in ['B.1', 'B.2', 'B.3']:
            value = data[c_drug].loc[b_drug, selected_a]
            summary_data.append({
                'Drug C': c_drug,
                'Drug B': b_drug,
                'Impact': value,
                'Interpretation': 'Drug A Preferred' if value > 0 else 'Drug B Preferred' if value < 0 else 'No Difference'
            })
    summary_df = pd.DataFrame(summary_data)
    styled_summary = summary_df.style.applymap(lambda v: 'background-color: lightgreen' if v > 0 else 'background-color: lightcoral' if v < 0 else 'background-color: lightyellow', subset=['Impact'])
    st.dataframe(styled_summary, use_container_width=True)
    summary_df['Label'] = summary_df['Drug C'] + " vs " + summary_df['Drug B']
    plot_centered_bar_chart(summary_df, x_col='Impact', y_col='Label', title=f"Impact of {selected_a} vs All Drug B Options")

def create_bar_analysis(selected_a=None, selected_b=None, selected_c=None):
    if selected_a and selected_b and selected_c:
        value = data[selected_c].loc[selected_b, selected_a]
        st.subheader(f"🎯 Specific Comparison: {selected_a} vs {selected_b} → {selected_c}")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if value > 0:
                st.success(f"**Impact Value: +{value:.2f}**")
                st.write("✅ **Drug A is preferred**")
            elif value < 0:
                st.error(f"**Impact Value: {value:.2f}**")
                st.write("❌ **Drug B is preferred**")
            else:
                st.info(f"**Impact Value: {value:.2f}**")
                st.write("⚖️ **No significant difference**")
        single_df = pd.DataFrame({'Comparison': [f"{selected_a} vs {selected_b}"], 'Impact': [value]})
        plot_centered_bar_chart(single_df, x_col='Impact', y_col='Comparison', title=f"{selected_a} vs {selected_b} on {selected_c}")
        desc = f"Using {selected_a} instead of {selected_b} results in a {abs(value):.2f} {'increase' if value < 0 else 'decrease'} in resistance to {selected_c}. {'Drug A is preferred.' if value > 0 else 'Drug B is preferred.' if value < 0 else 'No significant difference.'}"
        st.success(f"📊 **Clinical Interpretation:** {desc}")
        return True

    elif selected_a and selected_b:
        st.subheader(f"📊 Comparison: {selected_a} vs {selected_b} Across All Drug C")
        c_drugs = ['C.1', 'C.2', 'C.3']
        comp_data = []
        for c_drug in c_drugs:
            value = data[c_drug].loc[selected_b, selected_a]
            comp_data.append({'Drug C': c_drug, 'Impact': value, 'Preference': 'Drug A' if value > 0 else 'Drug B' if value < 0 else 'No Difference'})
        comp_df = pd.DataFrame(comp_data)
        st.dataframe(comp_df.style.applymap(lambda v: 'background-color: lightgreen' if v > 0 else 'background-color: lightcoral' if v < 0 else 'background-color: lightyellow', subset=['Impact']), use_container_width=True)
        plot_centered_bar_chart(comp_df, x_col='Impact', y_col='Drug C', title=f"{selected_a} vs {selected_b} Across All C")
        return True

    elif selected_c:
        if selected_a:
            st.subheader(f"📊 Analysis: {selected_a} vs All Drug B → {selected_c}")
            b_drugs = ['B.1', 'B.2', 'B.3']
            analysis_data = []
            for b_drug in b_drugs:
                value = data[selected_c].loc[b_drug, selected_a]
                analysis_data.append({'Comparison': f"{selected_a} vs {b_drug}", 'Impact': value})
            analysis_df = pd.DataFrame(analysis_data)
            st.dataframe(analysis_df.style.applymap(lambda v: 'background-color: lightgreen' if v > 0 else 'background-color: lightcoral' if v < 0 else 'background-color: lightyellow', subset=['Impact']), use_container_width=True)
            plot_centered_bar_chart(analysis_df, x_col='Impact', y_col='Comparison', title=f"{selected_a} vs All B on {selected_c}")
        else:
            st.subheader(f"📊 All Comparisons → {selected_c} Resistance")
            df = data[selected_c]
            all_data = []
            for b in df.index:
                for a in df.columns:
                    val = df.loc[b, a]
                    all_data.append({'Comparison': f"{a} vs {b}", 'Impact': val})
            all_df = pd.DataFrame(all_data)
            st.dataframe(all_df.style.applymap(lambda v: 'background-color: lightgreen' if v > 0 else 'background-color: lightcoral' if v < 0 else 'background-color: lightyellow', subset=['Impact']), use_container_width=True)
            plot_centered_bar_chart(all_df, x_col='Impact', y_col='Comparison', title=f"All A vs B on {selected_c}")
        return True

    return False

if not drug_a and not drug_b and not drug_c:
    create_overview_table()
    st.info("💡 **Interpretation:** Green = Drug A preferred, Red = Drug B preferred, Yellow = No significant difference")

elif drug_a and not drug_b and not drug_c:
    create_2d_analysis(drug_a)
    st.info(f"💡 **Interpretation:** Shows how {drug_a} compares to all Drug B options across different Drug C resistance outcomes")

elif (drug_a and drug_b) or drug_c:
    result = create_bar_analysis(drug_a, drug_b, drug_c)
    if drug_a and drug_b and drug_c and result:
        st.info("💡 **TMLE Approach:** This single comparison represents the targeted maximum likelihood estimation of the average treatment effect (ATE) for developing AMR.")

# Summary statistics
st.subheader("📈 Summary Statistics")
col1, col2, col3, col4 = st.columns(4)
all_values = np.concatenate([df.values.flatten() for df in data.values()])
with col1:
    st.metric("Total Comparisons", "27")
with col2:
    st.metric("Drug A Preferred", f"{(all_values > 0).sum()}/27")
with col3:
    st.metric("Drug B Preferred", f"{(all_values < 0).sum()}/27")
with col4:
    st.metric("Avg Impact Magnitude", f"{np.mean(np.abs(all_values)):.2f}")

# Quick Insights
st.subheader("🔍 Quick Insights")
col1, col2 = st.columns(2)
with col1:
    a_scores = {a: sum(data[c].loc[b, a] for c in data for b in data[c].index) for a in ['A.1', 'A.2', 'A.3']}
    best_a = max(a_scores, key=a_scores.get)
    st.success(f"**Best Drug A Overall:** {best_a} (Score: {a_scores[best_a]:.2f})")
with col2:
    c_avg = {c: np.mean(df.values.flatten()) for c, df in data.items()}
    worst_c = min(c_avg, key=c_avg.get)
    st.warning(f"**Most Concerning Resistance:** {worst_c} (Avg: {c_avg[worst_c]:.2f})")

with st.expander("📊 View Raw Data with Color Coding"):
    st.subheader("Raw Impact Data")
    for c_drug, df in data.items():
        st.write(f"**{c_drug} Resistance Impact:**")
        st.dataframe(style_dataframe(df), use_container_width=True)

with st.expander("🔬 Methodology"):
    st.markdown("""
    **Targeted Maximum Likelihood Estimation (TMLE) for Antimicrobial Resistance**
    
    - **Objective:** Estimate the average treatment effect (ATE) of choosing antibiotic A vs B on future resistance to antibiotic C
    - **Values:** Positive = Drug A preferred; Negative = Drug B preferred
    """)

with st.expander("📋 How to Use This Dashboard"):
    st.markdown("""
    - No selection = overview
    - A only = A vs all B
    - A + B = compare across all C
    - A + B + C = single comparison
    """)
