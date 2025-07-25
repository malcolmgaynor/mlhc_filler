import streamlit as st
import pandas as pd
import numpy as np

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
    # Data from the CSV files
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
    
    # Convert to DataFrames
    df_c1 = pd.DataFrame(data_c1, index=['B.1', 'B.2', 'B.3'])
    df_c2 = pd.DataFrame(data_c2, index=['B.1', 'B.2', 'B.3'])
    df_c3 = pd.DataFrame(data_c3, index=['B.1', 'B.2', 'B.3'])
    
    return {'C.1': df_c1, 'C.2': df_c2, 'C.3': df_c3}

data = load_data()

# Helper function to style dataframes
def style_dataframe(df):
    """Apply color styling to dataframe based on values"""
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

# Create dropdown selections
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
    """Create overview table showing all interactions"""
    st.subheader("📊 Complete Data Overview")
    
    # Create tabs for each drug C
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
    """Create analysis for specific Drug A"""
    st.subheader(f"📈 Analysis: {selected_a} vs All Drug B Options")
    
    # Create a summary table
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
    
    # Display as styled table
    def color_impact(val):
        if val > 0:
            return 'background-color: lightgreen'
        elif val < 0:
            return 'background-color: lightcoral'
        else:
            return 'background-color: lightyellow'
    
    styled_summary = summary_df.style.applymap(color_impact, subset=['Impact'])
    st.dataframe(styled_summary, use_container_width=True)
    
    # Show as bar chart using st.bar_chart
    chart_data = summary_df.set_index(['Drug C', 'Drug B'])['Impact']
    st.bar_chart(chart_data)

def create_bar_analysis(selected_a=None, selected_b=None, selected_c=None):
    """Create bar chart analysis based on selections"""
    
    if selected_a and selected_b and selected_c:
        # Single comparison
        value = data[selected_c].loc[selected_b, selected_a]
        
        st.subheader(f"🎯 Specific Comparison: {selected_a} vs {selected_b} → {selected_c}")
        
        # Create a simple display
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            # Display the value prominently
            if value > 0:
                st.success(f"**Impact Value: +{value:.2f}**")
                st.write("✅ **Drug A is preferred**")
            elif value < 0:
                st.error(f"**Impact Value: {value:.2f}**")
                st.write("❌ **Drug B is preferred**")
            else:
                st.info(f"**Impact Value: {value:.2f}**")
                st.write("⚖️ **No significant difference**")
        
        # Create single-item chart
        chart_df = pd.DataFrame({
            'Comparison': [f"{selected_a} vs {selected_b}"],
            'Impact': [value]
        })
        st.bar_chart(chart_df.set_index('Comparison'))
        
        # Clinical interpretation
        description = f"Using {selected_a} instead of {selected_b} results in a {abs(value):.2f} {'increase' if value < 0 else 'decrease'} in resistance to {selected_c}. {'Drug A is preferred.' if value > 0 else 'Drug B is preferred.' if value < 0 else 'No significant difference.'}"
        st.success(f"📊 **Clinical Interpretation:** {description}")
        
        return True
    
    elif selected_a and selected_b:
        # Compare across all C drugs
        st.subheader(f"📊 Comparison: {selected_a} vs {selected_b} Across All Drug C")
        
        c_drugs = ['C.1', 'C.2', 'C.3']
        comparison_data = []
        
        for c_drug in c_drugs:
            value = data[c_drug].loc[selected_b, selected_a]
            comparison_data.append({
                'Drug C': c_drug,
                'Impact': value,
                'Preference': 'Drug A' if value > 0 else 'Drug B' if value < 0 else 'No Difference'
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Display table
        def color_impact(val):
            if val > 0:
                return 'background-color: lightgreen'
            elif val < 0:
                return 'background-color: lightcoral'
            else:
                return 'background-color: lightyellow'
        
        styled_comparison = comparison_df.style.applymap(color_impact, subset=['Impact'])
        st.dataframe(styled_comparison, use_container_width=True)
        
        # Display bar chart
        st.bar_chart(comparison_df.set_index('Drug C')['Impact'])
        
        return True
    
    elif selected_c:
        if selected_a:
            # Specific A, all B for selected C
            st.subheader(f"📊 Analysis: {selected_a} vs All Drug B → {selected_c}")
            
            b_drugs = ['B.1', 'B.2', 'B.3']
            analysis_data = []
            
            for b_drug in b_drugs:
                value = data[selected_c].loc[b_drug, selected_a]
                analysis_data.append({
                    'Comparison': f"{selected_a} vs {b_drug}",
                    'Impact': value,
                    'Preference': 'Drug A' if value > 0 else 'Drug B' if value < 0 else 'No Difference'
                })
            
            analysis_df = pd.DataFrame(analysis_data)
            
            # Display styled table
            def color_impact(val):
                if val > 0:
                    return 'background-color: lightgreen'
                elif val < 0:
                    return 'background-color: lightcoral'
                else:
                    return 'background-color: lightyellow'
            
            styled_analysis = analysis_df.style.applymap(color_impact, subset=['Impact'])
            st.dataframe(styled_analysis, use_container_width=True)
            
            # Display bar chart
            st.bar_chart(analysis_df.set_index('Comparison')['Impact'])
            
        else:
            # All A vs all B for selected C
            st.subheader(f"📊 All Comparisons → {selected_c} Resistance")
            
            df = data[selected_c]
            all_comparisons = []
            
            for b_drug in df.index:
                for a_drug in df.columns:
                    value = df.loc[b_drug, a_drug]
                    all_comparisons.append({
                        'Comparison': f"{a_drug} vs {b_drug}",
                        'Impact': value,
                        'Preference': 'Drug A' if value > 0 else 'Drug B' if value < 0 else 'No Difference'
                    })
            
            all_comp_df = pd.DataFrame(all_comparisons)
            
            # Display styled table
            def color_impact(val):
                if val > 0:
                    return 'background-color: lightgreen'
                elif val < 0:
                    return 'background-color: lightcoral'
                else:
                    return 'background-color: lightyellow'
            
            styled_all = all_comp_df.style.applymap(color_impact, subset=['Impact'])
            st.dataframe(styled_all, use_container_width=True)
            
            # Display bar chart
            st.bar_chart(all_comp_df.set_index('Comparison')['Impact'])
        
        return True
    
    return False

# Display logic based on selections
if not drug_a and not drug_b and not drug_c:
    # No inputs - show overview
    create_overview_table()
    st.info("💡 **Interpretation:** Green = Drug A preferred, Red = Drug B preferred, Yellow = No significant difference")

elif drug_a and not drug_b and not drug_c:
    # A input only - show 2D analysis
    create_2d_analysis(drug_a)
    st.info(f"💡 **Interpretation:** Shows how {drug_a} compares to all Drug B options across different Drug C resistance outcomes")

elif (drug_a and drug_b) or drug_c:
    # Bar chart scenarios
    result = create_bar_analysis(drug_a, drug_b, drug_c)
    
    if drug_a and drug_b and drug_c and result:
        st.info("💡 **TMLE Approach:** This single comparison represents the targeted maximum likelihood estimation of the average treatment effect (ATE) for developing AMR.")

# Add summary statistics
st.subheader("📈 Summary Statistics")
col1, col2, col3, col4 = st.columns(4)

# Calculate statistics
all_values = []
for df in data.values():
    all_values.extend(df.values.flatten())

all_values = np.array(all_values)

with col1:
    st.metric("Total Comparisons", "27")
    
with col2:
    positive_count = (all_values > 0).sum()
    st.metric("Drug A Preferred", f"{positive_count}/27")
    
with col3:
    negative_count = (all_values < 0).sum()
    st.metric("Drug B Preferred", f"{negative_count}/27")
    
with col4:
    avg_magnitude = np.mean(np.abs(all_values))
    st.metric("Avg Impact Magnitude", f"{avg_magnitude:.2f}")

# Quick insights
st.subheader("🔍 Quick Insights")
col1, col2 = st.columns(2)

with col1:
    # Find best Drug A option
    a_scores = {}
    for a_drug in ['A.1', 'A.2', 'A.3']:
        total_score = 0
        for c_drug in data.keys():
            for b_drug in data[c_drug].index:
                total_score += data[c_drug].loc[b_drug, a_drug]
        a_scores[a_drug] = total_score
    
    best_a = max(a_scores.keys(), key=lambda x: a_scores[x])
    st.success(f"**Best Drug A Overall:** {best_a} (Score: {a_scores[best_a]:.2f})")

with col2:
    # Find most resistant Drug C
    c_impacts = {}
    for c_drug in data.keys():
        avg_impact = np.mean(data[c_drug].values.flatten())
        c_impacts[c_drug] = avg_impact
    
    most_resistant = min(c_impacts.keys(), key=lambda x: c_impacts[x])
    st.warning(f"**Most Concerning Resistance:** {most_resistant} (Avg: {c_impacts[most_resistant]:.2f})")

# Data summary
with st.expander("📊 View Raw Data with Color Coding"):
    st.subheader("Raw Impact Data")
    for c_drug, df in data.items():
        st.write(f"**{c_drug} Resistance Impact:**")
        st.dataframe(style_dataframe(df), use_container_width=True)
        st.write("")

# Methodology note
with st.expander("🔬 Methodology"):
    st.markdown("""
    **Targeted Maximum Likelihood Estimation (TMLE) for Antimicrobial Resistance**
    
    - **Objective:** Estimate the average treatment effect (ATE) of choosing antibiotic A vs B on future resistance to antibiotic C
    - **Data:** Observational data with inclusion/exclusion criteria similar to clinical trials
    - **Values:** Positive numbers indicate Drug A is preferred over Drug B
    - **Clinical Application:** Helps clinicians make evidence-based decisions about antibiotic selection to minimize future resistance development
    - **Color Coding:** 
      - 🟢 Green = Drug A preferred (positive values)
      - 🔴 Red = Drug B preferred (negative values)  
      - 🟡 Yellow = No significant difference (zero values)
    """)

# Instructions
with st.expander("📋 How to Use This Dashboard"):
    st.markdown("""
    **Selection Options:**
    
    1. **No Selection:** View complete overview with color-coded tables for all drug interactions
    2. **Select Drug A Only:** See detailed analysis of how your selected Drug A performs against all Drug B options
    3. **Select Drug A + B:** Compare this specific pair across all Drug C resistance outcomes  
    4. **Select Drug C Only:** See all Drug A vs B combinations for that specific resistance outcome
    5. **Select A + C:** See how Drug A performs vs all Drug B for specific resistance outcome
    6. **Select All Three:** Get specific comparison with detailed clinical interpretation
    
    **Color Coding:**
    - Green backgrounds/positive values = Drug A is preferred
    - Red backgrounds/negative values = Drug B is preferred  
    - Yellow backgrounds/zero values = No significant difference
    """)
