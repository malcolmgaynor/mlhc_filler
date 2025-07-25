import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

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

# Helper function to create 3D visualization using matplotlib
def create_3d_plot():
    """Create 3D scatter plot showing all data points"""
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    c_drugs = ['C.1', 'C.2', 'C.3']
    a_drugs = ['A.1', 'A.2', 'A.3']
    b_drugs = ['B.1', 'B.2', 'B.3']
    
    colors = ['red', 'green', 'blue']
    
    for i, c_drug in enumerate(c_drugs):
        df = data[c_drug]
        
        # Create coordinate arrays
        a_coords = []
        b_coords = []
        values = []
        
        for j, a_drug in enumerate(a_drugs):
            for k, b_drug in enumerate(b_drugs):
                a_coords.append(j)
                b_coords.append(k)
                values.append(df.loc[b_drug, a_drug])
        
        # Create scatter plot for this C drug
        scatter = ax.scatter(a_coords, b_coords, values, 
                           c=colors[i], label=c_drug, s=100, alpha=0.7)
        
        # Add value labels
        for j, (a, b, v) in enumerate(zip(a_coords, b_coords, values)):
            ax.text(a, b, v, f'{v:.1f}', fontsize=8)
    
    ax.set_xlabel('Drug A')
    ax.set_ylabel('Drug B') 
    ax.set_zlabel('Impact Value')
    ax.set_title('3D View: Drug A vs Drug B Impact on Drug C Resistance')
    
    # Set tick labels
    ax.set_xticks(range(len(a_drugs)))
    ax.set_xticklabels(a_drugs)
    ax.set_yticks(range(len(b_drugs)))
    ax.set_yticklabels(b_drugs)
    
    ax.legend()
    
    return fig

def create_2d_heatmap(selected_a=None):
    """Create 2D heatmap for specific Drug A vs all Drug B and C combinations"""
    if selected_a:
        # Show specific A vs all B, across all C
        c_drugs = ['C.1', 'C.2', 'C.3']
        b_drugs = ['B.1', 'B.2', 'B.3']
        
        z_matrix = []
        for c_drug in c_drugs:
            row = []
            for b_drug in b_drugs:
                row.append(data[c_drug].loc[b_drug, selected_a])
            z_matrix.append(row)
        
        df_heatmap = pd.DataFrame(z_matrix, index=c_drugs, columns=b_drugs)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(df_heatmap, annot=True, cmap='RdBu_r', center=0, 
                   ax=ax, fmt='.2f', cbar_kws={'label': 'Impact Value'})
        ax.set_title(f'Impact of {selected_a} vs Drug B on Drug C Resistance')
        ax.set_xlabel('Drug B')
        ax.set_ylabel('Drug C')
        
        return fig
    else:
        return create_3d_plot()

def create_bar_chart(selected_a=None, selected_b=None, selected_c=None):
    """Create bar chart based on selections"""
    
    if selected_a and selected_b and selected_c:
        # Single comparison
        value = data[selected_c].loc[selected_b, selected_a]
        
        fig, ax = plt.subplots(figsize=(8, 3))
        color = 'green' if value > 0 else 'red'
        
        bars = ax.barh([f"{selected_a} vs {selected_b} → {selected_c}"], [value], 
                      color=color, alpha=0.7)
        
        # Add value label on bar
        ax.text(value/2 if value != 0 else 0.1, 0, f'{value:.2f}', 
               ha='center', va='center', fontweight='bold')
        
        ax.set_xlabel('Impact Value (Positive = Drug A Preferred)')
        ax.set_title(f'Impact: {selected_a} vs {selected_b} on {selected_c} Resistance')
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        ax.set_xlim(-3, 3)
        
        description = f"Using {selected_a} instead of {selected_b} results in a {abs(value):.2f} {'increase' if value < 0 else 'decrease'} in resistance to {selected_c}. {'Drug A is preferred.' if value > 0 else 'Drug B is preferred.' if value < 0 else 'No significant difference.'}"
        
        return fig, description
    
    elif selected_a and selected_b:
        # Compare across all C drugs
        c_drugs = ['C.1', 'C.2', 'C.3']
        values = []
        labels = []
        colors = []
        
        for c_drug in c_drugs:
            value = data[c_drug].loc[selected_b, selected_a]
            values.append(value)
            labels.append(f"{selected_a} vs {selected_b} → {c_drug}")
            colors.append('green' if value > 0 else 'red')
        
        fig, ax = plt.subplots(figsize=(10, 4))
        bars = ax.barh(labels, values, color=colors, alpha=0.7)
        
        # Add value labels
        for i, (bar, value) in enumerate(zip(bars, values)):
            ax.text(value/2 if value != 0 else 0.1, i, f'{value:.2f}', 
                   ha='center', va='center', fontweight='bold')
        
        ax.set_xlabel('Impact Value (Positive = Drug A Preferred)')
        ax.set_title(f'Impact: {selected_a} vs {selected_b} on All Drug C Resistance')
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        
        return fig, None
    
    elif selected_c:
        # Show all A vs B combinations for selected C
        if selected_a:
            # Specific A, all B for selected C
            b_drugs = ['B.1', 'B.2', 'B.3']
            values = []
            labels = []
            colors = []
            
            for b_drug in b_drugs:
                value = data[selected_c].loc[b_drug, selected_a]
                values.append(value)
                labels.append(f"{selected_a} vs {b_drug}")
                colors.append('green' if value > 0 else 'red')
            
            fig, ax = plt.subplots(figsize=(10, 4))
            bars = ax.barh(labels, values, color=colors, alpha=0.7)
            
            # Add value labels
            for i, (bar, value) in enumerate(zip(bars, values)):
                ax.text(value/2 if value != 0 else 0.1, i, f'{value:.2f}', 
                       ha='center', va='center', fontweight='bold')
            
            ax.set_xlabel('Impact Value (Positive = Drug A Preferred)')
            ax.set_title(f'Impact: {selected_a} vs All Drug B on {selected_c} Resistance')
            ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
            
            return fig, None
        else:
            # All A vs all B for selected C
            df = data[selected_c]
            values = []
            labels = []
            colors = []
            
            for b_drug in df.index:
                for a_drug in df.columns:
                    value = df.loc[b_drug, a_drug]
                    values.append(value)
                    labels.append(f"{a_drug} vs {b_drug}")
                    colors.append('green' if value > 0 else 'red')
            
            fig, ax = plt.subplots(figsize=(10, 8))
            bars = ax.barh(labels, values, color=colors, alpha=0.7)
            
            # Add value labels
            for i, (bar, value) in enumerate(zip(bars, values)):
                ax.text(value/2 if value != 0 else 0.1, i, f'{value:.2f}', 
                       ha='center', va='center', fontweight='bold')
            
            ax.set_xlabel('Impact Value (Positive = Drug A Preferred)')
            ax.set_title(f'All Drug A vs Drug B Impact on {selected_c} Resistance')
            ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
            
            return fig, None

# Display logic based on selections
if not drug_a and not drug_b and not drug_c:
    # No inputs - show 3D plot
    st.subheader("3D Overview: All Drug Interactions")
    fig = create_3d_plot()
    st.pyplot(fig)
    
    st.info("💡 **Interpretation:** Positive values indicate Drug A is preferred over Drug B for reducing resistance to Drug C")

elif drug_a and not drug_b and not drug_c:
    # A input only - show 2D heatmap
    st.subheader(f"Impact Analysis: {drug_a} vs All Drug B Options")
    fig = create_2d_heatmap(drug_a)
    st.pyplot(fig)
    
    st.info(f"💡 **Interpretation:** Shows how {drug_a} compares to all Drug B options across different Drug C resistance outcomes")

elif (drug_a and drug_b) or drug_c:
    # Bar chart scenarios
    st.subheader("Comparative Analysis")
    result = create_bar_chart(drug_a, drug_b, drug_c)
    
    if result:
        fig, description = result
        st.pyplot(fig)
        
        if description:
            st.success(f"📊 **Clinical Interpretation:** {description}")
    
    if drug_a and drug_b and drug_c:
        st.info("💡 **TMLE Approach:** This single comparison represents the targeted maximum likelihood estimation of the average treatment effect (ATE) for developing AMR.")

# Add summary statistics
st.subheader("📈 Summary Statistics")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Comparisons", "27")
    
with col2:
    # Calculate percentage of positive values (Drug A preferred)
    all_values = []
    for df in data.values():
        all_values.extend(df.values.flatten())
    positive_pct = (np.array(all_values) > 0).mean() * 100
    st.metric("Drug A Preferred (%)", f"{positive_pct:.1f}%")
    
with col3:
    # Calculate average impact magnitude
    avg_magnitude = np.mean(np.abs(all_values))
    st.metric("Avg Impact Magnitude", f"{avg_magnitude:.2f}")

# Data summary
with st.expander("📊 View Raw Data"):
    st.subheader("Raw Impact Data")
    for c_drug, df in data.items():
        st.write(f"**{c_drug} Resistance Impact:**")
        st.dataframe(df, use_container_width=True)
        st.write("")

# Methodology note
with st.expander("🔬 Methodology"):
    st.markdown("""
    **Targeted Maximum Likelihood Estimation (TMLE) for Antimicrobial Resistance**
    
    - **Objective:** Estimate the average treatment effect (ATE) of choosing antibiotic A vs B on future resistance to antibiotic C
    - **Data:** Observational data with inclusion/exclusion criteria similar to clinical trials
    - **Values:** Positive numbers indicate Drug A is preferred over Drug B
    - **Clinical Application:** Helps clinicians make evidence-based decisions about antibiotic selection to minimize future resistance development
    - **Color Coding:** Green bars = Drug A preferred, Red bars = Drug B preferred
    """)

# Instructions
with st.expander("📋 How to Use"):
    st.markdown("""
    1. **No Selection:** View 3D overview of all drug interactions
    2. **Select Drug A Only:** See heatmap comparing selected Drug A vs all Drug B options
    3. **Select Drug A + B:** Compare the pair across all Drug C resistance outcomes  
    4. **Select Drug C Only:** See all Drug A vs B combinations for that resistance outcome
    5. **Select A + C:** See Drug A vs all Drug B for specific resistance outcome
    6. **Select All Three:** Get specific comparison with clinical interpretation
    """)
