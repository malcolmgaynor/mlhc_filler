import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from io import StringIO

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

# Determine what to display based on selections
def create_3d_heatmap():
    """Create 3D heatmap showing all data"""
    fig = go.Figure()
    
    # Create data for 3D surface
    c_drugs = ['C.1', 'C.2', 'C.3']
    a_drugs = ['A.1', 'A.2', 'A.3']
    b_drugs = ['B.1', 'B.2', 'B.3']
    
    colors = ['red', 'green', 'blue']
    
    for i, c_drug in enumerate(c_drugs):
        df = data[c_drug]
        z_values = []
        
        for b_drug in b_drugs:
            row = []
            for a_drug in a_drugs:
                row.append(df.loc[b_drug, a_drug])
            z_values.append(row)
        
        # Create surface for each C drug
        fig.add_trace(go.Surface(
            z=z_values,
            x=a_drugs,
            y=b_drugs,
            name=c_drug,
            colorscale=[[0, colors[i]], [1, colors[i]]],
            opacity=0.7,
            showscale=True if i == 0 else False
        ))
    
    fig.update_layout(
        title="3D Heatmap: Drug A vs Drug B Impact on Drug C Resistance",
        scene=dict(
            xaxis_title="Drug A",
            yaxis_title="Drug B", 
            zaxis_title="Impact Value",
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
        ),
        width=800,
        height=600
    )
    
    return fig

def create_2d_heatmap(selected_a=None):
    """Create 2D heatmap for specific Drug A vs all Drug B and C combinations"""
    if selected_a:
        # Show specific A vs all B, across all C
        fig_data = []
        c_drugs = ['C.1', 'C.2', 'C.3']
        b_drugs = ['B.1', 'B.2', 'B.3']
        
        z_matrix = []
        for c_drug in c_drugs:
            row = []
            for b_drug in b_drugs:
                row.append(data[c_drug].loc[b_drug, selected_a])
            z_matrix.append(row)
        
        fig = go.Figure(data=go.Heatmap(
            z=z_matrix,
            x=b_drugs,
            y=c_drugs,
            colorscale='RdBu',
            zmid=0,
            text=z_matrix,
            texttemplate="%{text:.2f}",
            textfont={"size": 12}
        ))
        
        fig.update_layout(
            title=f"Impact of {selected_a} vs Drug B on Drug C Resistance",
            xaxis_title="Drug B",
            yaxis_title="Drug C",
            width=600,
            height=400
        )
        
        return fig
    else:
        return create_3d_heatmap()

def create_bar_chart(selected_a=None, selected_b=None, selected_c=None):
    """Create bar chart based on selections"""
    
    if selected_a and selected_b and selected_c:
        # Single comparison
        value = data[selected_c].loc[selected_b, selected_a]
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=[value],
            y=[f"{selected_a} vs {selected_b} → {selected_c}"],
            orientation='h',
            marker_color='green' if value > 0 else 'red',
            text=[f"{value:.2f}"],
            textposition='auto'
        ))
        
        fig.update_layout(
            title=f"Impact: {selected_a} vs {selected_b} on {selected_c} Resistance",
            xaxis_title="Impact Value (Positive = Drug A Preferred)",
            width=600,
            height=200,
            xaxis=dict(range=[-3, 3])
        )
        
        return fig, f"Using {selected_a} instead of {selected_b} results in a {abs(value):.2f} {'increase' if value < 0 else 'decrease'} in resistance to {selected_c}. {'Drug A is preferred.' if value > 0 else 'Drug B is preferred.' if value < 0 else 'No significant difference.'}"
    
    elif selected_a and selected_b:
        # Compare across all C drugs
        c_drugs = ['C.1', 'C.2', 'C.3']
        values = []
        labels = []
        
        for c_drug in c_drugs:
            value = data[c_drug].loc[selected_b, selected_a]
            values.append(value)
            labels.append(f"{selected_a} vs {selected_b} → {c_drug}")
        
        fig = go.Figure()
        colors = ['green' if v > 0 else 'red' for v in values]
        
        fig.add_trace(go.Bar(
            x=values,
            y=labels,
            orientation='h',
            marker_color=colors,
            text=[f"{v:.2f}" for v in values],
            textposition='auto'
        ))
        
        fig.update_layout(
            title=f"Impact: {selected_a} vs {selected_b} on All Drug C Resistance",
            xaxis_title="Impact Value (Positive = Drug A Preferred)",
            width=700,
            height=300
        )
        
        return fig, None
    
    elif selected_c:
        # Show all A vs B combinations for selected C
        if selected_a:
            # Specific A, all B for selected C
            b_drugs = ['B.1', 'B.2', 'B.3']
            values = []
            labels = []
            
            for b_drug in b_drugs:
                value = data[selected_c].loc[b_drug, selected_a]
                values.append(value)
                labels.append(f"{selected_a} vs {b_drug}")
            
            fig = go.Figure()
            colors = ['green' if v > 0 else 'red' for v in values]
            
            fig.add_trace(go.Bar(
                x=values,
                y=labels,
                orientation='h',
                marker_color=colors,
                text=[f"{v:.2f}" for v in values],
                textposition='auto'
            ))
            
            fig.update_layout(
                title=f"Impact: {selected_a} vs All Drug B on {selected_c} Resistance",
                xaxis_title="Impact Value (Positive = Drug A Preferred)",
                width=700,
                height=300
            )
            
            return fig, None
        else:
            # All A vs all B for selected C
            df = data[selected_c]
            values = []
            labels = []
            
            for b_drug in df.index:
                for a_drug in df.columns:
                    value = df.loc[b_drug, a_drug]
                    values.append(value)
                    labels.append(f"{a_drug} vs {b_drug}")
            
            fig = go.Figure()
            colors = ['green' if v > 0 else 'red' for v in values]
            
            fig.add_trace(go.Bar(
                x=values,
                y=labels,
                orientation='h',
                marker_color=colors,
                text=[f"{v:.2f}" for v in values],
                textposition='auto'
            ))
            
            fig.update_layout(
                title=f"All Drug A vs Drug B Impact on {selected_c} Resistance",
                xaxis_title="Impact Value (Positive = Drug A Preferred)",
                width=700,
                height=500
            )
            
            return fig, None

# Display logic based on selections
if not drug_a and not drug_b and not drug_c:
    # No inputs - show 3D heatmap
    st.subheader("3D Overview: All Drug Interactions")
    fig = create_3d_heatmap()
    st.plotly_chart(fig, use_container_width=True)
    
    st.info("💡 **Interpretation:** Positive values indicate Drug A is preferred over Drug B for reducing resistance to Drug C")

elif drug_a and not drug_b and not drug_c:
    # A input only - show 2D heatmap
    st.subheader(f"Impact Analysis: {drug_a} vs All Drug B Options")
    fig = create_2d_heatmap(drug_a)
    st.plotly_chart(fig, use_container_width=True)
    
    st.info(f"💡 **Interpretation:** Shows how {drug_a} compares to all Drug B options across different Drug C resistance outcomes")

elif (drug_a and drug_b) or drug_c:
    # Bar chart scenarios
    st.subheader("Comparative Analysis")
    result = create_bar_chart(drug_a, drug_b, drug_c)
    
    if result:
        fig, description = result
        st.plotly_chart(fig, use_container_width=True)
        
        if description:
            st.success(f"📊 **Clinical Interpretation:** {description}")
    
    if drug_a and drug_b and drug_c:
        st.info("💡 **TMLE Approach:** This single comparison represents the targeted maximum likelihood estimation of the average treatment effect (ATE) for developing AMR.")

# Data summary
with st.expander("📊 View Raw Data"):
    st.subheader("Raw Impact Data")
    for c_drug, df in data.items():
        st.write(f"**{c_drug} Resistance Impact:**")
        st.dataframe(df)
        st.write("")

# Methodology note
with st.expander("🔬 Methodology"):
    st.markdown("""
    **Targeted Maximum Likelihood Estimation (TMLE) for Antimicrobial Resistance**
    
    - **Objective:** Estimate the average treatment effect (ATE) of choosing antibiotic A vs B on future resistance to antibiotic C
    - **Data:** Observational data with inclusion/exclusion criteria similar to clinical trials
    - **Values:** Positive numbers indicate Drug A is preferred over Drug B
    - **Clinical Application:** Helps clinicians make evidence-based decisions about antibiotic selection to minimize future resistance development
    """)
