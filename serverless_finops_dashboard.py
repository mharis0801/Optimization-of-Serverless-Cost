import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import os
from io import StringIO
warnings.filterwarnings('ignore')

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="FinOps Serverless Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CUSTOM CSS ====================
st.markdown("""
    <style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .section-header {
        font-size: 28px;
        font-weight: bold;
        color: #1f77b4;
        margin: 20px 0;
        border-bottom: 3px solid #1f77b4;
        padding-bottom: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# ==================== LOAD DATA ====================
@st.cache_data
def load_data():
    # Try multiple file path locations
    possible_paths = [
        'Serverless_Data.csv',
        './Serverless_Data.csv',
        '../Serverless_Data.csv',
        os.path.expanduser('~/Desktop/Cloud Economics/Optimization of Serverless Cost/Serverless_Data.csv'),
        '/Users/mharis/Desktop/Cloud Economics/Optimization of Serverless Cost/Serverless_Data.csv'
    ]
    
    df = None
    for path in possible_paths:
        if os.path.exists(path):
            try:
                # Read raw content first
                with open(path, 'r') as f:
                    content = f.read()
                
                # Check if rows are wrapped in quotes (unusual CSV format)
                lines = content.strip().split('\n')
                if lines[0].startswith('"') and lines[0].endswith('"'):
                    # Remove quotes from each line
                    cleaned_lines = []
                    for line in lines:
                        cleaned_line = line.strip()
                        if cleaned_line.startswith('"') and cleaned_line.endswith('"'):
                            cleaned_line = cleaned_line[1:-1]
                        cleaned_lines.append(cleaned_line)
                    cleaned_content = '\n'.join(cleaned_lines)
                    df = pd.read_csv(StringIO(cleaned_content))
                else:
                    # Normal CSV format
                    df = pd.read_csv(path)
                
                st.sidebar.success(f"✅ Loaded {len(df)} functions from: {os.path.basename(path)}")
                return df
            except Exception as e:
                st.sidebar.error(f"Error loading {path}: {str(e)}")
                continue
    
    # If no file found, show error and guidance
    if df is None:
        st.error(f"""
        ❌ **Serverless_Data.csv not found!**
        
        **How to fix:**
        1. Make sure `Serverless_Data.csv` is in the SAME folder as `serverless_finops_dashboard.py`
        2. Check the file name spelling (case-sensitive)
        3. **Current working directory:** `{os.getcwd()}`
        
        **Files in current directory:**
        {os.listdir('.')}
        """)
        st.stop()

df = load_data()

# Verify columns exist
required_columns = ['FunctionName', 'Environment', 'InvocationsPerMonth', 'AvgDurationMs', 
                   'MemoryMB', 'ColdStartRate', 'ProvisionedConcurrency', 'GBSeconds', 
                   'DataTransferGB', 'CostUSD']

missing_columns = [col for col in required_columns if col not in df.columns]
if missing_columns:
    st.error(f"❌ Missing columns in CSV: {missing_columns}")
    st.info(f"**Columns found in CSV:** {df.columns.tolist()}")
    st.info(f"**Expected columns:** {required_columns}")
    st.write("**First 3 rows of CSV:**")
    st.dataframe(df.head(3))
    st.stop()

# ==================== DATA PREPARATION ====================
df['ColdStartRate_pct'] = df['ColdStartRate'] * 100
df['DataTransferCost'] = df['DataTransferGB'] * 0.02  # Approx $0.02 per GB
df['ComputeCost'] = df['CostUSD'] - df['DataTransferCost']

# Environment filtering
environments = df['Environment'].unique()

# ==================== SIDEBAR FILTERS ====================
st.sidebar.markdown("### 🎯 Dashboard Filters")
selected_environments = st.sidebar.multiselect(
    "Select Environments",
    options=list(environments),
    default=list(environments)
)
filtered_df = df[df['Environment'].isin(selected_environments)].copy()

# ==================== MAIN DASHBOARD ====================
st.title("☁️ FinOps: Serverless Computing Cost Analysis")
st.markdown("**RetailNova** - Cloud Economics Optimization Dashboard")

# Key metrics row
col1, col2, col3, col4 = st.columns(4)
with col1:
    total_cost = filtered_df['CostUSD'].sum()
    st.metric("Total Monthly Cost", f"${total_cost:,.2f}", 
              delta=f"From {len(filtered_df)} functions")
with col2:
    avg_invocations = filtered_df['InvocationsPerMonth'].mean()
    st.metric("Avg Monthly Invocations", f"{avg_invocations:,.0f}")
with col3:
    total_invocations = filtered_df['InvocationsPerMonth'].sum()
    st.metric("Total Invocations", f"{total_invocations:,.0f}")
with col4:
    avg_duration = filtered_df['AvgDurationMs'].mean()
    st.metric("Avg Duration (ms)", f"{avg_duration:.2f}")

st.divider()

# ==================== EXERCISE 1: TOP COST CONTRIBUTORS (80/20 RULE) ====================
st.markdown("### 📈 Exercise 1: Identify Top Cost Contributors (Pareto Analysis)")

# Calculate cumulative cost percentage
cost_sorted = filtered_df.groupby('FunctionName')['CostUSD'].sum().sort_values(ascending=False)
cumulative_pct = (cost_sorted.cumsum() / cost_sorted.sum() * 100)
top_80_contributors = cost_sorted[cumulative_pct <= 80]
functions_for_80 = len(top_80_contributors)
total_functions = len(cost_sorted)

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Functions Contributing 80%", f"{functions_for_80} of {total_functions}", 
              delta=f"{(functions_for_80/total_functions*100):.1f}% of functions")
with col2:
    cost_80pct = top_80_contributors.sum()
    st.metric("80% of Costs", f"${cost_80pct:,.2f}")
with col3:
    remaining_20pct = cost_sorted.sum() - cost_80pct
    st.metric("Remaining 20% Costs", f"${remaining_20pct:,.2f}")

# Create Pareto chart
fig_pareto = make_subplots(specs=[[{"secondary_y": True}]])
fig_pareto.add_trace(
    go.Bar(x=cost_sorted.index[:15], y=cost_sorted.values[:15], 
           name="Monthly Cost", marker_color='#1f77b4'),
    secondary_y=False
)
fig_pareto.add_trace(
    go.Scatter(x=cost_sorted.index[:15], y=cumulative_pct[:15],
               name="Cumulative %", mode='lines+markers',
               line=dict(color='#ff7f0e', width=3),
               marker=dict(size=8)),
    secondary_y=True
)
fig_pareto.update_layout(
    title="Top 15 Functions - Pareto Analysis",
    xaxis_title="Function Name",
    height=450,
    hovermode='x unified'
)
fig_pareto.update_yaxes(title_text="Cost (USD)", secondary_y=False)
fig_pareto.update_yaxes(title_text="Cumulative %", secondary_y=True)
st.plotly_chart(fig_pareto, use_container_width=True)

# Cost vs Invocation Frequency
col1, col2 = st.columns(2)
with col1:
    fig_scatter = px.scatter(
        filtered_df,
        x='InvocationsPerMonth',
        y='CostUSD',
        size='MemoryMB',
        color='Environment',
        hover_name='FunctionName',
        hover_data={'MemoryMB': True, 'AvgDurationMs': True},
        title="Cost vs Invocation Frequency",
        labels={
            'InvocationsPerMonth': 'Monthly Invocations (log scale)',
            'CostUSD': 'Cost (USD)'
        },
        log_x=True
    )
    fig_scatter.update_layout(height=400)
    st.plotly_chart(fig_scatter, use_container_width=True)

with col2:
    # Top 10 cost functions table
    top_10_costs = filtered_df.nlargest(10, 'CostUSD')[
        ['FunctionName', 'Environment', 'CostUSD', 'InvocationsPerMonth', 'MemoryMB']
    ].copy()
    top_10_costs['CostUSD'] = top_10_costs['CostUSD'].apply(lambda x: f"${x:.2f}")
    top_10_costs.columns = ['Function', 'Env', 'Cost', 'Invocations', 'Memory (MB)']
    st.dataframe(top_10_costs, use_container_width=True, hide_index=True)

st.divider()

# ==================== EXERCISE 2: MEMORY RIGHT-SIZING ====================
st.markdown("### 🔧 Exercise 2: Memory Right-Sizing Opportunities")

# Find over-provisioned functions: low duration but high memory
df['MemoryEfficiency'] = df['AvgDurationMs'] / df['MemoryMB']
over_provisioned = df[(df['AvgDurationMs'] < 100) & (df['MemoryMB'] >= 1024)].copy()
df['SuggestedMemory'] = df.apply(
    lambda row: max(128, min(3072, int(row['MemoryMB'] * (row['AvgDurationMs'] / 1000) ** 0.5))) 
    if row['MemoryMB'] > 256 else row['MemoryMB'],
    axis=1
)

df['PotentialCostSavings'] = (df['MemoryMB'] - df['SuggestedMemory']) / df['MemoryMB'] * df['CostUSD']

col1, col2 = st.columns(2)
with col1:
    st.metric("Over-provisioned Functions Detected", len(over_provisioned),
              delta=f"Potential savings: ${over_provisioned['PotentialCostSavings'].sum():.2f}/month" if len(over_provisioned) > 0 else "None found")

with col2:
    total_potential_savings = df[df['PotentialCostSavings'] > 0]['PotentialCostSavings'].sum()
    st.metric("Total Potential Monthly Savings", f"${total_potential_savings:,.2f}",
              delta=f"{(total_potential_savings/df['CostUSD'].sum()*100):.1f}% of total cost")

# Memory efficiency visualization
fig_mem = px.scatter(
    df,
    x='AvgDurationMs',
    y='MemoryMB',
    color='Environment',
    size='CostUSD',
    hover_name='FunctionName',
    hover_data={'CostUSD': ':.2f', 'InvocationsPerMonth': ':,'},
    title="Memory Provisioning Analysis (Duration vs Allocated Memory)",
    labels={'AvgDurationMs': 'Avg Duration (ms)', 'MemoryMB': 'Configured Memory (MB)'}
)
fig_mem.add_hline(y=1024, line_dash="dash", line_color="red", 
                  annotation_text="High Memory Threshold", annotation_position="right")
fig_mem.add_vline(x=100, line_dash="dash", line_color="orange",
                  annotation_text="Short Duration Threshold", annotation_position="top")
fig_mem.update_layout(height=450)
st.plotly_chart(fig_mem, use_container_width=True)

# Right-sizing recommendations table
col1, col2 = st.columns([2, 1])
with col1:
    if len(df[df['PotentialCostSavings'] > 1]) > 0:
        rightsizing_candidates = df[df['PotentialCostSavings'] > 1].nlargest(8, 'PotentialCostSavings')[
            ['FunctionName', 'Environment', 'MemoryMB', 'SuggestedMemory', 'AvgDurationMs', 'PotentialCostSavings']
        ].copy()
        rightsizing_candidates.columns = ['Function', 'Env', 'Current (MB)', 'Suggested (MB)', 'Duration (ms)', 'Potential Savings']
        rightsizing_candidates['Potential Savings'] = rightsizing_candidates['Potential Savings'].apply(lambda x: f"${x:.2f}/mo")
        st.dataframe(rightsizing_candidates, use_container_width=True, hide_index=True)
    else:
        st.info("✅ No significant right-sizing opportunities found")

with col2:
    # Cost impact histogram
    if len(df[df['PotentialCostSavings'] > 0]) > 0:
        fig_hist = px.histogram(
            df[df['PotentialCostSavings'] > 0],
            x='PotentialCostSavings',
            nbins=20,
            title="Distribution of Potential Savings",
            labels={'PotentialCostSavings': 'Monthly Savings (USD)'}
        )
        fig_hist.update_layout(height=400)
        st.plotly_chart(fig_hist, use_container_width=True)

st.divider()

# ==================== EXERCISE 3: PROVISIONED CONCURRENCY OPTIMIZATION ====================
st.markdown("### ⚡ Exercise 3: Provisioned Concurrency Optimization")

pc_data = df[df['ProvisionedConcurrency'] > 0].copy()
pc_data['PCCost_Estimated'] = pc_data['ProvisionedConcurrency'] * 0.015 * 730  # $0.015/PC/hour
pc_data['ColdStartCostBenefit'] = (pc_data['ColdStartRate'] * pc_data['InvocationsPerMonth'] * 0.001)  # Estimated

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Functions with Provisioned Concurrency", len(pc_data),
              delta=f"Out of {len(df)} total")
with col2:
    total_pc_cost = pc_data['PCCost_Estimated'].sum()
    st.metric("Estimated PC Monthly Cost", f"${total_pc_cost:,.2f}")
with col3:
    cold_start_reduction = pc_data[pc_data['ColdStartRate'] > 0.02].shape[0]
    st.metric("Functions with High Cold Starts", cold_start_reduction)

# PC Optimization Analysis
if len(pc_data) > 0:
    fig_pc = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Cold Start Rate vs PC Allocation", "PC Cost Effectiveness")
    )

    pc_filtered = df[df['ProvisionedConcurrency'] > 0]
    fig_pc.add_trace(
        go.Scatter(
            x=pc_filtered['ProvisionedConcurrency'],
            y=pc_filtered['ColdStartRate_pct'],
            mode='markers+text',
            marker=dict(size=10, color=pc_filtered['CostUSD'], colorscale='Viridis', 
                       showscale=False),
            text=pc_filtered['FunctionName'].str.split('-').str[0],
            textposition='top center',
            name='Cold Start Rate',
            hovertemplate='<b>%{text}</b><br>PC Units: %{x}<br>Cold Start Rate: %{y:.2f}%<extra></extra>'
        ),
        row=1, col=1
    )

    fig_pc.add_trace(
        go.Bar(
            x=pc_filtered['FunctionName'].str.split('-').str[0:2].str.join('-'),
            y=pc_filtered['ProvisionedConcurrency'],
            name='PC Units',
            marker_color='#1f77b4'
        ),
        row=1, col=2
    )

    fig_pc.update_xaxes(title_text="Provisioned Concurrency Units", row=1, col=1)
    fig_pc.update_yaxes(title_text="Cold Start Rate (%)", row=1, col=1)
    fig_pc.update_xaxes(title_text="Function", row=1, col=2)
    fig_pc.update_yaxes(title_text="PC Units", row=1, col=2)
    fig_pc.update_layout(height=450, showlegend=True)
    st.plotly_chart(fig_pc, use_container_width=True)

    # PC Recommendations
    st.subheader("🎯 Provisioned Concurrency Recommendations")
    pc_recommendations = pc_data.nlargest(min(8, len(pc_data)), 'ProvisionedConcurrency')[
        ['FunctionName', 'Environment', 'ProvisionedConcurrency', 'ColdStartRate_pct', 'PCCost_Estimated']
    ].copy()
    pc_recommendations.columns = ['Function', 'Env', 'PC Units', 'Cold Start %', 'Est. PC Cost/mo']
    pc_recommendations['Est. PC Cost/mo'] = pc_recommendations['Est. PC Cost/mo'].apply(lambda x: f"${x:.2f}")

    # Add recommendation column
    def pc_recommendation(row):
        if row['Cold Start %'] < 1:
            return "❌ REDUCE/REMOVE - Low cold starts"
        elif row['Cold Start %'] > 3:
            return "✅ KEEP - Justified by high cold starts"
        else:
            return "⚠️ REVIEW - Monitor cold start impact"

    pc_recommendations['Recommendation'] = pc_recommendations.apply(pc_recommendation, axis=1)
    st.dataframe(pc_recommendations, use_container_width=True, hide_index=True)
else:
    st.info("ℹ️ No functions with Provisioned Concurrency found in this dataset")

st.divider()

# ==================== EXERCISE 4: UNUSED/LOW-VALUE WORKLOADS ====================
st.markdown("### 🗑️ Exercise 4: Detect Unused or Low-Value Workloads")

total_invocations = df['InvocationsPerMonth'].sum()
df['InvocationPercentage'] = (df['InvocationsPerMonth'] / total_invocations * 100)
df['CostPercentage'] = (df['CostUSD'] / df['CostUSD'].sum() * 100)

low_value = df[(df['InvocationPercentage'] < 1) & (df['CostUSD'] > 5)].copy()

col1, col2 = st.columns(2)
with col1:
    st.metric("Low-Value Functions Detected", len(low_value),
              delta=f"<1% invocations but consuming resources")
with col2:
    low_value_cost = low_value['CostUSD'].sum()
    st.metric("Combined Cost of Low-Value Functions", f"${low_value_cost:,.2f}",
              delta=f"{(low_value_cost/df['CostUSD'].sum()*100):.1f}% of total")

# Unused functions analysis
fig_unused = px.scatter(
    df,
    x='InvocationPercentage',
    y='CostPercentage',
    size='InvocationsPerMonth',
    color='Environment',
    hover_name='FunctionName',
    hover_data={'CostUSD': ':.2f', 'InvocationsPerMonth': ':,'},
    title="Cost vs Invocation Share - Identify Low-Value Workloads",
    labels={'InvocationPercentage': '% of Total Invocations', 'CostPercentage': '% of Total Cost'},
    log_x=True,
    log_y=True
)
fig_unused.add_vline(x=1, line_dash="dash", line_color="red", 
                     annotation_text="<1% threshold", annotation_position="top")
fig_unused.add_hline(y=1, line_dash="dash", line_color="red",
                     annotation_text="<1% cost threshold", annotation_position="right")
fig_unused.update_layout(height=450)
st.plotly_chart(fig_unused, use_container_width=True)

# Low-value functions table
if len(low_value) > 0:
    st.subheader("Functions to Review for Consolidation/Removal")
    low_value_table = low_value.nlargest(10, 'CostUSD')[
        ['FunctionName', 'Environment', 'InvocationsPerMonth', 'CostUSD', 'AvgDurationMs', 'MemoryMB']
    ].copy()
    low_value_table['InvocationPercentage'] = (low_value_table['InvocationsPerMonth'] / total_invocations * 100)
    low_value_table = low_value_table[['FunctionName', 'Environment', 'InvocationsPerMonth', 'InvocationPercentage', 'CostUSD']]
    low_value_table.columns = ['Function', 'Env', 'Monthly Invocations', 'Share %', 'Cost']
    low_value_table['Share %'] = low_value_table['Share %'].apply(lambda x: f"{x:.3f}%")
    low_value_table['Cost'] = low_value_table['Cost'].apply(lambda x: f"${x:.2f}")
    st.dataframe(low_value_table, use_container_width=True, hide_index=True)
else:
    st.info("✅ No low-value functions detected with current criteria")

st.divider()

# ==================== EXERCISE 5: COST FORECASTING MODEL ====================
st.markdown("### 📊 Exercise 5: Cost Forecasting Model")

st.info("📌 Forecasting Model: Cost = (Invocations × Duration × Memory × $0.0000166) + (DataTransfer × $0.02) + (PC × $0.015 × 730)")

# Build pricing model
LAMBDA_PRICE_PER_GB_SECOND = 0.0000166
TRANSFER_PRICE_PER_GB = 0.02
PC_PRICE_PER_UNIT_MONTH = 0.015 * 730

df['CalculatedCost'] = (
    (df['InvocationsPerMonth'] * df['AvgDurationMs'] / 1000 * df['MemoryMB'] / 1024 * LAMBDA_PRICE_PER_GB_SECOND) +
    (df['DataTransferGB'] * TRANSFER_PRICE_PER_GB) +
    (df['ProvisionedConcurrency'] * PC_PRICE_PER_UNIT_MONTH)
)

df['ForecastError'] = abs(df['CalculatedCost'] - df['CostUSD']) / df['CostUSD'] * 100

# Model accuracy
col1, col2, col3 = st.columns(3)
with col1:
    mae = (df['ForecastError']).mean()
    st.metric("Mean Absolute Error %", f"{mae:.1f}%")
with col2:
    rmse = np.sqrt((df['ForecastError']**2).mean())
    st.metric("Root Mean Squared Error %", f"{rmse:.1f}%")
with col3:
    r_squared = 1 - (((df['CostUSD'] - df['CalculatedCost'])**2).sum() / 
                     ((df['CostUSD'] - df['CostUSD'].mean())**2).sum())
    st.metric("R² Score", f"{r_squared:.4f}")

# Actual vs Predicted
fig_forecast = px.scatter(
    df,
    x='CostUSD',
    y='CalculatedCost',
    color='Environment',
    hover_name='FunctionName',
    title="Model Validation: Actual vs Predicted Cost",
    labels={'CostUSD': 'Actual Cost (USD)', 'CalculatedCost': 'Predicted Cost (USD)'}
)
fig_forecast.add_trace(
    go.Scatter(x=[0, df['CostUSD'].max()], y=[0, df['CostUSD'].max()],
               mode='lines', name='Perfect Prediction', 
               line=dict(color='red', dash='dash'))
)
fig_forecast.update_layout(height=450)
st.plotly_chart(fig_forecast, use_container_width=True)

# Forecasting scenarios
st.subheader("💡 Scenario Analysis: Cost Impact Forecast")
col1, col2, col3 = st.columns(3)

with col1:
    invocation_increase = st.slider("Invocation Increase (%)", 0, 100, 10, key='inv_scenario')
with col2:
    memory_reduction = st.slider("Memory Reduction (%)", 0, 50, 10, key='mem_scenario')
with col3:
    pc_reduction = st.slider("PC Reduction (%)", 0, 100, 25, key='pc_scenario')

# Calculate scenario
scenario_df = df.copy()
scenario_df['InvocationsPerMonth'] = scenario_df['InvocationsPerMonth'] * (1 + invocation_increase/100)
scenario_df['MemoryMB'] = scenario_df['MemoryMB'] * (1 - memory_reduction/100)
scenario_df['ProvisionedConcurrency'] = scenario_df['ProvisionedConcurrency'] * (1 - pc_reduction/100)

scenario_df['ScenarioCost'] = (
    (scenario_df['InvocationsPerMonth'] * scenario_df['AvgDurationMs'] / 1000 * 
     scenario_df['MemoryMB'] / 1024 * LAMBDA_PRICE_PER_GB_SECOND) +
    (scenario_df['DataTransferGB'] * TRANSFER_PRICE_PER_GB) +
    (scenario_df['ProvisionedConcurrency'] * PC_PRICE_PER_UNIT_MONTH)
)

current_total = df['CostUSD'].sum()
scenario_total = scenario_df['ScenarioCost'].sum()
delta_cost = scenario_total - current_total

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Current Total Cost", f"${current_total:,.2f}")
with col2:
    st.metric("Scenario Cost", f"${scenario_total:,.2f}", delta=f"${delta_cost:,.2f}")
with col3:
    pct_change = (delta_cost / current_total * 100)
    st.metric("% Change", f"{pct_change:.1f}%", delta=f"${abs(delta_cost):,.2f}")

st.divider()

# ==================== EXERCISE 6: CONTAINERIZATION CANDIDATES ====================
st.markdown("### 🐳 Exercise 6: Workloads Better Suited for Containerization")

# Criteria: Long-running (>3s), High memory (>2GB), Low invocation frequency (<500000/month)
containerization_candidates = df[
    (df['AvgDurationMs'] > 3000) & 
    (df['MemoryMB'] > 2048) & 
    (df['InvocationsPerMonth'] < 500000)
].copy()

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Functions Suitable for Containerization", len(containerization_candidates))
with col2:
    container_cost = containerization_candidates['CostUSD'].sum()
    st.metric("Combined Cost", f"${container_cost:,.2f}")
with col3:
    # Estimated ECS cost (typically 30-50% cheaper)
    estimated_savings = container_cost * 0.4
    st.metric("Est. Monthly Savings (40%)", f"${estimated_savings:,.2f}")

# Containerization analysis
fig_container = px.scatter(
    df,
    x='AvgDurationMs',
    y='MemoryMB',
    size='CostUSD',
    color='InvocationsPerMonth',
    hover_name='FunctionName',
    hover_data={'CostUSD': ':.2f', 'InvocationsPerMonth': ':,'},
    title="Containerization Candidates Matrix",
    labels={'AvgDurationMs': 'Avg Duration (ms)', 'MemoryMB': 'Memory (MB)', 
            'InvocationsPerMonth': 'Monthly Invocations'}
)
fig_container.add_vline(x=3000, line_dash="dash", line_color="orange",
                       annotation_text="Long-running threshold (3s)", annotation_position="top")
fig_container.add_hline(y=2048, line_dash="dash", line_color="red",
                       annotation_text="High memory threshold (2GB)", annotation_position="right")
fig_container.update_layout(height=500)
st.plotly_chart(fig_container, use_container_width=True)

# Containerization candidates table
if len(containerization_candidates) > 0:
    st.subheader("🎯 Top Containerization Candidates")
    container_table = containerization_candidates.nlargest(10, 'CostUSD')[
        ['FunctionName', 'Environment', 'AvgDurationMs', 'MemoryMB', 'InvocationsPerMonth', 'CostUSD']
    ].copy()
    container_table['Est. Savings (40%)'] = (container_table['CostUSD'] * 0.4).apply(lambda x: f"${x:.2f}")
    container_table['Duration (s)'] = (container_table['AvgDurationMs'] / 1000).apply(lambda x: f"{x:.1f}s")
    container_table['Memory'] = container_table['MemoryMB'].apply(lambda x: f"{x} MB")
    container_table = container_table[['FunctionName', 'Environment', 'Duration (s)', 'Memory', 'InvocationsPerMonth', 'CostUSD', 'Est. Savings (40%)']]
    container_table.columns = ['Function', 'Env', 'Duration', 'Memory', 'Invocations', 'Current Cost', 'Est. Savings']
    container_table['Current Cost'] = container_table['Current Cost'].apply(lambda x: f"${x:.2f}")
    st.dataframe(container_table, use_container_width=True, hide_index=True)
else:
    st.info("✅ No functions currently meet containerization criteria (long-running + high memory + low invocation)")

st.divider()

# ==================== COMPREHENSIVE RECOMMENDATIONS ====================
st.markdown("### 🚀 Executive Summary & Recommendations")

summary_col1, summary_col2 = st.columns(2)

with summary_col1:
    st.subheader("💰 Cost Optimization Opportunities")
    opportunities = {
        "Memory Right-sizing": f"${df[df['PotentialCostSavings'] > 0]['PotentialCostSavings'].sum():.2f}/mo",
        "Remove Unused Functions": f"${low_value['CostUSD'].sum():.2f}/mo",
        "Optimize Provisioned Concurrency": f"~${pc_data['PCCost_Estimated'].sum() * 0.3:.2f}/mo" if len(pc_data) > 0 else "$0/mo",
        "Containerize Long-running Tasks": f"${estimated_savings:.2f}/mo",
        "Data Transfer Optimization": f"~5-10% of egress cost"
    }
    for opportunity, saving in opportunities.items():
        st.write(f"• **{opportunity}**: {saving}")

with summary_col2:
    st.subheader("📋 Action Items")
    actions = [
        "1️⃣ Review top 10 functions accounting for 80% of costs",
        "2️⃣ Downsize functions with low duration/high memory",
        "3️⃣ Audit Provisioned Concurrency allocation",
        "4️⃣ Archive or consolidate <1% invocation functions",
        "5️⃣ Consider migrating ETL functions to ECS/Fargate",
        "6️⃣ Implement cost monitoring alerts by function",
        "7️⃣ Schedule quarterly FinOps reviews",
        "8️⃣ Enable AWS Cost Explorer tagging strategy"
    ]
    for action in actions:
        st.write(action)

# Total potential savings
total_potential = (
    df[df['PotentialCostSavings'] > 0]['PotentialCostSavings'].sum() +
    low_value['CostUSD'].sum() * 0.5 +
    estimated_savings
)

st.success(f"""
### 🎯 Total Potential Monthly Savings: **${total_potential:,.2f}**
**This represents ~{(total_potential/df['CostUSD'].sum()*100):.1f}% reduction** from current serverless spend
""")

st.divider()

# ==================== DATA EXPORT ====================
st.markdown("### 📥 Data Export")
col1, col2 = st.columns(2)

with col1:
    if st.button("📊 Export Full Analysis Dataset"):
        export_df = df[[
            'FunctionName', 'Environment', 'CostUSD', 'InvocationsPerMonth',
            'AvgDurationMs', 'MemoryMB', 'GBSeconds', 'DataTransferGB',
            'ProvisionedConcurrency', 'ColdStartRate', 'SuggestedMemory',
            'PotentialCostSavings', 'CalculatedCost'
        ]].copy()
        export_df.to_csv('finops_analysis_export.csv', index=False)
        st.success("✅ Dataset exported to finops_analysis_export.csv")

with col2:
    if st.button("📈 Export Recommendations"):
        # Combine all recommendations into one dataframe
        all_recs = []
        
        if len(df[df['PotentialCostSavings'] > 1]) > 0:
            mem_recs = df[df['PotentialCostSavings'] > 1].nlargest(8, 'PotentialCostSavings')[
                ['FunctionName', 'Environment', 'MemoryMB', 'SuggestedMemory', 'PotentialCostSavings']
            ].copy()
            mem_recs['Category'] = 'Memory Right-sizing'
            all_recs.append(mem_recs)
        
        if len(low_value) > 0:
            low_val_recs = low_value[['FunctionName', 'Environment', 'CostUSD']].copy()
            low_val_recs['Category'] = 'Low-Value Functions'
            all_recs.append(low_val_recs)
        
        if len(all_recs) > 0:
            recommendations = pd.concat(all_recs, ignore_index=True)
            recommendations.to_csv('finops_recommendations.csv', index=False)
            st.success("✅ Recommendations exported to finops_recommendations.csv")
        else:
            st.info("No recommendations to export")

st.markdown("---")
st.markdown("""
**Dashboard Created for**: RetailNova FinOps Analysis  
**Course**: INFO49971 - Cloud Economics (FALL 2025)  
**Institution**: Sheridan College
""")
