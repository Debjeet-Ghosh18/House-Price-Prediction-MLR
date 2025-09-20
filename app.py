import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import plotly.express as px
import plotly.graph_objects as go

# Set page configuration
st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-bottom: 1rem;
    }
    .metric-container {
        background-color: #f0f8ff;
        padding: 1rem;
        border-radius: 10px;
        border: 2px solid #1f77b4;
    }
    .prediction-result {
        font-size: 2rem;
        color: #2e8b57;
        font-weight: bold;
        text-align: center;
        padding: 1rem;
        background-color: #f0fff0;
        border-radius: 10px;
        border: 2px solid #2e8b57;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🏠 House Price Prediction System</h1>', unsafe_allow_html=True)

# Sample data (converted to rupees)
@st.cache_data
def load_data():
    # Converted from dollars to rupees (approx 1 USD = 83 INR)
    data = {
        'Area': [2600, 2800, 3000, 3200, 3400, 4000, 4200, 4300, 4400, 4500],
        'Bedrooms': [2, 2, 3, 3, 3, 4, 3, 3, 4, 4],
        'Age': [10, 12, 14, 15, 18, 15, 14, 16, 18, 20],
        'Price': [41500000, 45650000, 49800000, 51460000, 54780000, 58100000, 62250000, 63910000, 65570000, 67230000]
    }
    return pd.DataFrame(data)

# Train model
@st.cache_data
def train_model():
    df = load_data()
    X = df[['Area', 'Bedrooms', 'Age']]
    y = df['Price']
    
    model = LinearRegression()
    model.fit(X, y)
    return model, df

# Load data and train model
model, df = train_model()

# Sidebar for input parameters
st.sidebar.markdown('<h2 class="sub-header">🔧 Input Parameters</h2>', unsafe_allow_html=True)

# Input widgets
area = st.sidebar.slider(
    "House Area (sq ft)", 
    min_value=1000, 
    max_value=6000, 
    value=3500, 
    step=100,
    help="Select the total area of the house in square feet"
)

bedrooms = st.sidebar.selectbox(
    "Number of Bedrooms", 
    options=[1, 2, 3, 4, 5, 6],
    index=2,
    help="Select the number of bedrooms"
)

age = st.sidebar.slider(
    "Age of House (years)", 
    min_value=0, 
    max_value=50, 
    value=10, 
    step=1,
    help="Select the age of the house in years"
)

# Create two columns for main content
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown('<h2 class="sub-header">📊 Training Data</h2>', unsafe_allow_html=True)
    st.dataframe(df, use_container_width=True)
    
    # Display model coefficients
    st.markdown('<h2 class="sub-header">🔍 Model Insights</h2>', unsafe_allow_html=True)
    
    coeffs = model.coef_
    intercept = model.intercept_
    
    st.write("**Model Coefficients:**")
    st.write(f"- Area coefficient: ₹{coeffs[0]:.2f}")
    st.write(f"- Bedrooms coefficient: ₹{coeffs[1]:.2f}")
    st.write(f"- Age coefficient: ₹{coeffs[2]:.2f}")
    st.write(f"- Intercept: ₹{intercept:.2f}")
    
    st.info("""
    **Model Interpretation:**
    - Each additional sq ft increases price by ₹11,000
    - Each additional bedroom decreases price by ₹9,67,000 (holding area constant)
    - Each additional year of age increases price by ₹6,28,000
    """)

with col2:
    # Make prediction
    prediction = model.predict([[area, bedrooms, age]])[0]
    
    st.markdown('<h2 class="sub-header">🎯 Price Prediction</h2>', unsafe_allow_html=True)
    
    # Display prediction in a styled container
    st.markdown(f'''
    <div class="prediction-result">
        Predicted Price: ₹{prediction:,.0f}
    </div>
    ''', unsafe_allow_html=True)
    
    # Display input summary
    st.markdown('<div class="metric-container">', unsafe_allow_html=True)
    st.write("**Input Summary:**")
    st.write(f"🏠 Area: {area:,} sq ft")
    st.write(f"🛏️ Bedrooms: {bedrooms}")
    st.write(f"📅 Age: {age} years")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Visualizations
    st.markdown('<h2 class="sub-header">📈 Visualizations</h2>', unsafe_allow_html=True)
    
    # Price vs Area scatter plot
    fig_scatter = px.scatter(
        df, 
        x='Area', 
        y='Price', 
        size='Bedrooms', 
        color='Age',
        title='Price vs Area (Size = Bedrooms, Color = Age)',
        hover_data=['Bedrooms', 'Age']
    )
    
    # Add prediction point
    fig_scatter.add_trace(
        go.Scatter(
            x=[area],
            y=[prediction],
            mode='markers',
            marker=dict(size=15, color='red', symbol='star'),
            name='Your Prediction',
            hovertext=f'Predicted: ₹{prediction:,.0f}'
        )
    )
    
    st.plotly_chart(fig_scatter, use_container_width=True)

# Additional insights section
st.markdown('<h2 class="sub-header">📋 Additional Insights</h2>', unsafe_allow_html=True)

col3, col4, col5 = st.columns(3)

with col3:
    st.metric("Average Price", f"₹{df['Price'].mean():,.0f}")
    
with col4:
    st.metric("Price Range", f"₹{df['Price'].min():,.0f} - ₹{df['Price'].max():,.0f}")
    
with col5:
    r2_score = model.score(df[['Area', 'Bedrooms', 'Age']], df['Price'])
    st.metric("Model R² Score", f"{r2_score:.3f}")

# Feature importance visualization
st.markdown('<h2 class="sub-header">🎯 Feature Importance</h2>', unsafe_allow_html=True)

# Calculate feature importance based on absolute coefficients
feature_names = ['Area', 'Bedrooms', 'Age']
abs_coeffs = np.abs(model.coef_)
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': abs_coeffs / abs_coeffs.sum() * 100
})

fig_importance = px.bar(
    importance_df,
    x='Feature',
    y='Importance',
    title='Feature Importance (% of total absolute coefficient weight)',
    color='Importance',
    color_continuous_scale='viridis'
)
st.plotly_chart(fig_importance, use_container_width=True)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        <p>🏠 House Price Prediction Model using Multiple Linear Regression</p>
        <p>Built with Streamlit | Data Science Project</p>
    </div>
    """, 
    unsafe_allow_html=True
)