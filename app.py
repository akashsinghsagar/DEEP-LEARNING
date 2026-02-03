import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Deep Learning Hub",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stAlert {
        padding: 1rem;
        border-radius: 0.5rem;
    }
    h1 {
        color: #2E86AB;
        font-weight: bold;
        text-align: center;
    }
    .big-font {
        font-size: 20px !important;
        font-weight: bold;
    }
    .card {
        padding: 2rem;
        border-radius: 1rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        margin: 1rem 0;
    }
    .feature-box {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border-left: 5px solid #2E86AB;
    }
    </style>
    """, unsafe_allow_html=True)

# Header
st.title("🧠 Deep Learning & Gradient Descent Hub")
st.markdown("### *Interactive Learning Platform for Machine Learning Optimization*")
st.markdown("---")

# Welcome section
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown("""
    <div class="card">
        <h2 style="text-align: center; margin: 0;">Welcome to Deep Learning Hub!</h2>
        <p style="text-align: center; margin-top: 1rem;">
        Explore gradient descent algorithms and neural networks through interactive visualizations.
        </p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("##")

# Two main sections
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="feature-box">
        <h3>📉 Gradient Descent Visualizer</h3>
        <p><b>Learn the fundamentals of optimization</b></p>
        <ul>
            <li>Visualize gradient descent on mathematical functions</li>
            <li>Interactive parameter controls</li>
            <li>Linear regression implementation</li>
            <li>Real-time convergence tracking</li>
            <li>Multiple function types</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("🚀 Open Gradient Descent", use_container_width=True, type="primary"):
        st.switch_page("pages/1_📉_Gradient_Descent.py")

with col2:
    st.markdown("""
    <div class="feature-box">
        <h3>🧠 Neural Network Trainer</h3>
        <p><b>Build and train deep learning models</b></p>
        <ul>
            <li>Custom neural network architectures</li>
            <li>Multiple optimizers (SGD, Adam, RMSprop, Momentum)</li>
            <li>Various activation functions</li>
            <li>Decision boundary visualization</li>
            <li>Real-time training metrics</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("🚀 Open Neural Network", use_container_width=True, type="primary"):
        st.switch_page("pages/2_🧠_Neural_Network.py")

st.markdown("---")

# Features section
st.markdown("## 🌟 Key Features")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    #### 🎯 Interactive Learning
    - Real-time parameter adjustment
    - Instant visualization updates
    - Intuitive user interface
    - Educational tooltips
    """)

with col2:
    st.markdown("""
    #### 📊 Comprehensive Analytics
    - Loss and accuracy tracking
    - Weight distribution analysis
    - Gradient flow monitoring
    - Performance metrics
    """)

with col3:
    st.markdown("""
    #### 🔧 Flexible Configuration
    - Multiple datasets
    - Various optimizers
    - Custom architectures
    - Adjustable hyperparameters
    """)

st.markdown("---")

# Quick start guide
with st.expander("📚 Quick Start Guide"):
    st.markdown("""
    ### Getting Started
    
    #### For Gradient Descent Visualizer:
    1. Navigate to **Gradient Descent** page from the sidebar
    2. Select a function type (Quadratic, Cubic, etc.)
    3. Adjust learning rate and iterations
    4. Set starting point
    5. Watch the optimization in action!
    
    #### For Neural Network Trainer:
    1. Navigate to **Neural Network** page from the sidebar
    2. Choose a dataset
    3. Configure network architecture
    4. Select optimizer and activation function
    5. Set hyperparameters
    6. Click "Train Network" and observe results!
    
    ### Tips for Best Results:
    - Start with **moderate learning rates** (0.01 - 0.1)
    - Use **Adam optimizer** for quick convergence
    - Try **ReLU activation** for deep networks
    - Experiment with **different architectures**
    """)

# About section
with st.expander("ℹ️ About This Project"):
    st.markdown("""
    ### About
    
    This interactive platform is designed to help students, educators, and ML practitioners 
    understand optimization algorithms and neural networks through hands-on experimentation.
    
    **Technologies Used:**
    - Python 3.10
    - Streamlit (Web Framework)
    - NumPy (Numerical Computing)
    - Plotly (Interactive Visualizations)
    - Scikit-learn (Dataset Generation)
    
    **Author:** Akash Singh Sagar
    
    **GitHub:** [DEEP-LEARNING Repository](https://github.com/akashsinghsagar/DEEP-LEARNING)
    
    ### License
    Open source - Feel free to use and modify for educational purposes!
    """)

# Footer
st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: #666; padding: 2rem;">
        <p>Made with ❤️ using Streamlit | © 2026 Deep Learning Hub</p>
        <p>Navigate using the sidebar to explore different modules ➡️</p>
    </div>
    """, unsafe_allow_html=True)
