import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

# Page configuration
st.set_page_config(
    page_title="Gradient Descent Visualizer",
    page_icon="📉",
    layout="wide"
)

# Custom CSS for better UI
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
        color: #1f77b4;
        font-weight: bold;
    }
    h2 {
        color: #ff7f0e;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Title and description
st.title("📉 Gradient Descent Optimizer")
st.markdown("**Visualize and understand gradient descent with interactive controls**")

# Sidebar for configuration
st.sidebar.header("⚙️ Configuration")

# Function selection
function_type = st.sidebar.selectbox(
    "Select Function",
    ["Quadratic", "Linear Regression", "Cubic", "Sin Wave"]
)

# Parameters
learning_rate = st.sidebar.slider(
    "Learning Rate (α)",
    min_value=0.001,
    max_value=1.0,
    value=0.1,
    step=0.001,
    help="Controls how big each step in gradient descent is"
)

iterations = st.sidebar.slider(
    "Number of Iterations",
    min_value=10,
    max_value=500,
    value=100,
    step=10
)

start_x = st.sidebar.slider(
    "Starting Point (x₀)",
    min_value=-10.0,
    max_value=10.0,
    value=8.0,
    step=0.5
)

# Gradient Descent Functions
def quadratic_function(x):
    """f(x) = x^2"""
    return x**2

def quadratic_derivative(x):
    """f'(x) = 2x"""
    return 2*x

def cubic_function(x):
    """f(x) = x^3 - 3x^2 + 2x"""
    return x**3 - 3*x**2 + 2*x

def cubic_derivative(x):
    """f'(x) = 3x^2 - 6x + 2"""
    return 3*x**2 - 6*x + 2

def sin_function(x):
    """f(x) = sin(x) + x^2/10"""
    return np.sin(x) + x**2/10

def sin_derivative(x):
    """f'(x) = cos(x) + x/5"""
    return np.cos(x) + x/5

# Select function and derivative based on user choice
if function_type == "Quadratic":
    func = quadratic_function
    func_derivative = quadratic_derivative
elif function_type == "Cubic":
    func = cubic_function
    func_derivative = cubic_derivative
elif function_type == "Sin Wave":
    func = sin_function
    func_derivative = sin_derivative
else:
    func = quadratic_function
    func_derivative = quadratic_derivative

# Gradient Descent Algorithm
def gradient_descent(start, learning_rate, iterations, func, func_derivative):
    """
    Perform gradient descent optimization
    
    Parameters:
    - start: starting point
    - learning_rate: step size
    - iterations: number of iterations
    - func: function to minimize
    - func_derivative: derivative of the function
    
    Returns:
    - history: list of (x, y) points during optimization
    - gradients: list of gradients at each step
    """
    x = start
    history = [(x, func(x))]
    gradients = []
    
    for i in range(iterations):
        gradient = func_derivative(x)
        gradients.append(gradient)
        x = x - learning_rate * gradient
        history.append((x, func(x)))
    
    return history, gradients

# Linear Regression with Gradient Descent
def linear_regression_gd():
    st.subheader("📊 Linear Regression with Gradient Descent")
    
    # Data source selection
    st.markdown("### 📂 Select Data Source")
    data_source = st.radio(
        "Choose your data source:",
        ["Synthetic Data", "Iris Dataset", "Upload CSV File"],
        horizontal=True
    )
    
    X = None
    y = None
    true_m = None
    true_b = None
    feature_name = "X"
    target_name = "Y"
    
    if data_source == "Synthetic Data":
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Data Generation")
            n_samples = st.slider("Number of samples", 10, 200, 50)
            noise_level = st.slider("Noise level", 0.0, 10.0, 2.0)
            
        # Generate synthetic data
        np.random.seed(42)
        X = np.linspace(0, 10, n_samples)
        true_m = 2.5
        true_b = 1.0
        y = true_m * X + true_b + np.random.randn(n_samples) * noise_level
        
    elif data_source == "Iris Dataset":
        st.markdown("### 🌸 Iris Dataset")
        st.info("Using the famous Iris dataset for regression analysis")
        
        # Load iris dataset
        iris = load_iris()
        iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
        iris_df['target'] = iris.target
        
        col1, col2 = st.columns(2)
        with col1:
            feature_options = iris.feature_names
            selected_feature = st.selectbox(
                "Select Feature (X):",
                feature_options,
                index=0
            )
            feature_name = selected_feature
            
        with col2:
            target_options = iris.feature_names + ['target']
            selected_target = st.selectbox(
                "Select Target (Y):",
                target_options,
                index=1
            )
            target_name = selected_target
        
        # Extract selected columns
        X = iris_df[selected_feature].values
        y = iris_df[selected_target].values
        
        # Display dataset info
        st.markdown(f"**Dataset shape:** {len(X)} samples")
        with st.expander("📊 View Dataset Preview"):
            st.dataframe(iris_df.head(10))
            
    elif data_source == "Upload CSV File":
        st.markdown("### 📤 Upload Your CSV File")
        
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=["csv"],
            help="Upload a CSV file with numerical data"
        )
        
        if uploaded_file is not None:
            try:
                # Read CSV file
                df = pd.read_csv(uploaded_file)
                
                st.success(f"File uploaded successfully! Shape: {df.shape}")
                
                # Display data preview
                with st.expander("📊 View Data Preview"):
                    st.dataframe(df.head(10))
                
                # Let user select columns
                numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
                
                if len(numeric_columns) < 2:
                    st.error("CSV file must contain at least 2 numerical columns!")
                else:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        selected_feature = st.selectbox(
                            "Select Feature (X):",
                            numeric_columns,
                            index=0
                        )
                        feature_name = selected_feature
                        
                    with col2:
                        selected_target = st.selectbox(
                            "Select Target (Y):",
                            numeric_columns,
                            index=min(1, len(numeric_columns)-1)
                        )
                        target_name = selected_target
                    
                    # Extract data
                    X = df[selected_feature].values
                    y = df[selected_target].values
                    
                    # Remove NaN values
                    mask = ~(np.isnan(X) | np.isnan(y))
                    X = X[mask]
                    y = y[mask]
                    
                    st.markdown(f"**Valid samples:** {len(X)}")
                    
            except Exception as e:
                st.error(f"Error reading CSV file: {str(e)}")
        else:
            st.warning("Please upload a CSV file to continue")
            return
    
    # Check if data is loaded
    if X is None or y is None:
        st.warning("Please select a valid data source with proper configuration")
        return
    
    # Initialize parameters
    m = 0.0
    b = 0.0
    
    # Store history
    m_history = [m]
    b_history = [b]
    cost_history = []
    
    # Gradient descent for linear regression
    lr_rate = learning_rate / 10  # Scale down for linear regression
    
    for _ in range(iterations):
        # Predictions
        y_pred = m * X + b
        
        # Calculate cost (MSE)
        cost = np.mean((y - y_pred)**2)
        cost_history.append(cost)
        
        # Calculate gradients
        dm = -2 * np.mean(X * (y - y_pred))
        db = -2 * np.mean(y - y_pred)
        
        # Update parameters
        m = m - lr_rate * dm
        b = b - lr_rate * db
        
        m_history.append(m)
        b_history.append(b)
    
    # Display results
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Final Slope (m)", f"{m:.4f}", f"True: {true_m:.4f}" if true_m is not None else None)
    with col2:
        st.metric("Final Intercept (b)", f"{b:.4f}", f"True: {true_b:.4f}" if true_b is not None else None)
    with col3:
        st.metric("Final Cost", f"{cost_history[-1]:.4f}")
    
    # Create visualizations
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Data & Fitted Line', 'Cost vs Iterations', 
                       'Slope Convergence', 'Intercept Convergence'),
        specs=[[{"type": "scatter"}, {"type": "scatter"}],
               [{"type": "scatter"}, {"type": "scatter"}]]
    )
    
    # Plot 1: Data and fitted line
    fig.add_trace(
        go.Scatter(x=X, y=y, mode='markers', name='Data', 
                  marker=dict(color='lightblue', size=8)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=X, y=m*X+b, mode='lines', name='Fitted Line',
                  line=dict(color='red', width=3)),
        row=1, col=1
    )
    
    # Add true line only for synthetic data
    if true_m is not None and true_b is not None:
        fig.add_trace(
            go.Scatter(x=X, y=true_m*X+true_b, mode='lines', name='True Line',
                      line=dict(color='green', width=2, dash='dash')),
            row=1, col=1
        )
    
    # Plot 2: Cost history
    fig.add_trace(
        go.Scatter(y=cost_history, mode='lines', name='Cost',
                  line=dict(color='green', width=2)),
        row=1, col=2
    )
    
    # Plot 3: Slope convergence
    fig.add_trace(
        go.Scatter(y=m_history, mode='lines', name='Slope (m)',
                  line=dict(color='purple', width=2)),
        row=2, col=1
    )
    if true_m is not None:
        fig.add_hline(y=true_m, line_dash="dash", line_color="red", 
                      annotation_text="True m", row=2, col=1)
    
    # Plot 4: Intercept convergence
    fig.add_trace(
        go.Scatter(y=b_history, mode='lines', name='Intercept (b)',
                  line=dict(color='orange', width=2)),
        row=2, col=2
    )
    if true_b is not None:
        fig.add_hline(y=true_b, line_dash="dash", line_color="red",
                      annotation_text="True b", row=2, col=2)
    
    fig.update_layout(height=700, showlegend=True)
    fig.update_xaxes(title_text=feature_name, row=1, col=1)
    fig.update_yaxes(title_text=target_name, row=1, col=1)
    fig.update_xaxes(title_text="Iteration", row=1, col=2)
    fig.update_yaxes(title_text="Cost (MSE)", row=1, col=2)
    fig.update_xaxes(title_text="Iteration", row=2, col=1)
    fig.update_yaxes(title_text="Slope (m)", row=2, col=1)
    fig.update_xaxes(title_text="Iteration", row=2, col=2)
    fig.update_yaxes(title_text="Intercept (b)", row=2, col=2)
    
    st.plotly_chart(fig, use_container_width=True)

# Main content area
tab1, tab2, tab3 = st.tabs(["📈 Function Optimization", "📊 Linear Regression", "📚 Learn More"])

with tab1:
    st.subheader(f"Optimizing {function_type} Function")
    
    # Run gradient descent
    history, gradients = gradient_descent(start_x, learning_rate, iterations, func, func_derivative)
    
    # Extract x and y values
    x_values = [point[0] for point in history]
    y_values = [point[1] for point in history]
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Starting Point", f"{start_x:.2f}")
    with col2:
        st.metric("Final Point", f"{x_values[-1]:.4f}")
    with col3:
        st.metric("Starting Cost", f"{y_values[0]:.4f}")
    with col4:
        st.metric("Final Cost", f"{y_values[-1]:.4f}")
    
    # Create visualization
    x_range = np.linspace(-10, 10, 400)
    y_range = func(x_range)
    
    # Create subplots
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Optimization Path', 'Cost vs Iterations'),
        column_widths=[0.6, 0.4]
    )
    
    # Plot function
    fig.add_trace(
        go.Scatter(x=x_range, y=y_range, mode='lines', name='Function',
                  line=dict(color='blue', width=2)),
        row=1, col=1
    )
    
    # Plot gradient descent path
    fig.add_trace(
        go.Scatter(x=x_values, y=y_values, mode='markers+lines',
                  name='GD Path',
                  marker=dict(size=8, color=list(range(len(x_values))),
                            colorscale='Reds', showscale=True,
                            colorbar=dict(title="Iteration")),
                  line=dict(color='red', width=1, dash='dot')),
        row=1, col=1
    )
    
    # Highlight start and end points
    fig.add_trace(
        go.Scatter(x=[x_values[0]], y=[y_values[0]], mode='markers',
                  name='Start', marker=dict(size=15, color='green',
                  symbol='star')),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=[x_values[-1]], y=[y_values[-1]], mode='markers',
                  name='End', marker=dict(size=15, color='red',
                  symbol='star')),
        row=1, col=1
    )
    
    # Plot cost over iterations
    fig.add_trace(
        go.Scatter(y=y_values, mode='lines', name='Cost',
                  line=dict(color='purple', width=2)),
        row=1, col=2
    )
    
    fig.update_xaxes(title_text="x", row=1, col=1)
    fig.update_yaxes(title_text="f(x)", row=1, col=1)
    fig.update_xaxes(title_text="Iteration", row=1, col=2)
    fig.update_yaxes(title_text="Cost", row=1, col=2)
    
    fig.update_layout(height=500, showlegend=True)
    st.plotly_chart(fig, use_container_width=True)
    
    # Show iteration details
    with st.expander("📋 View Iteration Details"):
        # Create DataFrame with iteration details
        df = pd.DataFrame({
            'Iteration': list(range(len(x_values))),
            'x': x_values,
            'f(x)': y_values,
            'Gradient': [None] + gradients
        })
        st.dataframe(df.round(6), use_container_width=True)

with tab2:
    linear_regression_gd()

with tab3:
    st.subheader("📚 What is Gradient Descent?")
    
    st.markdown("""
    ### Overview
    Gradient Descent is an optimization algorithm used to minimize a cost function by iteratively 
    moving in the direction of steepest descent.
    
    ### The Algorithm
    The basic update rule is:
    """)
    
    st.latex(r"x_{new} = x_{old} - \alpha \cdot \frac{\partial f}{\partial x}")
    
    st.markdown("""
    Where:
    - **x** is the parameter we're optimizing
    - **α** (alpha) is the learning rate
    - **∂f/∂x** is the gradient (derivative) of the function
    
    ### Key Concepts
    
    #### 1. Learning Rate (α)
    - **Too small**: Convergence is slow
    - **Too large**: May overshoot the minimum or diverge
    - **Just right**: Efficient convergence
    
    #### 2. Gradient
    - Points in the direction of steepest ascent
    - We move in the opposite direction (negative gradient) to minimize
    
    #### 3. Convergence
    - The algorithm stops when:
      - Maximum iterations reached
      - Gradient becomes very small
      - Cost change becomes negligible
    
    ### Applications
    - **Machine Learning**: Training neural networks, linear regression
    - **Deep Learning**: Backpropagation
    - **Optimization**: Finding minimum/maximum of functions
    - **Computer Vision**: Image processing
    
    ### Types of Gradient Descent
    1. **Batch Gradient Descent**: Uses entire dataset
    2. **Stochastic Gradient Descent (SGD)**: Uses one sample at a time
    3. **Mini-batch Gradient Descent**: Uses small batches of data
    
    ### Tips for Using This App
    - Start with a **moderate learning rate** (0.01 - 0.1)
    - Observe how different **starting points** affect convergence
    - Try different **functions** to see various optimization landscapes
    - Watch for **oscillations** (learning rate too high) or **slow progress** (learning rate too low)
    """)

# Footer
st.sidebar.markdown("---")
st.sidebar.info("""
    **Created with ❤️ using Streamlit**
    
    Adjust the parameters to see how gradient descent works in real-time!
""")
