import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time

# Page configuration
st.set_page_config(
    page_title="Deep Learning Gradient Descent",
    page_icon="🧠",
    layout="wide"
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
        color: #e74c3c;
        font-weight: bold;
    }
    h2 {
        color: #3498db;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        background-color: #f0f2f6;
        border-radius: 5px;
    }
    </style>
    """, unsafe_allow_html=True)

# Title
st.title("🧠 Deep Learning Gradient Descent")
st.markdown("**Train Neural Networks with Different Optimizers**")

# Activation Functions
class Activations:
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    @staticmethod
    def sigmoid_derivative(x):
        return x * (1 - x)
    
    @staticmethod
    def relu(x):
        return np.maximum(0, x)
    
    @staticmethod
    def relu_derivative(x):
        return (x > 0).astype(float)
    
    @staticmethod
    def tanh(x):
        return np.tanh(x)
    
    @staticmethod
    def tanh_derivative(x):
        return 1 - x**2
    
    @staticmethod
    def softmax(x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# Optimizers
class SGD:
    """Stochastic Gradient Descent"""
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
    
    def update(self, param, grad):
        return param - self.learning_rate * grad

class SGDMomentum:
    """SGD with Momentum"""
    def __init__(self, learning_rate=0.01, momentum=0.9):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocity = {}
    
    def update(self, param, grad, param_name):
        if param_name not in self.velocity:
            self.velocity[param_name] = np.zeros_like(param)
        
        self.velocity[param_name] = self.momentum * self.velocity[param_name] - self.learning_rate * grad
        return param + self.velocity[param_name]

class RMSprop:
    """RMSprop Optimizer"""
    def __init__(self, learning_rate=0.001, decay_rate=0.9, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.epsilon = epsilon
        self.cache = {}
    
    def update(self, param, grad, param_name):
        if param_name not in self.cache:
            self.cache[param_name] = np.zeros_like(param)
        
        self.cache[param_name] = self.decay_rate * self.cache[param_name] + (1 - self.decay_rate) * grad**2
        param_update = param - self.learning_rate * grad / (np.sqrt(self.cache[param_name]) + self.epsilon)
        return param_update

class Adam:
    """Adam Optimizer"""
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {}
        self.v = {}
        self.t = 0
    
    def update(self, param, grad, param_name):
        if param_name not in self.m:
            self.m[param_name] = np.zeros_like(param)
            self.v[param_name] = np.zeros_like(param)
        
        self.t += 1
        self.m[param_name] = self.beta1 * self.m[param_name] + (1 - self.beta1) * grad
        self.v[param_name] = self.beta2 * self.v[param_name] + (1 - self.beta2) * (grad**2)
        
        m_hat = self.m[param_name] / (1 - self.beta1**self.t)
        v_hat = self.v[param_name] / (1 - self.beta2**self.t)
        
        param_update = param - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
        return param_update

# Neural Network Class
class NeuralNetwork:
    def __init__(self, layers, activation='relu', optimizer='adam', learning_rate=0.01):
        """
        Initialize neural network
        layers: list of layer sizes [input_size, hidden1, hidden2, ..., output_size]
        """
        self.layers = layers
        self.num_layers = len(layers)
        self.activation_name = activation
        self.weights = []
        self.biases = []
        
        # Initialize weights and biases
        for i in range(len(layers) - 1):
            # He initialization for ReLU, Xavier for others
            if activation == 'relu':
                w = np.random.randn(layers[i], layers[i+1]) * np.sqrt(2.0 / layers[i])
            else:
                w = np.random.randn(layers[i], layers[i+1]) * np.sqrt(1.0 / layers[i])
            b = np.zeros((1, layers[i+1]))
            self.weights.append(w)
            self.biases.append(b)
        
        # Set activation function
        if activation == 'sigmoid':
            self.activation = Activations.sigmoid
            self.activation_derivative = Activations.sigmoid_derivative
        elif activation == 'relu':
            self.activation = Activations.relu
            self.activation_derivative = Activations.relu_derivative
        elif activation == 'tanh':
            self.activation = Activations.tanh
            self.activation_derivative = Activations.tanh_derivative
        
        # Set optimizer
        if optimizer == 'sgd':
            self.optimizer = SGD(learning_rate)
        elif optimizer == 'momentum':
            self.optimizer = SGDMomentum(learning_rate)
        elif optimizer == 'rmsprop':
            self.optimizer = RMSprop(learning_rate)
        elif optimizer == 'adam':
            self.optimizer = Adam(learning_rate)
    
    def forward(self, X):
        """Forward propagation"""
        self.layer_inputs = [X]
        self.layer_outputs = [X]
        
        for i in range(self.num_layers - 1):
            z = np.dot(self.layer_outputs[-1], self.weights[i]) + self.biases[i]
            self.layer_inputs.append(z)
            
            # Use softmax for output layer in classification
            if i == self.num_layers - 2 and self.layers[-1] > 1:
                a = Activations.softmax(z)
            else:
                a = self.activation(z)
            
            self.layer_outputs.append(a)
        
        return self.layer_outputs[-1]
    
    def backward(self, X, y, output):
        """Backward propagation"""
        m = X.shape[0]
        
        # Calculate output layer error
        if self.layers[-1] > 1:  # Multi-class
            delta = output - y
        else:  # Binary or regression
            delta = (output - y) * self.activation_derivative(output)
        
        # Store gradients
        weight_gradients = []
        bias_gradients = []
        
        # Backpropagate through layers
        for i in range(self.num_layers - 2, -1, -1):
            dW = np.dot(self.layer_outputs[i].T, delta) / m
            db = np.sum(delta, axis=0, keepdims=True) / m
            
            weight_gradients.insert(0, dW)
            bias_gradients.insert(0, db)
            
            if i > 0:
                delta = np.dot(delta, self.weights[i].T) * self.activation_derivative(self.layer_outputs[i])
        
        # Update weights using optimizer
        for i in range(len(self.weights)):
            if hasattr(self.optimizer, 'm'):  # Adam or Momentum
                self.weights[i] = self.optimizer.update(self.weights[i], weight_gradients[i], f'W{i}')
                self.biases[i] = self.optimizer.update(self.biases[i], bias_gradients[i], f'b{i}')
            else:  # SGD or RMSprop
                self.weights[i] = self.optimizer.update(self.weights[i], weight_gradients[i], f'W{i}')
                self.biases[i] = self.optimizer.update(self.biases[i], bias_gradients[i], f'b{i}')
        
        return weight_gradients, bias_gradients
    
    def train(self, X, y, epochs, batch_size=32):
        """Train the network"""
        history = {
            'loss': [],
            'accuracy': [],
            'weight_norms': [],
            'gradient_norms': []
        }
        
        n_samples = X.shape[0]
        
        for epoch in range(epochs):
            # Mini-batch training
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            epoch_loss = 0
            epoch_gradients = []
            
            for i in range(0, n_samples, batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                
                # Forward pass
                output = self.forward(X_batch)
                
                # Calculate loss
                if self.layers[-1] > 1:  # Multi-class
                    loss = -np.mean(y_batch * np.log(output + 1e-8))
                else:  # Binary or regression
                    loss = np.mean((output - y_batch)**2)
                
                epoch_loss += loss
                
                # Backward pass
                weight_grads, bias_grads = self.backward(X_batch, y_batch, output)
                epoch_gradients.extend([np.linalg.norm(g) for g in weight_grads])
            
            # Calculate metrics
            output = self.forward(X)
            
            if self.layers[-1] > 1:  # Multi-class
                predictions = np.argmax(output, axis=1)
                true_labels = np.argmax(y, axis=1)
                accuracy = np.mean(predictions == true_labels)
            else:
                predictions = (output > 0.5).astype(int)
                accuracy = np.mean(predictions == y)
            
            # Store history
            history['loss'].append(epoch_loss / (n_samples // batch_size))
            history['accuracy'].append(accuracy)
            history['weight_norms'].append(np.mean([np.linalg.norm(w) for w in self.weights]))
            history['gradient_norms'].append(np.mean(epoch_gradients))
        
        return history

# Sidebar configuration
st.sidebar.header("⚙️ Network Configuration")

# Dataset selection
dataset_type = st.sidebar.selectbox(
    "Select Dataset",
    ["Binary Classification (Moons)", "Multi-class Classification (Circles)", 
     "XOR Problem", "Regression (Non-linear)"]
)

# Network architecture
st.sidebar.subheader("🏗️ Architecture")
hidden_layers = st.sidebar.slider("Number of Hidden Layers", 1, 4, 2)
hidden_sizes = []
for i in range(hidden_layers):
    size = st.sidebar.slider(f"Hidden Layer {i+1} Size", 4, 128, 32, key=f"hidden_{i}")
    hidden_sizes.append(size)

# Activation function
activation = st.sidebar.selectbox(
    "Activation Function",
    ["relu", "sigmoid", "tanh"]
)

# Optimizer
optimizer = st.sidebar.selectbox(
    "Optimizer",
    ["adam", "sgd", "momentum", "rmsprop"]
)

# Training parameters
st.sidebar.subheader("📊 Training Parameters")
learning_rate = st.sidebar.slider(
    "Learning Rate",
    min_value=0.0001,
    max_value=0.1,
    value=0.01,
    step=0.0001,
    format="%.4f"
)

epochs = st.sidebar.slider("Epochs", 10, 500, 100, step=10)
batch_size = st.sidebar.slider("Batch Size", 8, 128, 32, step=8)

# Generate dataset
def generate_dataset(dataset_type, n_samples=1000):
    np.random.seed(42)
    
    if dataset_type == "Binary Classification (Moons)":
        from sklearn.datasets import make_moons
        X, y = make_moons(n_samples=n_samples, noise=0.1)
        y = y.reshape(-1, 1)
        return X, y, 2
    
    elif dataset_type == "Multi-class Classification (Circles)":
        from sklearn.datasets import make_circles
        X, y_temp = make_circles(n_samples=n_samples, noise=0.1, factor=0.5)
        # Add third class
        theta = np.random.rand(n_samples // 3) * 2 * np.pi
        r = 1.5 + np.random.randn(n_samples // 3) * 0.1
        X_outer = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
        X = np.vstack([X, X_outer])
        y_temp = np.hstack([y_temp, np.ones(n_samples // 3) * 2])
        
        # One-hot encode
        y = np.zeros((len(y_temp), 3))
        y[np.arange(len(y_temp)), y_temp.astype(int)] = 1
        return X, y, 3
    
    elif dataset_type == "XOR Problem":
        X = np.random.randn(n_samples, 2)
        y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(int).reshape(-1, 1)
        return X, y, 2
    
    else:  # Regression
        X = np.random.rand(n_samples, 2) * 4 - 2
        y = (np.sin(X[:, 0]) * np.cos(X[:, 1]) + np.random.randn(n_samples) * 0.1).reshape(-1, 1)
        return X, y, 1

# Main content
tab1, tab2, tab3 = st.tabs(["🎯 Training", "📊 Analysis", "📚 Theory"])

with tab1:
    st.subheader("Neural Network Training")
    
    # Generate data
    X, y, output_size = generate_dataset(dataset_type)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Samples", X.shape[0])
    with col2:
        st.metric("Input Features", X.shape[1])
    with col3:
        st.metric("Output Classes", output_size if output_size > 2 else "Binary")
    
    # Build architecture
    architecture = [X.shape[1]] + hidden_sizes + [output_size]
    
    # Display architecture
    st.info(f"**Network Architecture:** {' → '.join(map(str, architecture))}")
    
    # Train button
    if st.button("🚀 Train Network", type="primary"):
        with st.spinner("Training neural network..."):
            # Create network
            nn = NeuralNetwork(architecture, activation, optimizer, learning_rate)
            
            # Train
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            history = nn.train(X, y, epochs, batch_size)
            
            progress_bar.progress(100)
            status_text.success(f"✅ Training complete! Final Loss: {history['loss'][-1]:.4f}")
            
            # Store in session state
            st.session_state['history'] = history
            st.session_state['nn'] = nn
            st.session_state['X'] = X
            st.session_state['y'] = y
            st.session_state['architecture'] = architecture
            
            # Display results
            st.subheader("📈 Training Results")
            
            # Create visualization
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Loss Curve', 'Accuracy', 'Weight Norms', 'Gradient Norms')
            )
            
            # Loss curve
            fig.add_trace(
                go.Scatter(y=history['loss'], mode='lines', name='Loss',
                          line=dict(color='red', width=2)),
                row=1, col=1
            )
            
            # Accuracy
            fig.add_trace(
                go.Scatter(y=history['accuracy'], mode='lines', name='Accuracy',
                          line=dict(color='green', width=2)),
                row=1, col=2
            )
            
            # Weight norms
            fig.add_trace(
                go.Scatter(y=history['weight_norms'], mode='lines', name='Weight Norm',
                          line=dict(color='blue', width=2)),
                row=2, col=1
            )
            
            # Gradient norms
            fig.add_trace(
                go.Scatter(y=history['gradient_norms'], mode='lines', name='Gradient Norm',
                          line=dict(color='purple', width=2)),
                row=2, col=2
            )
            
            fig.update_xaxes(title_text="Epoch", row=1, col=1)
            fig.update_xaxes(title_text="Epoch", row=1, col=2)
            fig.update_xaxes(title_text="Epoch", row=2, col=1)
            fig.update_xaxes(title_text="Epoch", row=2, col=2)
            
            fig.update_layout(height=600, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            
            # Decision boundary visualization
            if X.shape[1] == 2:
                st.subheader("🎨 Decision Boundary")
                
                # Create mesh
                h = 0.02
                x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
                y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
                xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                                     np.arange(y_min, y_max, h))
                
                # Predict on mesh
                Z = nn.forward(np.c_[xx.ravel(), yy.ravel()])
                if output_size > 1:
                    Z = np.argmax(Z, axis=1)
                else:
                    Z = (Z > 0.5).astype(int).ravel()
                Z = Z.reshape(xx.shape)
                
                fig = go.Figure()
                
                # Plot decision boundary
                fig.add_trace(go.Contour(
                    x=np.arange(x_min, x_max, h),
                    y=np.arange(y_min, y_max, h),
                    z=Z,
                    colorscale='RdBu',
                    opacity=0.3,
                    showscale=False
                ))
                
                # Plot data points
                if output_size > 1:
                    labels = np.argmax(y, axis=1)
                else:
                    labels = y.ravel()
                
                for label in np.unique(labels):
                    mask = labels == label
                    fig.add_trace(go.Scatter(
                        x=X[mask, 0],
                        y=X[mask, 1],
                        mode='markers',
                        name=f'Class {int(label)}',
                        marker=dict(size=8, line=dict(width=1, color='white'))
                    ))
                
                fig.update_layout(
                    title="Decision Boundary",
                    xaxis_title="Feature 1",
                    yaxis_title="Feature 2",
                    height=500
                )
                st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("📊 Detailed Analysis")
    
    if 'history' in st.session_state:
        history = st.session_state['history']
        nn = st.session_state['nn']
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Final Loss", f"{history['loss'][-1]:.4f}")
        with col2:
            st.metric("Final Accuracy", f"{history['accuracy'][-1]:.2%}")
        with col3:
            st.metric("Best Accuracy", f"{max(history['accuracy']):.2%}")
        with col4:
            improvement = (history['loss'][0] - history['loss'][-1]) / history['loss'][0] * 100
            st.metric("Loss Reduction", f"{improvement:.1f}%")
        
        # Weight distribution
        st.subheader("⚖️ Weight Distribution")
        
        fig = make_subplots(
            rows=1, cols=len(nn.weights),
            subplot_titles=[f'Layer {i+1}' for i in range(len(nn.weights))]
        )
        
        for i, w in enumerate(nn.weights):
            fig.add_trace(
                go.Histogram(x=w.flatten(), nbinsx=50, name=f'Layer {i+1}'),
                row=1, col=i+1
            )
        
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
        # Training metrics table
        st.subheader("📋 Training Metrics")
        metrics_df = pd.DataFrame({
            'Epoch': list(range(len(history['loss']))),
            'Loss': history['loss'],
            'Accuracy': history['accuracy'],
            'Weight Norm': history['weight_norms'],
            'Gradient Norm': history['gradient_norms']
        })
        
        st.dataframe(metrics_df.round(6), use_container_width=True, height=300)
        
    else:
        st.info("👈 Train a network first to see analysis")

with tab3:
    st.subheader("📚 Deep Learning Gradient Descent")
    
    st.markdown("""
    ## Understanding Neural Networks
    
    ### Architecture
    A neural network consists of:
    - **Input Layer**: Receives input features
    - **Hidden Layers**: Process and transform data
    - **Output Layer**: Produces predictions
    
    ### Forward Propagation
    Data flows forward through the network:
    """)
    
    st.latex(r"z^{[l]} = W^{[l]} \cdot a^{[l-1]} + b^{[l]}")
    st.latex(r"a^{[l]} = g(z^{[l]})")
    
    st.markdown("""
    ### Backpropagation
    Gradients flow backward to update weights:
    """)
    
    st.latex(r"\frac{\partial L}{\partial W^{[l]}} = \frac{\partial L}{\partial z^{[l]}} \cdot (a^{[l-1]})^T")
    
    st.markdown("""
    ## Optimization Algorithms
    
    ### 1. SGD (Stochastic Gradient Descent)
    - Basic optimizer
    - Updates: $W = W - \\alpha \\nabla W$
    - **Pros**: Simple, memory efficient
    - **Cons**: Can be slow, sensitive to learning rate
    
    ### 2. SGD with Momentum
    - Accumulates velocity
    - $v = \\beta v - \\alpha \\nabla W$
    - $W = W + v$
    - **Pros**: Faster convergence, dampens oscillations
    - **Cons**: Additional hyperparameter
    
    ### 3. RMSprop
    - Adapts learning rate per parameter
    - Uses moving average of squared gradients
    - **Pros**: Works well with non-stationary objectives
    - **Cons**: Can converge too quickly
    
    ### 4. Adam (Adaptive Moment Estimation)
    - Combines momentum and RMSprop
    - Maintains both first and second moments
    - **Pros**: Fast convergence, robust
    - **Cons**: Can overfit, memory intensive
    
    ## Activation Functions
    
    ### ReLU (Rectified Linear Unit)
    - $f(x) = \\max(0, x)$
    - Most popular for hidden layers
    - Solves vanishing gradient problem
    
    ### Sigmoid
    - $f(x) = \\frac{1}{1 + e^{-x}}$
    - Output range: (0, 1)
    - Good for binary classification
    
    ### Tanh
    - $f(x) = \\tanh(x)$
    - Output range: (-1, 1)
    - Zero-centered
    
    ## Best Practices
    
    1. **Learning Rate**: Start with 0.001-0.01 for Adam, 0.01-0.1 for SGD
    2. **Batch Size**: 32-128 for most problems
    3. **Architecture**: Start small, increase if underfitting
    4. **Activation**: Use ReLU for hidden layers
    5. **Optimizer**: Adam is a good default choice
    
    ## Common Issues
    
    ### Vanishing Gradients
    - **Problem**: Gradients become too small
    - **Solution**: Use ReLU, batch normalization
    
    ### Exploding Gradients
    - **Problem**: Gradients become too large
    - **Solution**: Gradient clipping, lower learning rate
    
    ### Overfitting
    - **Problem**: Model memorizes training data
    - **Solution**: Dropout, regularization, more data
    """)

# Footer
st.sidebar.markdown("---")
st.sidebar.info(f"""
    **Optimizer:** {optimizer.upper()}
    **Activation:** {activation.upper()}
    **Learning Rate:** {learning_rate}
    
    Experiment with different configurations!
""")
