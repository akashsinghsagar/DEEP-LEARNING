# 🧠 Deep Learning Gradient Descent

A comprehensive collection of interactive deep learning and gradient descent visualization tools built with Python and Streamlit.

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Applications](#applications)
- [Usage](#usage)
- [Technologies](#technologies)
- [Screenshots](#screenshots)
- [Contributing](#contributing)

## 🎯 Overview

This repository contains two powerful Streamlit applications for understanding and visualizing gradient descent algorithms in machine learning and deep learning contexts. Perfect for students, educators, and practitioners who want to gain intuitive understanding of optimization algorithms.

## ✨ Features

### Gradient Descent Visualizer
- 📈 **Function Optimization**: Visualize gradient descent on multiple mathematical functions
- 🎛️ **Interactive Controls**: Adjust learning rate, iterations, and starting points in real-time
- 📊 **Linear Regression**: Complete implementation with cost visualization
- 📉 **Convergence Tracking**: Monitor parameter convergence and cost reduction

### Deep Learning Neural Network Trainer
- 🏗️ **Flexible Architecture**: Configure custom neural network architectures (1-4 hidden layers)
- 🔧 **Multiple Optimizers**: 
  - SGD (Stochastic Gradient Descent)
  - SGD with Momentum
  - RMSprop
  - Adam
- ⚡ **Activation Functions**: ReLU, Sigmoid, Tanh
- 🎨 **Decision Boundary Visualization**: See how networks learn classification boundaries
- 📊 **Real-time Training Metrics**: Loss, accuracy, weight norms, gradient norms
- 🎯 **Multiple Datasets**: Moons, Circles, XOR, Non-linear Regression

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/akashsinghsagar/DEEP-LEARNING.git
cd DEEP-LEARNING
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

### Required Packages
```
streamlit==1.31.0
numpy==1.24.3
pandas==2.0.3
plotly==5.18.0
scikit-learn==1.3.0
```

## 🎮 Applications

### 1. Gradient Descent Visualizer
Explore basic gradient descent concepts with interactive visualizations.

**Run the app:**
```bash
streamlit run gradient_descent_app.py
```

**Features:**
- Optimize quadratic, cubic, and sinusoidal functions
- Adjust learning rate (0.001 - 1.0)
- Set iterations (10 - 500)
- Choose starting points
- View optimization path on function landscape
- Linear regression with synthetic data

### 2. Deep Learning Neural Network
Train neural networks with various architectures and optimizers.

**Run the app:**
```bash
streamlit run deep_learning_gradient_descent.py
```

**Features:**
- Build custom architectures (adjustable layers and neurons)
- Train on 4 different datasets
- Compare optimizer performance
- Visualize decision boundaries
- Monitor training metrics in real-time
- Analyze weight distributions

## 📖 Usage

### Example 1: Basic Gradient Descent

1. Launch the gradient descent app
2. Select a function type (e.g., Quadratic)
3. Adjust the learning rate to 0.1
4. Set starting point to 8.0
5. Click to visualize the optimization path

### Example 2: Training a Neural Network

1. Launch the deep learning app
2. Select "Binary Classification (Moons)" dataset
3. Configure architecture: [2, 32, 16, 2]
4. Choose Adam optimizer
5. Set learning rate to 0.01
6. Train for 100 epochs
7. View decision boundary and metrics

### Example 3: Optimizer Comparison

Try training the same network with different optimizers:
- **SGD**: Learning rate ~0.1
- **Momentum**: Learning rate ~0.05
- **RMSprop**: Learning rate ~0.01
- **Adam**: Learning rate ~0.001-0.01 (usually best results)

## 🛠️ Technologies

- **Python 3.10**: Core programming language
- **Streamlit**: Interactive web applications
- **NumPy**: Numerical computations
- **Pandas**: Data manipulation
- **Plotly**: Interactive visualizations
- **Scikit-learn**: Dataset generation

## 📊 Project Structure

```
DEEP-LEARNING/
│
├── gradient_descent_app.py              # Function optimization visualizer
├── deep_learning_gradient_descent.py   # Neural network trainer
├── requirements.txt                     # Python dependencies
├── README.md                           # Project documentation
│
├── assignment 1.py                     # Course assignments
├── class assignment 1.py
├── class assignment 2.py
├── class assignment1.2.py
└── forward propogation.py
```

## 🎓 Learning Resources

### Key Concepts Covered

**Gradient Descent**
- Learning rate selection
- Convergence analysis
- Local vs global minima
- Optimization landscapes

**Neural Networks**
- Forward propagation
- Backpropagation
- Activation functions
- Loss functions

**Optimizers**
- Vanilla SGD
- Momentum-based methods
- Adaptive learning rates
- Adam algorithm

## 📈 Features in Detail

### Gradient Descent App

#### Function Optimization
- Visualize gradient descent on mathematical functions
- Interactive parameter adjustment
- Real-time path visualization
- Iteration details table

#### Linear Regression
- Synthetic data generation
- Parameter convergence visualization
- Cost function tracking
- True vs learned parameters comparison

### Deep Learning App

#### Network Configuration
- Input layer size (auto-detected from dataset)
- Configurable hidden layers (1-4 layers)
- Adjustable neurons per layer (4-128)
- Output layer (auto-configured)

#### Training Features
- Mini-batch gradient descent
- Real-time progress tracking
- Comprehensive metrics visualization
- Decision boundary plotting for 2D data

#### Analysis Tools
- Weight distribution histograms
- Training metrics tables
- Loss and accuracy curves
- Gradient flow monitoring

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Ideas for Contributions
- Add more optimization algorithms (AdaGrad, Adadelta, Nadam)
- Implement regularization techniques (L1, L2, Dropout)
- Add more datasets and problems
- Create 3D visualization for higher dimensions
- Add model saving/loading functionality
- Implement early stopping
- Add learning rate scheduling

## 📝 License

This project is open source and available under the MIT License.

## 👨‍💻 Author

**Akash Singh Sagar**
- GitHub: [@akashsinghsagar](https://github.com/akashsinghsagar)

## 🙏 Acknowledgments

- Thanks to the Streamlit team for the amazing framework
- Inspired by various deep learning courses and tutorials
- Built for educational purposes

## 📞 Contact

For questions, suggestions, or issues:
- Open an issue on GitHub
- Contact via GitHub profile

---

## 🚀 Quick Start

```bash
# Clone the repo
git clone https://github.com/akashsinghsagar/DEEP-LEARNING.git

# Navigate to directory
cd DEEP-LEARNING

# Install dependencies
pip install -r requirements.txt

# Run gradient descent visualizer
streamlit run gradient_descent_app.py

# OR run deep learning trainer
streamlit run deep_learning_gradient_descent.py
```

## 📚 Additional Resources

- [Gradient Descent Explained](https://en.wikipedia.org/wiki/Gradient_descent)
- [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Deep Learning Specialization by Andrew Ng](https://www.coursera.org/specializations/deep-learning)

---

⭐ **Star this repository if you find it helpful!**

Made with ❤️ and Python
