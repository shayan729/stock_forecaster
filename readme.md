# AI-Powered Stock Forecasting

A **Flask web application** that uses **Machine Learning (LSTM and Random Forest)** to analyze and forecast stock price trends based on historical market data.

## Prerequisites
- **Python 3.10** (or any version explicitly compatible with TensorFlow)
- **pip** (Python package installer)

It’s recommended to use Python 3.10 for optimal compatibility with TensorFlow and other ML libraries.

## 🛠 Setup & Installation

Follow these steps to set up and run the project locally.

### 1. Clone the repository
`git clone <your-repository-url>`

`cd <repository-folder-name>`

### 2. Create and activate a virtual environment (Mandatory Step)
Using a virtual environment helps manage project dependencies cleanly.

On macOS/Linux:

`python3.10 -m venv .venv`

`source .venv/bin/activate`

### 3. Install dependencies
Install all required Python packages from the requirements.txt file:

`pip install -r requirements.txt`

### 4. Create model directories
The application needs folders to store trained model files:

`mkdir saved_models`

`mkdir saved_models_rf`

## 🚀 Running the Application

### 1. Start the Flask server
`flask run`


Alternatively:

`python app.py`