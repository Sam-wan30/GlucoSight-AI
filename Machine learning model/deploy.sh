#!/bin/bash

# Diabetes Risk Prediction App - Deployment Script
# ==================================================

set -e

echo "🏥 Diabetes Risk Prediction App - Deployment Script"
echo "===================================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to print colored output
print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Check Python version
echo "📋 Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
print_success "Python version: $python_version"

# Check if required directories exist
echo ""
echo "📁 Checking directory structure..."
if [ ! -d "models" ]; then
    mkdir -p models
    print_success "Created models directory"
else
    print_success "models directory exists"
fi

if [ ! -d "../data" ]; then
    print_error "data directory not found. Please ensure the data directory exists."
    exit 1
else
    print_success "data directory exists"
fi

# Install dependencies
echo ""
echo "📦 Installing dependencies..."
if [ -f "../requirements.txt" ]; then
    pip3 install -r ../requirements.txt
    print_success "Dependencies installed successfully"
else
    print_error "../requirements.txt not found"
    exit 1
fi

# Check if dataset exists
echo ""
echo "📊 Checking dataset..."
if [ -f "../data/diabetes_cleaned.csv" ]; then
    print_success "Dataset found: diabetes_cleaned.csv"
elif [ -f "../data/diabetes.csv" ]; then
    print_warning "Using raw dataset: diabetes.csv (will be preprocessed automatically)"
else
    print_error "Dataset not found. Please place diabetes.csv or diabetes_cleaned.csv in the data directory."
    exit 1
fi

# Set environment variables
echo ""
echo "⚙️  Setting environment variables..."
export STREAMLIT_SERVER_PORT=8501
export STREAMLIT_SERVER_ADDRESS=localhost
print_success "Environment variables set"

# Run the application
echo ""
echo "🚀 Starting Diabetes Risk Prediction App..."
echo "   The app will be available at: http://localhost:8501"
echo "   Press Ctrl+C to stop the server"
echo ""

streamlit run streamlit_app_new.py
