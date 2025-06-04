# Network Intrusion Detection System using MBGWO and Random Forest



## 📋 Overview

This project implements an advanced Network Intrusion Detection System (IDS) that uses the Modified Binary Grey Wolf Optimizer (MBGWO) for feature selection and Random Forest for classification. By analyzing network flow data from the ToN-IoT dataset, the system can accurately detect various types of network intrusions and security threats.

## 🔍 Technical Approach

- **Feature Selection**: Uses Modified Binary Grey Wolf Optimizer (MBGWO) to identify the most relevant features for intrusion detection
- **Classification**: Employs Random Forest algorithm to classify network traffic as normal or malicious
- **Dataset**: Works with the ToN-IoT dataset, specifically the NF-ToN-IoT-v2.csv file
- **Terminal-based**: Operates via command-line interface for processing and analysis

## ✨ Features

- **Efficient Feature Selection**: MBGWO algorithm selects optimal features to improve detection accuracy
- **High Classification Accuracy**: Random Forest classifier provides robust detection of anomalies
- **Batch Processing**: Analyzes large volumes of network flow data efficiently
- **Detailed Results**: Provides comprehensive metrics on detection performance
- **Visualization Tools**: Includes scripts for visualizing results and model performance

## 🔧 Prerequisites

- Python 3.8 or higher
- Required Python libraries (see requirements.txt)
- Sufficient RAM for processing large datasets (8GB+ recommended)
- Access to the ToN-IoT dataset (NF-ToN-IoT-v2.csv)

## 📦 Installation

### Setup

```bash
# Clone the repository
git clone https://github.com/sharjeel-siddiqui12/Intrusion-detection-system-using-MBGWO-and-Random-Forest-TON_IoT_dataset.git
cd Intrusion-detection-system-using-MBGWO-and-Random-Forest-TON_IoT_dataset

# Install dependencies
pip install -r [requirements.txt](http://_vscodecontentref_/0)

# Download the dataset (if not already done)
# Place [NF-ToN-IoT-v2.csv](http://_vscodecontentref_/1) in the project root directory