# amazon_product_chatbot_STC
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![SQLite](https://img.shields.io/badge/SQLite-044A64?style=for-the-badge&logo=sqlite&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![FAISS](https://img.shields.io/badge/FAISS-vector_search-00A98F?style=for-the-badge&logo=facebook&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Groq](https://img.shields.io/badge/Groq-LLM_inference-00DC80?style=for-the-badge&logo=ai&logoColor=white)
![Git](https://img.shields.io/badge/Git-F05032?style=for-the-badge&logo=git&logoColor=white)

**Transform your Product reviews into actionable insights with AI-powered analysis.**

## Table of Contents
- [Visual Demo](#visual-demo)
  - [](#employee-attrition-dashboard)
- [Overview](#overview)
- [Data Source & Dictionary](#data-source--dictionary)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Required Libraries](#required-libraries)
- [How to Run](#how-to-run)
- [Key Insights](#key-insights)
- [Project Structure](#project-structure)
- [Results](#results)
- [Author & Acknowledgments](#author--acknowledgments)
- [License](#license)

## Visual Demo
![Demo 1](demo/Dashboard.gif)


## Overview

The Amazon Review Analysis Chatbot is an intelligent web application that transforms raw Amazon review data into meaningful business insights. Built with Streamlit and powered by advanced AI technologies, this tool enables:

- Smart Review Analysis: Understand customer sentiments, preferences, and pain points
- Product Comparison: Compare multiple products based on real user experiences
- Automated Data Processing: Convert Excel files to structured databases instantly
- Semantic Search: Find relevant reviews using FAISS vector similarity
- Interactive Dashboard: Visualize trends and patterns through an intuitive interface

Perfect for e-commerce managers, product analysts, and customer experience teams seeking data-driven decision making.

## Data Source & Dictionary

**Data Dictionary (Key Columns)**  



## Features of the Chatbot
- **📁 Data Upload & Processing:**
  - Upload any Excel file with review data
  
  -  data cleaning and standardization
  
  - Converts to SQLite database automatically
  
  - Creates analytical views for easy querying

- **🤖 Intelligent Chatbot:**
  - FAISS-based semantic search
  
  -  Context-aware responses
  
  - Multiple query types (comparison, recommendation, summary)
  
  - Filtering by category, rating, sentiment


- **📊 Interactive UI:**
  - Real-time data processing
  
  -  Data overview statistics
  
  - Filtering options

- **🔍 Smart Query Handling:**
  - Semantic Search: Finds relevant reviews using FAISS
  
  -  Database Queries: Handles statistical and analytical questions
  
  - Context Awareness: Understands different types of questions
  
  - Filter Integration: Applies user-selected filters automatically

- **📁 Data Upload & Processing:**
  - Upload any Excel file with review data
  
  -  data cleaning and standardization
  
  - Converts to SQLite database automatically
  
  - Creates analytical views for easy querying
 
## 🛠 Technology Stack

### **Core Framework**
![Python](https://img.shields.io/badge/Python-3.8%2B-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)

### **AI & Machine Learning**
![Groq](https://img.shields.io/badge/Groq-Llama_3.3_70B-00DC80?style=for-the-badge&logo=ai&logoColor=white)
![FAISS](https://img.shields.io/badge/FAISS-Vector_Search-00A98F?style=for-the-badge&logo=facebook&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![GloVe](https://img.shields.io/badge/GloVe-100D_Embeddings-8B0000?style=for-the-badge&logo=ai&logoColor=white)

### **Data Processing & Analysis**
![Pandas](https://img.shields.io/badge/Pandas-2.0%2B-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-1.24%2B-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-5.15%2B-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)

### **Database & Storage**
![SQLite](https://img.shields.io/badge/SQLite-3.35%2B-044A64?style=for-the-badge&logo=sqlite&logoColor=white)
![Pickle](https://img.shields.io/badge/Pickle-Serialization-000000?style=for-the-badge&logo=python&logoColor=white)

### **Development Tools**
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![Git](https://img.shields.io/badge/Git-Version_Control-F05032?style=for-the-badge&logo=git&logoColor=white)

### **Data Formats**
![Excel](https://img.shields.io/badge/Excel-.xlsx%2F.xls-217346?style=for-the-badge&logo=microsoftexcel&logoColor=white)
![OpenPyXL](https://img.shields.io/badge/OpenPyXL-Excel_Processing-217346?style=for-the-badge&logo=microsoftexcel&logoColor=white)

## Required Libraries

To run this project, you need to install the following Python packages:

```bash
pip install 
```


## How to Run
1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-directory>
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```
3. Set up your API key:
```bash
export GROQ_API_KEY="your-groq-api-key-here"
```
4. Run the application:
```bash
streamlit run app.py
```
5. Access the Chatbot:
Open your web browser and navigate to `http://127.0.0.1:8050/`


## Key Insights


## Project Structure
```text
├── src/
│   │── app.py
│   └── .streamlit 
├── data/
│   │── amazon_product_reviews.xlsx # Original data
│   │── cleaned_data.xlsx
│   └── processedDF.xlsx
├── embeddings/
│   │── chunks_data.pkl
│   │── faiss_index.bin
│   └── metadata.pkl
├── db/
│   └── reviews_database.db # SQLite database file
├── notebooks/
│   │── chatbot.ipynb
│   └── amazon_product_reviews_analysis.ipynb
├── demo/  
│   └── EmployeeManagement.gif
├── requirements.txt
├── LICENSE
├── .gitignore
└── README.md             # Project documentation
```

## Results


## Author & Acknowledgments

### Author:
- Waref Alyousef

### Acknowledgments:


I gratefully acknowledge the dataset provider for making this data accessible for analysis and learning.

## License

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
