# E-Commerce-Product-Review-Analysis

A data analysis project exploring product trends, pricing dynamics, customer sentiment, and predictive modeling in the e-commerce domain. This project aims to deliver actionable insights through interactive visualizations and a real-time sales prediction tool.

---

## Project Intent

This project combines analytical depth with practical presentation. While the full technical report details the complete workflow — including data preparation, exploratory analysis, natural language processing, and predictive modeling — the accompanying dashboard offers a **concise, interactive overview** of the key insights and results.

The dashboard is intended as the primary touchpoint for reviewing outcomes and exploring patterns, while the report provides a deeper look into the underlying methods and decisions supporting the analysis.

---

## Project Structure

1. **Data Preparation**  
   - Data cleaning and transformation  
   - Handling missing values and inconsistent formats  
   - Feature engineering

2. **Exploratory Data Analysis (EDA)**  
   - Product catalog structure  
   - Pricing and popularity trends  
   - Rating distribution and category-level performance

3. **Text Analysis (NLP)**  
   - Sentiment classification using BERT
   - Alignment of review sentiment with rating data
   - Aspect-based sentiment analysis and topic modeling with BERTopic

4. **Predictive Modeling**  
   - Feature selection and preprocessing  
   - Model training and evaluation (Poly Ridge, Random Forest, XGBoost)  
   - Performance comparison and model selection

5. **Interactive Dashboard**  
   - Built with Streamlit  
   - Visual summaries of EDA and NLP and Predictive Modeling
   - Real-time sales prediction using user-defined inputs

6. **Deployment**  
   - Model and dashboard deployed via Streamlit Cloud  
   - Lightweight, browser-accessible interface for business users

---

## Technologies Used

- **Languages**: Python  
- **Data Processing**: pandas, numpy, re  
- **Visualization**: matplotlib, seaborn, plotly  
- **NLP**: spaCy, TextBlob, transformers (BERT), wordcloud  
- **Modeling**: scikit-learn  
- **Web App**: Streamlit  
- **Deployment**: Streamlit Cloud

---

## Sample Visuals

<details>
<summary>Click to expand</summary>

- Price vs Rating distribution by subcategory  
- Sentiment mismatch heatmaps by product category  
- Wordclouds for positive and negative review themes  
- Model performance bar charts  
- Interactive filters in dashboard

</details>

---

## Results Summary

- Identified subcategories with significant mismatches between star ratings and review sentiment  
- Demonstrated clear patterns in review volume and satisfaction trends  
- Built and compared multiple predictive models with solid accuracy  
- Delivered a fully functional dashboard with insights and prediction capabilities

---

## Live Demo

> **🔗 [Streamlit App Link – Try It Live](#)** *(URL wheb i set it io)](https://e-commerce-analysis-dashboard.onrender.com)*  
> Use the dashboard to explore the data and predict sales based on your inputs.

---


