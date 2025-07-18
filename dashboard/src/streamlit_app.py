import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
from webapp_utlities import (plot_preset, load_data, plot_products_by_subcategory, plot_distribuion_of_reviews_by_rating, plot_top10_most_review_products,
                             distribution_of_experience_related_feedback, distribution_of_size_related_feedback, normalized_size_feedback_by_product_category,
                             average_rating_by_category, total_variants_and_products, total_variants_and_products_dumbell, avg_price_by_product_category,
                            price_distribution_by_category, price_distribution_by_category_binned, user_pie, bert_pie, mismatch_by_category,
                            mismatch_heatmap, mismatch_size_feedback, review_length, review_length_by_category, aspects_raw, aspects_percentage, aspects_shoe, 
                            bertopic_plot, poly_ridge_prediction, random_forest_prediction, xgboost_prediction, plot_poly_ridge_features, plot_random_forest_features,
                            plot_xg_boost_features)

st.set_page_config(page_title="Project Dashboard", layout="wide")
df_main, df_variants, df_reviews, mismatch_df, aspects_df, shoe_aspects_df, topic_summary_df, poly_ridge_prediction_df, random_forest_prediction_df, xgboost_prediction_df, poly_ridge_features, random_forest_features, xgboost_features = load_data()

# Header
st.title("E-Commerce Product Data, Reviews & Sales Analysis Dashboard")

st.markdown(
    """
    Explore interactive insights into product trends, customer sentiment, and sales predictions.
    Navigate through detailed exploratory analysis, natural language processing of reviews, 
    and advanced predictive models — all designed to support data-driven decision-making.
    """
)
# Sidebar Navigation
sections = {
    "1. Exploratory Data Analysis (EDA)": "section-1",
    "2. Natural Language Processing (NLP)": "section-2",
    "3. Predictive Modeling": "section-3",
    "4. Interactive Sales Prediction": "section-4"
}
st.sidebar.markdown("## Navigation")
for title, anchor in sections.items():
    st.sidebar.markdown(f"- [{title}](#{anchor})")

# Eda Header
st.header("1. Exploratory Data Analysis (EDA)", anchor=sections["1. Exploratory Data Analysis (EDA)"])
st.write("""Disclaimer: All visualizations in this project up to predictive modeling were created using Plotly to provide interactive exploration capabilities (e.g., zoom, tooltip, filtering). A logarithmic scale was applied to improve clarity when visualizing data with large value disparities across groups. 
    Log scales show relative differences rather than absolute, making it easier to interpret patterns in skewed data. """)
st.markdown("---")

# 1. Unique products with total category (and variants)
st.header("1.1 Product Catalog Depth & Breadth") 
st.info("Apllied log scale on Product Amount")
col1, col2 = st.columns(2)
with col1:
    fig = total_variants_and_products(df_variants)
    fig = plot_preset(fig)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Grouped bar chart: Quick overview of product and variant volumes.")
with col2:
    fig = total_variants_and_products_dumbell(df_variants)
    fig = plot_preset(fig)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Dumbbell chart: Emphasizes the gap between unique products and variants.")

st.write("This chart compares unique product lines (master products) with the total number of product variants offered in each category. It helps identify categories with high diversification or deep variant customization.")
st.markdown("""
**Insight:** The 'Woman's' category significantly dominates the product catalog with 836 unique products and 2,452 variants. Most categories maintain a healthy 1:2 to 1:3 unique-to-variant ratio, indicating good product diversification, except for 'Clothing' which has a lower variant ratio (276 unique to 413 variants).
""") 
st.markdown("---")


# 2. Unique products by subcategory
st.header("1.2 Dominant Product Subcategories") 
fig = plot_products_by_subcategory(df_main)
fig = plot_preset(fig)
fig.update_layout(height=850)
st.plotly_chart(fig, use_container_width=True)
st.write("This chart shows the number of distinct products available in each subcategory of the catalog, highlighting the most populated product categories.")

st.markdown(f"""
**Insight:** With 36 unique subcategories, 'Women's Shoes' is the overwhelming majority of the catalog, especially within Heels, Boots, and Platforms. This highlights a strong focus on women's footwear within the product offering.
""") 
st.markdown("---")


# 3. Price Section (Avg Price by Category, Price Distribution with Violin Plot, Price Distribution Across Bins)
st.header("1.3 Product Pricing Characteristics") 

fig = avg_price_by_product_category(df_variants)
fig = plot_preset(fig)
st.plotly_chart(fig, use_container_width=True)
st.write("This visualization helps highlight which categories command higher average prices.")
st.markdown("""
**Insight:** Men’s products have the highest average price (116.65), followed by Women’s (91.67), while Accessories are the most affordable (23.25). This suggests deliberate positioning in premium vs. budget segments.""") 

fig = price_distribution_by_category(df_variants)
fig = plot_preset(fig)
st.plotly_chart(fig, use_container_width=True)
st.write("This violin plot shows that most categories are evenly distributed and have a wide range, except for Kids 49-59 and Accessories 10-50.")
st.markdown("""
**Insight:** Most categories show broad price variability, but Kids and Accessories stand out with tightly clustered prices — typically 49–59 and 10–50 respectively — implying more standardized pricing within these lines.""") 

fig = price_distribution_by_category_binned(df_variants)
fig = plot_preset(fig)
st.plotly_chart(fig, use_container_width=True)
st.write("This plot shows how each category is distributed across different price tiers")
st.markdown("""
**Insight:** The majority of 'Men's' products (45 out of 214 are in the >100 price bin. In contrast, 'Accessories' are exclusively in the <50 bins. This bin analysis confirms distinct pricing strategies and market segments for each product category.
""") 
st.markdown("---")


# 4. Distribution of reviews
st.header("1.4 Overall Customer Rating Trend") # Changed header
fig = plot_distribuion_of_reviews_by_rating(df_main)
fig = plot_preset(fig)
st.plotly_chart(fig, use_container_width=True)
st.write("""This chart illustrates that the majority of products have high average ratings, typically clustered between 4.0 and 5.0.
          Since the data reflects per-product average ratings, not individual reviews, the visualization shows the distribution of product-level satisfaction.
         Products with average ratings below 3.5 are extremely rare, making the full scale (1–5) visually sparse and unbalanced. To improve readability and focus on meaningful segments, the interactive Plotly chart allows viewers to zoom into the 3.5–5.0 range, where the bulk of rated products reside.
         """)
st.markdown("""
**Insight:** Product-level ratings are overwhelmingly positive — most items average around 4.3 stars, and only 173 products fall below a 3.6 average. This pattern suggests strong general customer satisfaction or potential rating inflation across the catalog.""") 
st.markdown("---")


# 5. Most reviewed products
st.header("1.5 Top Products by Customer Engagement") 
fig = plot_top10_most_review_products(df_main)
fig = plot_preset(fig)
st.plotly_chart(fig, use_container_width=True)
st.write("This chart identifies the top 10 most reviewed products allong with categories the ybelong to, indicating high popularity and customer engagement for these items.")
st.markdown("""
**Insight:** 'Maxima-r' and 'Possession' from the 'Woman-sneakers' subcategory are the most popular products by review count. The top 10 is predominantly 'Woman's' products (sandals, boots, sneakers, platforms), confirming their market dominance.
""") #
st.markdown("---")


# 6. Size/Experience/Verified feedback
st.header("1.6 Deep Dive into Customer Feedback Types") 
col1, col2 = st.columns(2)

with col1:
    fig1 = distribution_of_size_related_feedback(df_main)
    fig1 = plot_preset(fig1)
    fig1.update_layout(width=800, height=550)  
    st.plotly_chart(fig1, use_container_width=False)  
    st.write("This chart illustrates the distribution of size-related feedback, highlighting how users perceive the fit of products (e.g., true to size, too big, too small).")
    st.markdown("""
    **Insight:** While ~11,000 users are satisfied with sizing ('True to Size'), 6,306 reported issues are split between 'Too Big' and 'Small' and others 2,111 between 'Too small' and 'Big'. This highlights sizing as a significant area for potential product improvement.
    """) 


with col2:
    fig2 = distribution_of_experience_related_feedback(df_main)
    fig2 = plot_preset(fig2)
    fig2.update_layout(width=700, height=525)
    st.plotly_chart(fig2, use_container_width=False)
st.write(
    """This chart shows the distribution of experience-related feedback tags, based on user input. It reflects how often users described products as 'Stylish', 'Comfortable', or 'High Quality'. The chart also emphasizes that most users did not leave any experience-related feedback at all."""
)
st.markdown("""
**Insight:** Experience feedback is largely underused — 18,464 users skipped it entirely. Among the 3,263 who did engage, 'Stylish' (1,355) and 'Comfort' (1,037) dominate, while 'Quality' (871) lags behind. This suggests that users prioritize looks and comfort in their feedback, and there’s room to encourage more detailed engagement in this section.
""")

st.subheader("Verified Reviewer Status") 
st.write(
    """The verification data shows a strong imbalance across user types: 18,379 are Verified Buyers, 569 are Verified Reviewers, and 1,243 are Unverified. This suggests that most reviews originate from confirmed purchasers.
"""
)
st.markdown("""
**Insight:** The overwhelming presence of 'Verified Buyers' significantly boosts the credibility of the review dataset. With unverified users making up only a small fraction, the overall trustworthiness of the feedback is high.
""")
st.markdown("---")


# 7. Normalized size feedback by category
st.header("1.7 Category-Specific Sizing Accuracy") # Changed header
fig = normalized_size_feedback_by_product_category(df_main)
fig = plot_preset(fig)
st.plotly_chart(fig, use_container_width=True)
st.write("This visualization compares how users perceive sizing across product categories.")
st.markdown("""**Insight:** Sizing satisfaction is highest in the 'Men’s' category (61.1% report 'True to Size'), while 'Clothing' shows the lowest agreement (43.1%). Across most categories, 'Too Big' is the dominant complaint — except in 'Women’s', where more users report items as 'Small'. This points to category-specific fit issues that may require targeted sizing adjustments.
""")
st.markdown("---")


# 8. Average rating by category
st.header("1.8 Customer Satisfaction vs. Review Volume by Category") # Changed header
fig = average_rating_by_category(df_main)
fig = plot_preset(fig)
st.plotly_chart(fig, use_container_width=True)
st.write("This plot reveals how categories differ in both popularity and customer satisfaction. Using a log scale for review counts helps reduce skew caused by highly reviewed categories, while displaying the average rating directly on each bar maintains clarity.")
st.markdown("""
**Insight:** The 'Women’s' category accounts for the majority of reviews (~43,000), far surpassing all others. Despite having the lowest average rating (4.56), this likely reflects its large volume rather than poor performance. In contrast, 'Handbags' achieve the highest average rating (4.87), suggesting consistently strong user satisfaction across smaller, more focused categories.""") 
st.markdown("---")

# NLP header 
st.title("2. Neural Language Processing (NLP)", anchor=sections["2. Natural Language Processing (NLP)"])
st.markdown("---")
st.info(
    "**Understanding BERT in this Context:** "
    "BERT (Bidirectional Encoder Representations from Transformers) is an advanced AI model "
    "that processes and understands human language, like customer reviews. In this analysis, "
    "it's used to automatically determine the sentiment (e.g., positive, negative, neutral) "
    "expressed in the text, providing an objective interpretation of customer feedback."
)
st.markdown("---")

st.header("2.1 Users and BERT score distribution") 
col1, col2 = st.columns(2)
with col1:
    fig = user_pie(df_reviews)
    fig = plot_preset(fig)
    st.plotly_chart(fig, use_container_width=True)
with col2:
    fig = bert_pie(df_reviews)
    fig = plot_preset(fig)
    st.plotly_chart(fig, use_container_width=True)
st.write("""This comparison reveals consistently positive customer sentiment in both user-assigned ratings and AI-driven sentiment analysis. While the majority of reviews are classified as positive in both cases, the AI model shows a slight drop in positivity. This may indicate more nuanced or mixed emotions expressed in the text that aren't fully reflected in the numerical ratings, suggesting that users sometimes rate generously even when their wording is more reserved.
""")
st.markdown("---")

# 2 Heatmap
st.header("2.2 BERT Model Agreement Insight")
fig = mismatch_heatmap(df_reviews)
fig = plot_preset(fig)
st.plotly_chart(fig, use_container_width=True)
st.write("""While 5-star reviews largely align with positive textual sentiment, significant divergence occurs elsewhere. 
   While 5-star reviews generally align with highly positive textual sentiment, notable discrepancies appear in other ratings. For instance, many 1-star reviews reflect only mildly negative (2-star) sentiment, while some 5-star reviews correspond to 4-star sentiment in text. This suggests that user-assigned ratings tend to be more polarized than the actual language used, highlighting the value of analyzing review content beyond star scores to fully understand customer experience.
    """)
st.markdown("---")

# 3. Mismatch comparison by category
st.header("2.3 Mismatch Analysis Between User Scores and BERT Sentiment by Category")
fig = mismatch_by_category(df_reviews)
fig = plot_preset(fig)
st.plotly_chart(fig, use_container_width=True)
st.write("""While the 'Women’s' category has the highest number of mismatched reviews due to its large volume, categories like 'Men’s' exhibit a proportionally higher rate of sentiment mismatches. This suggests that reviews in these categories may carry more nuanced or mixed signals, making them valuable targets for deeper text-based analysis and product improvement opportunities.""")
st.markdown("---")

# 4 Mismatch size feedback
st.header("2.4 Mismatch size")
fig = mismatch_size_feedback(df_reviews)
fig = plot_preset(fig)
st.plotly_chart(fig, use_container_width=True)
st.write("""While the 'Women’s' category has the highest number of mismatched reviews due to volume, categories like 'Men’s' show a proportionally greater rate of sentiment discrepancies. This points to more nuanced or conflicted opinions within those reviews. Interestingly, many users still give high ratings despite mentioning sizing issues — except in cases marked as 'Too Small', which appears to be the most negatively impactful fit issue. These patterns emphasize the need to evaluate written feedback alongside star ratings for a fuller understanding of customer sentiment.
""")
st.markdown("---")

# 5 Review length
st.header("2.5 Review Length Insights")
col1, col2 = st.columns([3, 7])  # 30% and 70%

with col1:
    fig = review_length(df_reviews)
    fig = plot_preset(fig)
    st.plotly_chart(fig, use_container_width=True)


with col2:
    fig = review_length_by_category(df_reviews)
    fig = plot_preset(fig)
    st.plotly_chart(fig, use_container_width=True)

st.write("""These plots show sentiment distribution across key product aspects from two perspectives: absolute counts and percentage breakdowns. Aspects like 'style_and_appearance' and 'comfort' lead in positive review volume, reflecting strong customer satisfaction. However, 'size_and_fit' stands out with the highest number of neutral and negative responses — both in absolute terms and percentage — indicating consistent issues with product fit. Notably, 'style_and_appearance' also has the second-highest count of neutral and negative reviews in absolute terms, though its overall sentiment remains highly positive when viewed proportionally. While most aspects maintain over 80% positive sentiment, 'delivery' shows the highest percentage of negative feedback (~15%), highlighting sizing and logistics as the most commonly mentioned areas of dissatisfaction despite the general positivity.
""")
st.markdown("---")    

# 6 ABSA
st.header("2.6 Aspect Based Sentiment Analysis")
st.info(
    "**Understanding Aspect-Based Sentiment Analysis (ABSA):** "
    "ABSA breaks down customer reviews to identify sentiment about specific product features or aspects. "
    "Instead of just knowing if a review is positive or negative overall, it shows how customers feel about particular parts, "
    "like price, quality, or delivery, providing more detailed insights."
)
# 6.1 Main aspects
st.subheader("2.6.1 Detailed Customer Sentiment by Aspect")
col1, col2 = st.columns(2) 
with col1:
    fig = aspects_raw(aspects_df)
    fig = plot_preset(fig)
    st.plotly_chart(fig, use_container_width=True)
    st.write("""
    This plot shows the absolute count of reviews for each general product aspect, broken down by sentiment. 
    Most aspects, especially 'style_and_appearance' and 'comfort', have very high positive review counts, indicating overall strong satisfaction. 
    'Delivery' and 'price_and_value' show noticeable negative and neutral counts, suggesting specific areas of concern despite the overall positive trend.
    """)

with col2:
    fig = aspects_percentage(aspects_df)
    fig = plot_preset(fig)
    st.plotly_chart(fig, use_container_width=True)
    st.write("""
    This plot displays the percentage distribution of sentiment within each general product aspect. 
    All aspects consistently show high positive sentiment percentages, often above 80%. 
    'Delivery' has the highest proportion of negative sentiment at approximately 15%, while 'size_and_fit' shows significant neutral and negative proportions, indicating more mixed feedback in these areas.
    """)
st.markdown("---")

# 6.2 Shoe aspects
st.subheader("2.6.2 Detailed Customer Sentiment by Shoe Type")
fig = aspects_shoe(shoe_aspects_df)
fig = plot_preset(fig)
st.plotly_chart(fig, use_container_width=True)
st.write("""
This plot offers a detailed breakdown of sentiment across individual shoe types. Categories like 'Boots', 'Heels', 'Sandals', and 'Sneakers' receive consistently high volumes of positive reviews, reflecting strong customer satisfaction in these popular segments. In contrast, 'Flats' show a more balanced sentiment distribution, with relatively higher proportions of neutral and negative feedback. This granularity helps identify specific footwear categories that may benefit from closer attention and targeted improvements.
         """)
st.markdown("---")

# 7 BERTopic
st.header("2.7 Detailed Customer Sentiment & Key Discussion Themes")
st.info(
    "**Understanding BERTopic:** "
    "BERTopic is a technique that groups customer reviews into topics based on the themes or subjects they discuss. "
    "This helps identify common issues, preferences, or features customers talk about without manually reading all reviews."
)
fig = bertopic_plot(topic_summary_df)
fig = plot_preset(fig)
fig.update_layout(
        xaxis_title="Topic",
        yaxis_title="Percentage",
        legend_title="",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1,
            xanchor="center",
            x=0.5
        ),
        xaxis_tickangle=30,
        margin=dict(b=150, t=80)
    )
st.plotly_chart(fig, use_container_width=True)
st.write("""Customer sentiment is overwhelmingly positive across core product attributes like 'Style & Appearance' and 'Comfort'. However, certain areas such as 'Delivery' and 'Size & Fit' receive a noticeably higher share of neutral and negative feedback.
Sentiment analysis by shoe type shows variation: 'Boots' and 'Sneakers' perform particularly well, while 'Flats' receive more mixed feedback, indicating potential fit or comfort concerns.
BERTopic analysis further uncovers key themes behind the reviews. While most topics—such as general product praise and footwear comfort—are strongly positive, others like 'Discomfort & Foot Pain' and 'Foot Width & Fit Issues' exhibit elevated levels of negative or neutral sentiment. Together, these layered insights highlight clear opportunities for targeted product improvements and service enhancements across the catalog.
""")
st.markdown("---")

st.header("3. Predictive Modeling: Sales Forecasting", anchor=sections["3. Predictive Modeling"])

st.subheader("Understanding Key Drivers Behind Sales and Making Accurate Predictions")

st.info(
    "**Understanding Predictive Modeling:** "
    "Predictive modeling uses historical data to forecast future outcomes. "
    "Here, the model estimates expected sales based on product features, pricing, and other factors, "
    "helping to anticipate demand and make informed business decisions."
)
st.header("3.1 Model Performance: Actual vs. Predicted Sales")
col1, col2, col3 = st.columns(3)

with col1:
    poly_ridge_prediction(poly_ridge_prediction_df)
    st.write("This chart compares predicted vs. actual sales using the Polynomial Ridge Regression model.")
    st.markdown("**Insight:** Polynomial Ridge captures both linear and non-linear relationships, revealing how combined feature effects (like price and holiday timing) influence sales outcomes.")
    st.markdown("---")

with col2:
    random_forest_prediction(random_forest_prediction_df)
    st.write("This chart evaluates Random Forest model accuracy by comparing its predicted and actual sales values.")
    st.markdown("**Insight:** Random Forest delivers strong predictive performance, with close alignment between actual and predicted values—proving useful for reliable forecasting.")
    st.markdown("---")

with col3:
    xgboost_prediction(xgboost_prediction_df)
    st.write("This chart shows how well XGBoost predicts sales, demonstrating minimal error compared to actual sales.")
    st.markdown("**Insight:** XGBoost achieves the highest predictive accuracy, making it ideal for critical business forecasting and decision support.")
    st.markdown("---")

st.header("3.2 Key Sales Drivers: Feature Importance Analysis")

left, center, right = st.columns([1, 2, 1])

with center:
    st.subheader("Polynomial Ridge: Feature Contributions")
    plot_poly_ridge_features(poly_ridge_features)
    st.write("This chart highlights how each feature contributes to sales in the Polynomial Ridge model, including interaction and curved relationships.")
    st.markdown("**Insight:** This model reveals complex behaviors—for example, the impact of price or discounts may depend on timing like holiday promotions.")
    st.markdown("---")

    st.subheader("Random Forest: Feature Importance")
    plot_random_forest_features(random_forest_features)
    st.write("This chart ranks the most influential variables in the Random Forest model for predicting sales.")
    st.markdown("**Insight:** Average price, weekday, and holiday events emerge as dominant sales drivers, validating their role in strategic planning.")
    st.markdown("---")

    st.subheader("XGBoost: Feature Importance")
    plot_xg_boost_features(xgboost_features)
    st.write("This chart shows which features most strongly influence XGBoost's predictions of sales outcomes.")
    st.markdown("**Insight:** XGBoost highlights holiday events, ad status, and price as top predictors—supporting highly targeted marketing and pricing strategies.")
    st.markdown("---")


# Model metrics comparison
data = {
    "Model": ["Ridge", "Random Forest", "XGBoost"],
    "R² Score": [0.949, 0.971, 0.975],
    "MAE": [5.84, 4.42, 4.109],
    "MSE": [54.89, 31.07, 26.665],
    "RMSE": [7.41, 5.57, 5.164],
}

metrics_df = pd.DataFrame(data)
st.subheader("3.3 Comparative Model Performance Metrics")
st.dataframe(metrics_df, height=200, width=400)
st.markdown("**Insight:** XGBoost demonstrates superior performance across all evaluation metrics (highest R² and lowest errors), making it the most reliable choice for sales forecasting within this analysis.")

st.write("---") 
st.header("3.4 Overall Business Insights from Predictive Modeling")
st.write("""
Based on our predictive modeling analysis, several key insights emerge to support data-driven sales optimization:

* **High-Accuracy Forecasting:** The **XGBoost model** demonstrates outstanding accuracy (R² = 0.975), offering a reliable tool for forecasting sales. This enables a shift from intuition-based decisions to confident, data-backed planning.

* **Key Drivers of Sales Identified:** Across all models, the most influential factors include:
    * **Pricing & Discounts:** Higher prices tend to reduce sales, while discounts significantly boost them—especially when combined with holiday events. This supports the use of dynamic pricing strategies tailored to market timing.
    * **Marketing Effectiveness:** Active ads (`ad_status`) and holiday-driven campaigns (`holiday_event`) are strongly associated with sales growth, highlighting where to focus marketing budgets for maximum ROI.
    * **Customer Satisfaction:** Higher product ratings directly correlate with increased sales, emphasizing the importance of quality, service, and encouraging positive feedback.
    * **Sales Timing:** Weekly sales patterns (`weekday`) provide opportunities to optimize staffing, logistics, and inventory based on predictable demand fluctuations.

* **Nonlinear Effects Revealed:** While XGBoost offers predictive power, the **Polynomial Ridge model** adds interpretability by capturing **non-linear effects and variable interactions**—such as how discounts behave differently during promotions—offering valuable context for strategic planning.

* **Actionable Business Value:** These insights inform real-world strategies across:
    - Inventory and logistics optimization
    - Marketing and ad spend efficiency
    - Customer experience management
    - Dynamic pricing and promotion planning

**Note:** While the underlying sales data is simulated, the modeling approach and resulting insights are directly transferable to real-world datasets, providing a solid foundation for future analytics-driven decision-making.
""")
st.header("**4. Interactive Sales Prediction:**", anchor=sections["4. Interactive Sales Prediction"])
st.write(""" This dashboard is equipped to provide real-time sales predictions, allowing for immediate "what-if" scenario analysis. Users can interact with the prediction capabilities in two ways:
    1.  **Bulk Prediction (CSV Upload):** Upload a CSV file containing new product data (with features like price, rating, discount, etc.) to receive sales predictions for all entries at once. This is ideal for forecasting sales across a large product catalog or for future inventory planning.
    2.  **Single Prediction (Manual Input):** Manually input specific values for each sales driver (e.g., average price, whether it's a holiday, discount percentage) directly into the dashboard. This allows for quick, on-the-fly forecasts for individual products or hypothetical scenarios to assess the potential impact of changing specific variables.
""")
# Subcategory dropdown
subcategory_options = [
    "Womens-sandals", "Womens-heels", "Womens-sneakers", "Womens-loafers-flats",
    "Womens-boots-booties", "Womens-platforms", "Womens-wedges", "Womens-slip-on-slides",
    "Womens-mules", "Mens-dress-shoes", "Mens-sneakers", "Mens-boots", "Mens-loafers",
    "Mens-sandals", "Mens-casual", "Kids-mini-me", "Kids-special-occasion-shoes",
    "Kids-flower-girl-shoes", "Kids-pretty-in-pink", "Kids-sporty-chic", "Kids-beach-days",
    "Handbags-crossbody-bags", "Handbags-clutches", "Handbags-belt-bags",
    "Handbags-wallets-charms", "Clothing-dresses", "Clothing-tops-shirts",
    "Clothing-bottoms", "Clothing-jackets-coats", "Clothing-blazers",
    "Clothing-faux-leather", "Accessories-socks-tights", "Accessories-sunglasses",
    "Accessories-hats-gloves-scarves", "Accessories-shoe-care",
    "Accessories-fashion-jewelry"
]
st.markdown(
    "**Instructions:**\n"
    "- Upload a CSV with columns: `subcategory`, `avg_price`, `rating`, `month`, `weekday`, `discount_percent`, `ad_status`, `holiday_event`.\n"
    "- Use the dropdown below to check valid `subcategory` values.\n"
    "- Make sure your CSV only contains one subcategory and it matches one from the list.\n"
    "- Example row: `Womens-sandals, 100, 4, 6, 6, 10, 0, 1`"
)
subcategory = st.selectbox("Subcategory", options=subcategory_options)
# XGBoost model
models_path = Path(__file__).parent.parent / "models"
path = (models_path / "xgboost.pkl")
xgb_m = joblib.load(path)

uploaded_file = st.file_uploader("Upload CSV file for prediction", type=["csv"])

if uploaded_file is not None:
    new_data = pd.read_csv(uploaded_file)
    new_data["subcategory"] = new_data["subcategory"].astype("category")
    st.write("Preview of input data:")
    st.dataframe(new_data.head())

    # Predict
    preds = xgb_m.predict(new_data)

    # Show results
    new_data["Predicted Sales"] = preds
    st.write("Predicted Sales:")
    st.dataframe(new_data)

    # Download option
    csv = new_data.to_csv(index=False).encode('utf-8')
    st.download_button("Download predictions CSV", data=csv, file_name="predictions.csv")


st.header("Manual Input for Prediction")

with st.form("input_form"):
    subcategory = st.selectbox("Subcategory", options=subcategory_options)
    avg_price = st.number_input("Average Price", min_value=0, step=1)
    rating = st.slider("Rating", min_value=1, max_value=5, step=1)
    month = st.slider("Month", min_value=1, max_value=12, step=1)
    weekday = st.slider("Weekday", min_value=0, max_value=6, step=1)
    discount_percent = st.slider("Discount Percent", min_value=1, max_value=100, step=1)
    ad_status = st.checkbox("Ad Status")
    holiday_event = st.checkbox("Holiday Event")

    submitted = st.form_submit_button("Predict")

if submitted:
    input_df = pd.DataFrame({
        'subcategory': pd.Categorical([subcategory], categories=subcategory_options),
        'avg_price': [avg_price],
        'rating': [rating],
        'month': [month],
        'weekday': [weekday],
        'discount_percent': [discount_percent],
        'ad_status': [int(ad_status)],
        'holiday_event': [int(holiday_event)]
    })
    prediction = xgb_m.predict(input_df)[0]
    st.success(f"Predicted Sales: {prediction:.2f}")