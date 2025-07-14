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
# Eda Header
st.header("1. Exploratory Data Analysis (EDA)")
st.write("Disclaimer: All visualizations in this project were created using Plotly to provide interactive exploration capabilities (e.g., zoom, tooltip, filtering). In certain plots, a logarithmic scale was applied to the y-axis or x-axis to improve clarity when displaying data with large value disparities across categories.")
st.markdown("---")

# 1. Unique products with total category (and variants)
st.header("1.1 Product Catalog Depth & Breadth") 
st.subheader("Aplliead log scale on Product Amount")
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
st.header("1.2 Dominant Product Subcategories") # Changed header
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
**Insight:** 'Men's' products are the most expensive on average 116.65, followed by 'Woman's' 91.67. 'Accessories' are the most affordable 23.25, indicating distinct pricing strategies across categories.
""") 

fig = price_distribution_by_category(df_variants)
fig = plot_preset(fig)
st.plotly_chart(fig, use_container_width=True)
st.write("This violin plot shows that most categories are evenly distributed and have a wide range, except for Kids 49-59 and Accessories 10-50.")
st.markdown("""
**Insight:** While most categories exhibit wide price ranges, 'Kids' and 'Accessories' show highly concentrated distributions. 'Kids' products primarily fall between 49-59, and 'Accessories' between 10-50, suggesting specific target price points for these segments.
""") 

fig = price_distribution_by_category_binned(df_variants)
fig = plot_preset(fig)
st.plotly_chart(fig, use_container_width=True)
st.write("This plot shows how each category is distributed across different price tiers")
st.markdown("""
**Insight:** The majority of 'Men's' products (45 out of 214 are in the >100 price bin. In contrast, 'Accessories' are exclusively in the <50 bins. This bin analysis confirms distinct pricing strategies and market segments for each product category.
""") # Removed div for styling
st.markdown("---")


# 4. Distribution of reviews
st.header("1.4 Overall Customer Rating Trend") # Changed header
fig = plot_distribuion_of_reviews_by_rating(df_main)
fig = plot_preset(fig)
st.plotly_chart(fig, use_container_width=True)
st.write("This chart shows that the majority of user reviews are clustered at the higher end of the rating scale (4–5 star review), suggesting either strong product satisfaction or rating inflation. Analysis of the raw data revealed that ratings below 3.5 had a negligible number of associated reviews, leading to a highly stretched and uninformative visualization when viewed in its entirety. To enhance clarity and highlight the ratings with significant user engagement, viewers can focus on rating range from 3.5 to 5.0  utilizing the interactive features of the Plotly graph.")
st.markdown("""
**Insight:** Product average ratings are overwhelmingly positive, with most products averaging a 4.3 rating. Average ratings below 3.6 are nearly non-existent (only 173 products), indicating high overall customer satisfaction.
""") # Removed div for styling
st.markdown("---")


# 5. Most reviewed products
st.header("1.5 Top Products by Customer Engagement") # Changed header
fig = plot_top10_most_review_products(df_main)
fig = plot_preset(fig)
st.plotly_chart(fig, use_container_width=True)
st.write("This chart identifies the top 10 most reviewed products, indicating high popularity and customer engagement for these items.")
st.markdown("""
**Insight:** 'Maxima-r' and 'Possession' from the 'Woman-sneakers' subcategory are the most popular products by review count. The top 10 is predominantly 'Woman's' products (sandals, boots, sneakers, platforms), confirming their market dominance.
""") # Removed div for styling
st.markdown("---")


# 6. Size/Experience/Verified feedback
st.header("1.6 Deep Dive into Customer Feedback Types") # Changed header
col1, col2 = st.columns(2)

with col1:
    fig1 = distribution_of_size_related_feedback(df_main)
    fig1 = plot_preset(fig1)
    fig1.update_layout(width=800, height=550)  # Explicit width and height
    st.plotly_chart(fig1, use_container_width=False)  # Disable auto-width
    st.write("This chart illustrates the distribution of size-related feedback, highlighting how users perceive the fit of products (e.g., true to size, too big, too small).")
    st.markdown("""
    **Insight:** While ~11,000 users are satisfied with sizing ('True to Size'), 6,306 reported issues are split between 'Too Big' and 'Small' and others 2,111 between 'Too small' and 'Big'. This highlights sizing as a significant area for potential product improvement.
    """) 


with col2:
    fig2 = distribution_of_experience_related_feedback(df_main)
    fig2 = plot_preset(fig2)
    fig2.update_layout(width=700, height=525)
    st.plotly_chart(fig2, use_container_width=False)
    st.write("Analysis of Experience related feedback suggests that users don't utilize this metric enough when leaving feedback as overwhelming majority 18464 didn't leave any while only 3263 users left experience feedbakc from which majority highlight stylisg 1355 and comfort 1037 with qulity beein last at 871")
    st.markdown("""
    **Insight:** Experience feedback is underutilized (18,464 users left none). Among those who did, 'Stylish' (1,355) and 'Comfort' (1,037) are the most common positive attributes, while 'Quality' (871) is less frequently mentioned. This suggests opportunities to encourage more specific feedback.
    """) 

st.subheader("Verified Reviewer Status") # New subheader for clarity
st.write("""
"The distribution of verification statuses is heavily skewed, with 18,379 users identified as Verified Buyers, 569 as Verified Reviewers, and 1,243 as Unverified users."
""")
st.markdown("""
**Insight:** A strong majority (18,379 users) are 'Verified Buyers', lending significant credibility to the reviews. Only a small fraction (1,243) are unverified, reinforcing trust in the feedback system.
""") 
st.markdown("---")


# 7. Normalized size feedback by category
st.header("1.7 Category-Specific Sizing Accuracy") # Changed header
fig = normalized_size_feedback_by_product_category(df_main)
fig = plot_preset(fig)
st.plotly_chart(fig, use_container_width=True)
st.write("This visualization compares how users perceive sizing across product categories.")
st.markdown("""
**Insight:** 'Men's' products have the highest satisfaction with sizing (61.1% 'True to Size'), while 'Clothing' has the lowest (43.1%). 'Too Big' is the most common complaint across most categories, except 'Woman's' where 'Small' is more prevalent. This highlights category-specific fit challenges.
""") 
st.markdown("---")


# 8. Average rating by category
st.header("1.8 Customer Satisfaction vs. Review Volume by Category") # Changed header
fig = average_rating_by_category(df_main)
fig = plot_preset(fig)
st.plotly_chart(fig, use_container_width=True)
st.write("This plot reveals how categories differ in both popularity and customer satisfaction. Using a log scale for review counts helps reduce skew caused by highly reviewed categories, while displaying the average rating directly on each bar maintains clarity.")
st.markdown("""
**Insight:** 'Woman's' category dominates review volume (43,000 reviews), significantly more than others. 'Handbags' boast the highest average rating (4.87). While 'Woman's' has the lowest average rating (4.56), this is likely due to its massive review count, demonstrating robust satisfaction across all categories.
""") 
st.markdown("---")

# NLP header 
st.title("2. Neural Language Processing (NLP)")
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
st.write("This comparison highlights a consistently positive customer sentiment across both directly assigned user ratings and AI-driven analysis. The overwhelming majority of reviews fall into the positive sentiment categories, reinforcing strong product satisfaction.However, AI score shows a decrease in positive rating, which might hint on a more nuanced sentiment subtly expressed by users that isn't fully captured in their direct numerical rating. ")
st.markdown("---")

# 2 Heatmap
st.header("2.2 BERT Model Agreement Insight")
fig = mismatch_heatmap(df_reviews)
fig = plot_preset(fig)
st.plotly_chart(fig, use_container_width=True)
st.write("""While 5-star reviews largely align with positive textual sentiment, significant divergence occurs elsewhere. 
    User 1-star reviews often reflect 2-star textual sentiment, and user 5-star reviews often reflect 4-star text. 
    This highlights that numerical ratings are often more polarized than the nuanced sentiment in written feedback, pinpointing areas for deeper analysis of customer experience.
          """)
st.markdown("---")

# 3. Mismatch comparison by category
st.header("2.3 Mismatch Analysis Between User Scores and BERT Sentiment by Category")
fig = mismatch_by_category(df_reviews)
fig = plot_preset(fig)
st.plotly_chart(fig, use_container_width=True)
st.write("""While 'Woman's' has the most mismatched reviews due to high volume, categories like 'Men's' show a proportionally higher rate of sentiment discrepancies. 
    This indicates that reviews in these categories may contain more nuanced or mixed sentiments, offering valuable areas for deeper textual analysis and product insight.
    """)
st.markdown("---")

# 4 Mismatch size feedback
st.header("2.4 Mismatch size")
fig = mismatch_size_feedback(df_reviews)
fig = plot_preset(fig)
st.plotly_chart(fig, use_container_width=True)
st.write("""Customers generally rate higher than the sentiment expressed in their text, especially for sizing issues like 'Big', 'Small', and 'Too Big'. 
    'Too Small' feedback consistently shows the lowest sentiment. This highlights that numerical ratings may mask dissatisfaction with product fit, indicating areas for sizing improvement.
    """)
st.markdown("---")

# 5 Review length
st.header("2.5 Review Length Insights")
col1, col2 = st.columns([3, 7])  # 30% and 70%

with col1:
    fig = review_length(df_reviews)
    fig = plot_preset(fig)
    st.plotly_chart(fig, use_container_width=True)
    st.write("""Most customer reviews are concise, with a median length of around 120 characters. 
    This indicates customers prefer providing brief, focused feedback, which is also beneficial for consistent NLP analysis.
 """)

with col2:
    fig = review_length_by_category(df_reviews)
    fig = plot_preset(fig)
    st.plotly_chart(fig, use_container_width=True)
    st.write("""Average review length differs by product category. Customers write longer reviews for 'Woman's' and 'Clothing' items, suggesting more detailed feedback. 
    Conversely, reviews for 'Handbags' and 'Accessories' are typically more concise. This insight helps prioritize where to seek deeper textual insights.
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
This plot provides a granular look at sentiment specifically by shoe type. 
'Boots', 'heels', 'sandals', and 'sneakers' exhibit strong positive review counts, indicating high satisfaction within these popular categories. 
Conversely, 'Flats' show a more mixed sentiment profile with noticeable negative and neutral counts relative to their positive reviews. 
This detailed breakdown helps pinpoint specific footwear categories for targeted product and service improvements.
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
st.write("""Customer sentiment is highly positive overall across product attributes like 'Style & Appearance' and 'Comfort'. However, specific areas like 'Delivery' and 'Size & Fit' show proportionally more neutral or negative feedback in general.

Analysis by shoe type reveals varying sentiment: 'Boots' and 'Sneakers' are very strong, while 'Flats' show a more mixed sentiment profile.

BERTopic analysis further uncovers key themes: most topics are overwhelmingly positive (e.g., General Product Praise, Footwear Comfort), but topics related to 'Discomfort & Foot Pain' or 'Foot Width & Fit Issues' show higher negative/neutral sentiment. This multi-layered insight pinpoints precise areas for product improvement and service optimization across your catalog.
""")
st.markdown("---")

st.header("Predictive Modeling: Sales Forecasting")

st.subheader("Understanding Key Drivers Behind Sales and Making Accurate Predictions")

st.info(
    "**Understanding Predictive Modeling:** "
    "Predictive modeling uses historical data to forecast future outcomes. "
    "Here, the model estimates expected sales based on product features, pricing, and other factors, "
    "helping to anticipate demand and make informed business decisions."
)
st.header("Model Performance: Actual vs. Predicted Sales")
col1, col2, col3 = st.columns(3)

with col1:
    # This function call will load your plot
    poly_ridge_prediction(poly_ridge_prediction_df)
    st.write("This chart illustrates the performance of the Polynomial Ridge Regression model by comparing its predicted sales against the actual sales figures.")
    st.markdown("**Insight:** The Polynomial Ridge model effectively captures both linear and complex non-linear relationships, offering detailed insights into how various factors, including their interactions, influence sales.")
    st.markdown("---")

with col2:
    # This function call will load your plot
    random_forest_prediction(random_forest_prediction_df)
    st.write("This chart displays the Random Forest Regressor's accuracy by showing the alignment between predicted and actual sales data.")
    st.markdown("**Insight:** The Random Forest model demonstrates robust predictive accuracy, with its predictions closely aligning with actual sales, indicating strong reliability for sales forecasting.")
    st.markdown("---")

with col3:
    # This function call will load your plot
    xgboost_prediction(xgboost_prediction_df)
    st.write("This chart visualizes the superior performance of the XGBoost Regressor, comparing its highly accurate sales predictions against actual values.")
    st.markdown("**Insight:** XGBoost exhibits the highest predictive accuracy among the models, providing exceptionally precise sales forecasts critical for strategic business decisions.")
    st.markdown("---")

st.header("Key Sales Drivers: Feature Importance Analysis")

left, center, right = st.columns([1, 2, 1])

with center:
    st.subheader("Polynomial Ridge Feature Contributions")
    # This function call will load your plot
    plot_poly_ridge_features(poly_ridge_features)
    st.write("This chart presents the coefficients and their impact on sales within the Polynomial Ridge model, highlighting non-linear and interaction effects.")
    st.markdown("**Insight:** Polynomial Ridge uniquely reveals that the influence of certain factors on sales is not always straightforward. For instance, the optimal price point or how discounts combine with holiday events can be crucial for maximizing sales.")
    st.markdown("---")

    st.subheader("Random Forest Feature Importance")
    # This function call will load your plot
    plot_random_forest_features(random_forest_features)
    st.write("This chart ranks features by their importance in the Random Forest model, indicating their contribution to sales prediction accuracy.")
    st.markdown("**Insight:** Random Forest underscores that average price, weekday, and holiday events are consistently the most significant factors driving sales, affirming their importance for targeted business strategies.")
    st.markdown("---")

    st.subheader("XGBoost Feature Importance")
    # This function call will load your plot
    plot_xg_boost_features(xgboost_features)
    st.write("This chart illustrates the relative importance of features in the XGBoost model, showing which variables contribute most to accurate sales forecasts.")
    st.markdown("**Insight:** XGBoost reinforces that holiday events, advertising status, and average price are top drivers. Leveraging these insights allows for optimized marketing campaigns, pricing, and promotional efforts.")
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
st.subheader("Comparative Model Performance Metrics") # Adjusted subheader
st.dataframe(metrics_df, height=200, width=400)
st.markdown("**Insight:** XGBoost demonstrates superior performance across all evaluation metrics (highest R² and lowest errors), making it the most reliable choice for sales forecasting within this analysis.")

st.write("---") # Separator for new section
st.header("Overall Business Insights from Predictive Modeling")
st.write("""
    Based on our predictive modeling analysis, several critical insights emerge for optimizing sales strategies:

    * **Data-Driven Forecasting with High Accuracy:** The **XGBoost model** stands out as highly accurate (R²=0.975), providing a robust and reliable tool for forecasting sales. This empowers businesses to move from guesswork to data-backed predictions for future planning, leading to more confident decision-making.

    * **Identified Key Sales Levers:** Our models consistently highlight that **pricing (`avg_price`, `discount_percent`), promotional activities (`ad_status`, `holiday_event`), customer satisfaction (`rating`), and timing (`weekday`)** are the most influential factors driving sales.
        * **Strategic Pricing & Promotions:** We observe that higher average prices generally lead to lower sales, while discounts significantly boost them. The analysis also suggests that the optimal price might not be a simple linear relationship, and that discounts are particularly effective when synergized with holiday events. This allows for highly dynamic and profitable pricing strategies that adapt to market conditions and special periods.
        * **Effective Marketing Channels:** Active advertising campaigns and leveraging holiday periods are strongly correlated with increased sales. This insight clearly indicates where marketing efforts and budget should be concentrated for maximum return on investment.
        * **Customer-Centric Approach Pays Off:** Product ratings directly influence sales, underscoring the paramount importance of prioritizing product quality and exceptional customer service to foster positive reviews and build brand loyalty.
        * **Optimized Operations Through Timing:** Understanding the distinct sales patterns throughout the week (`weekday`) allows businesses to optimize inventory levels, staffing arrangements, and logistics, ensuring resources are perfectly aligned with anticipated demand fluctuations.

    * **Unveiling Nuanced Relationships:** While powerful ensemble models like XGBoost provide excellent predictions, the Polynomial Ridge model offers unique value by revealing **complex non-linear effects and variable interactions**. This means that the impact of one factor might change significantly based on another (e.g., ad effectiveness during holidays), enabling the development of more sophisticated and synergistic business strategies that capture hidden opportunities.

    * **Actionable Intelligence for Growth:** These combined insights provide an actionable framework for comprehensive business decision-making, including streamlined inventory management, optimized marketing budget allocation, dynamic pricing adjustments, and overall efficient operational planning.

    **Important Note:** As previously stated, all sales data and related variables used in these models are simulated. However, the methodology and the types of actionable insights derived are directly applicable to real-world sales data once available, providing a strong foundation for a data-driven business strategy.
    
    **Interactive Sales Prediction in this Dashboard:**
    This dashboard is equipped to provide real-time sales predictions, allowing for immediate "what-if" scenario analysis. Users can interact with the prediction capabilities in two ways:
    1.  **Bulk Prediction (CSV Upload):** Upload a CSV file containing new product data (with features like price, rating, discount, etc.) to receive sales predictions for all entries at once. This is ideal for forecasting sales across a large product catalog or for future inventory planning.
    2.  **Single Prediction (Manual Input):** Manually input specific values for each sales driver (e.g., average price, whether it's a holiday, discount percentage) directly into the dashboard. This allows for quick, on-the-fly forecasts for individual products or hypothetical scenarios to assess the potential impact of changing specific variables.
     
""")
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

subcategory = st.selectbox("Subcategory", options=subcategory_options)

with st.form("input_form"):
    subcategory = st.selectbox("Subcategory", options=subcategory_options)
    avg_price = st.number_input("Average Price", min_value=0, step=1)
    rating = st.slider("Rating", min_value=1, max_value=5, step=1)
    month = st.slider("Month", min_value=1, max_value=4, step=1)
    weekday = st.slider("Weekday", min_value=0, max_value=6, step=1)
    discount_percent = st.slider("Discount Percent", min_value=1, max_value=100, step=1)
    ad_status = st.checkbox("Ad Status (True=1, False=0)")
    holiday_event = st.checkbox("Holiday Event (True=1, False=0)")

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