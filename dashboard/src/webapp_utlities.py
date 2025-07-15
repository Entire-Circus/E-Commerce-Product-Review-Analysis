import pandas as pd
import numpy as np
from pathlib import Path
import plotly.io as pio
import plotly.graph_objects as go
import vizro.plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from pathlib import Path
from plotly.colors import get_colorscale

# Utility functions
def get_earth_colorscale():
    # Retrieve the 'Earth' colorscale and extract only the color values
    earth_colors = get_colorscale("Earth")
    earth_palette = [color for _, color in earth_colors]
    return earth_palette

def plot_preset(fig):
    # Apply consistent styling and layout to Plotly figures for a clean and readable aesthetic
    fig.update_layout(
        width=1000, 
        height=500,
        xaxis_title_font=dict(family="Arial", size=18, color="white"),
        yaxis_title_font=dict(family="Arial", size=18, color="white"),
        title={
            'x': 0.5,
            "y": 1,
            'xanchor': 'center',
            'font': {'size': 24, 'family': "Arial, sans-serif", 'color': 'white'}
        },
        margin=dict(l=50, r=50, t=50, b=50),
        bargap=0.1,  # Add slight spacing between bars for visual clarity
        yaxis=dict(
            tickfont=dict(size=14, color="white")  # Ensure tick labels are legible on dark backgrounds
        ),
    )
    return fig

# Load cleaned data
def load_data():
    base_path = Path(__file__).parent.parent / "data"

    df_main = pd.read_csv(base_path / "cleaned_main.csv")
    df_variants = pd.read_csv(base_path / "cleaned_variants.csv")
    df_reviews = pd.read_csv(base_path / "df_reviews_final.csv")
    mismatch_df = pd.read_csv(base_path / "BERT_mismatch_data.csv")
    aspects_df = pd.read_csv(base_path / "main_aspects.csv")
    shoe_aspects_df = pd.read_csv(base_path / "shoe_aspects.csv")
    topic_summary_df = pd.read_csv(base_path / "BERTopic.csv")
    poly_ridge_prediction_df = pd.read_csv(base_path / "prediction_poly_ridge.csv")
    random_forest_prediction_df = pd.read_csv(base_path / "prediction_random_forest.csv")
    xgboost_prediction_df = pd.read_csv(base_path / "prediction_xgboost.csv")
    poly_ridge_features = pd.read_csv(base_path / "poly_ridge_features.csv")
    random_forest_features = pd.read_csv(base_path / "random_forest_features.csv")
    xgboost_features = pd.read_csv(base_path / "xgboost_features.csv")

    return (
        df_main, df_variants, df_reviews, mismatch_df, aspects_df, shoe_aspects_df,
        topic_summary_df, poly_ridge_prediction_df, random_forest_prediction_df,
        xgboost_prediction_df, poly_ridge_features, random_forest_features, xgboost_features
    )

def plot_products_by_subcategory(df):
    # Prepare and plot the bar chart
    df = df.groupby("subcategory")["master_name"].count().reset_index()
    df = df.sort_values(by="master_name", ascending=False)

    fig = px.bar(df, y="subcategory", x="master_name", color="master_name", color_continuous_scale="Earth",
        title="Number of Unique Products by Subcategory"
    )
    fig.update_layout(
        xaxis_title="Product Amount",
        yaxis_title="Subcategory Name",
    )

    return fig

def plot_distribuion_of_reviews_by_rating(df):
    # Aggregate total number of reviews for each rating value
    df = df.groupby("rating")["review_amount"].sum().reset_index()
    # Sort by rating in ascending order for better visual flow
    df = df.sort_values(by= "rating", ascending= True)
    # Create bar chart showing total review count per rating
    fig = px.bar(df, y = "review_amount", x = "rating", color= "review_amount", color_continuous_scale="Earth",
                  title = "Distribution of Reviews by Average Rating")
    # Update axis labels
    fig.update_layout(
        xaxis_title="Rating",
        yaxis_title="Number of Reviews",
    )
    return fig

def plot_top10_most_review_products(df):
    # Add name_adnd_subcategory column 
    df["name_and_subcategory"] = df["master_name"] + " " + df["subcategory"]
    # Select top 10 products by total review count, ensuring unique product names
    df_top10 = df.sort_values("review_amount", ascending=False).drop_duplicates(subset=["master_name"]).iloc[:10]

    # Initialize a Plotly figure for the lollipop chart visualization
    fig = go.Figure()

    # Add scatter trace combining markers and lines to create the lollipop effect
    fig.add_trace(go.Scatter(
        x=df_top10["review_amount"],
        y=df_top10["name_and_subcategory"],
        mode="markers+lines",
        marker=dict(size=12),
        name="Reviews"
    ))

    fig.update_layout(title="Top 10 Most Reviewed Products",
                    xaxis_title="Review Count", yaxis_title="Product")
    return fig

def distribution_of_size_related_feedback(df):
    # Aggregate total counts for all size-related feedback categories
    df_size = df[[col for col in df.columns if "(size_ag)" in col]].sum().reset_index()
    # Apply custom order to size dataframe
    size_order = [
        "True to size (size_ag)",
        "Too Big (size_ag)",
        "Big (size_ag)",
        "Too Small (size_ag)",
        "Small (size_ag)",
        "Not specified (size_ag)"
    ]
    df_size["index"] = pd.Categorical(df_size["index"], categories=size_order, ordered=True)
    df_size = df_size.sort_values("index")
    # Plot distribution of size-related feedback on a logarithmic scale to handle wide value range
    fig = px.bar(df_size, x = "index", y = 0, log_y=True, color = 0, color_continuous_scale= "Earth", title = "Distribution of Size-Related Feedback<br>(Log scale)")
    fig.update_layout(
        xaxis_title="Size Feedback Category",
        yaxis_title="Number of Responses",
        yaxis_tickvals = ["1000", "1500", "2500", "3500", "5000", "7000", "10000"],
        yaxis_ticktext = ["1000", "1500", "2500", "3500", "5000", "7000", "10000"],
        width=800, 
        height=500,    
    )
    fig.update_coloraxes(colorbar_title_text="")
    return fig

def distribution_of_experience_related_feedback(df):
    # Aggregate total counts for all experience-related feedback categories
    df_exp = df[[col for col in df.columns if "(experience)" in col]].sum().reset_index()

    fig = px.bar(df_exp, x = "index", y = 0, log_y=True, color = 0, color_continuous_scale= "Earth", title = "Distribution of Experience-Related Feedback<br>(Log scale)")
    fig.update_layout(
        xaxis_title="Experience Feedback Category",
        yaxis_title="Number of Responses",
        yaxis_tickvals = ["1000", "2000", "3500", "5000", "8000", "12000" ,"18000"],
        yaxis_ticktext = ["1000", "2000", "3500", "5000", "8000", "12000", "18000"],
        width=800, 
        height=500,  
    )
    fig.update_coloraxes(colorbar_title_text="")
    return fig

def normalized_size_feedback_by_product_category(df):
    # Select size-related feedback columns and group by main [roduct category
    df = df[["main_category"] + [col for col in df.columns if "(size_ag)" in col ]]

    # Aggregate feedback counts by main category, summing responses per size feedback type
    df = df.groupby("main_category").agg({
        "True to size (size_ag)": "sum",
        "Too Big (size_ag)" : "sum",
        "Small (size_ag)" : "sum",
        "Too Small (size_ag)" : "sum",
        "Not specified (size_ag)" : "sum",
        "Big (size_ag)" : "sum"
    }).reset_index()

    # Normalize feedback counts to percentages within each main category for relative comparison
    df_wc = df.drop(["main_category"], axis =1)
    df_sum = df_wc.sum(axis=1)
    df_wc = df_wc.div(df_sum, axis=0) * 100
    df_wc["main_category"] = df["main_category"]
    df = df_wc

    # Convert data to long format for easier plotting of grouped bar chart
    df = df.melt(id_vars = "main_category", value_vars= ["True to size (size_ag)", "Too Big (size_ag)",	"Small (size_ag)", "Too Small (size_ag)", "Not specified (size_ag)", "Big (size_ag)"],
                var_name = "Size", value_name= "Count")

    # Define order of size feedback categories and sort accordingly for consistent plot appearance
    order = ["True to size (size_ag)", "Too Big (size_ag)", "Big (size_ag)", "Too Small (size_ag)", "Small (size_ag)", "Not specified (size_ag)"]
    df["Size"] = pd.Categorical(df["Size"], categories=order, ordered=True)
    df = df.sort_values(by=["main_category", "Size"])

    # Get colorscale for plot
    earth_palette = get_earth_colorscale()

    # Create grouped bar chart showing normalized size feedback percentages by product category
    fig = px.bar(df, x = "main_category", y = "Count", color = "Size", barmode="group", color_discrete_sequence=earth_palette, title = "Normalized Size Feedback by Product Category")
    # Apply formatting
    fig.update_layout(
        width = 1500,
        xaxis_title = "Product Category",
        yaxis_title = "Feedback Share (%)")
    return fig

def average_rating_by_category(df):
    # Aggregate total review counts and average ratings by main product category
    df = df[["main_category", "master_name", "review_amount", "rating"]]
    df = df.groupby("main_category").agg({
        "review_amount": "sum",     # Sum total reviews per category
        "rating": "mean"            # Calculate average rating per category
    }).reset_index()

    # Sort categories by average rating descending for better visual ranking
    df = df.sort_values(by = "rating", ascending=False)

    earth_palette = get_earth_colorscale()

    # Create horizontal bar chart: total reviewss with average rating labeled
    fig = px.bar(df, x="review_amount", y="main_category", color="main_category", orientation="h", text="rating",  title="Average Rating by Category (Log Scale for Review Amount)",
        color_discrete_sequence=earth_palette, hover_data=["review_amount"]
    )

    # Configure log scale on x-axis with custom ticks and labels for readability
    fig.update_layout(
        xaxis_title = "Number of Reviews",
        yaxis_title = "Product Category",
        xaxis_type="log",
        xaxis_tickvals=[10, 100, 1000, 10000, 40000],
        xaxis_ticktext=["10", "100", "1K", "10K", "40K"],
        uniformtext_minsize=8,
        uniformtext_mode="hide",
        showlegend = False
    )

    # Display average rating on bars with two decimal precision outside bars
    fig.update_traces(
        texttemplate="%{text:.2f}",  
        textposition="outside"       
    )
    return fig

def total_variants_and_products(df):
    # Select relevant columns and group by main category
    # Calculate number of unique master products and total product variants per category
    df = df[["main_category", "master_name", "full_product_name"]]
    # Rename columns for clarity
    df = df.rename(columns={
        "master_name" : "Unique products",
        "full_product_name" : "Total variants"
    }
    )
    df = df.groupby("main_category").agg({
        "Unique products" : pd.Series.nunique, # count unique master products
        "Total variants" : "count" # count total variants
    }).reset_index()

    # Transform data to long format for grouped bar plotting
    df = df.melt(id_vars="main_category", value_vars=["Unique products", "Total variants"], var_name="type", value_name="count")

    earth_palette = get_earth_colorscale()

    # Create grouped bar chart comparing unique products and variants by category (log scale)
    fig = px.bar(df, x = "main_category", y ="count", log_y=True, color = "type",  color_discrete_sequence=earth_palette, barmode= "group",
                title= "Unique Products vs Total Variants by Category")
    fig.update_layout(
        xaxis_title = "Category name",
        yaxis_title = "Product Amount",
        yaxis_tickvals = ["50", "75", "100", "150", "225", "350", "500", "700", "1000", "1500", "2200"]
    )
    return fig

def total_variants_and_products_dumbell(df):
    # Group and aggregate data to get the number of unique master products and total variants per category
    df_dumbbell = df.groupby("main_category").agg({
        "master_name": pd.Series.nunique,         # Count of distinct master products
        "full_product_name": "count"              # Total number of product variants
    }).reset_index().rename(columns={
        "master_name": "Unique Products",
        "full_product_name": "Total Variants"
    })

    # Sort categories by total variant count for cleaner visualization
    df_dumbbell = df_dumbbell.sort_values("Total Variants")

    # Initialize Plotly figure
    fig = go.Figure()

    # Add connecting lines between unique product and total variant counts (the "dumbbells")
    for i, row in df_dumbbell.iterrows():
        fig.add_trace(go.Scatter(
            x=[row["Unique Products"], row["Total Variants"]],        # Line from left (unique) to right (total)
            y=[row["main_category"], row["main_category"]],
            mode="lines",
            line=dict(color="gray", width=3),
            showlegend=False
        ))

    # Add left-side markers for unique product counts
    fig.add_trace(go.Scatter(
        x=df_dumbbell["Unique Products"],
        y=df_dumbbell["main_category"],
        mode="markers",
        marker=dict(color="blue", size=10),
        name="Unique Products"
    ))

    # Add right-side markers for total variant counts
    fig.add_trace(go.Scatter(
        x=df_dumbbell["Total Variants"],
        y=df_dumbbell["main_category"],
        mode="markers",
        marker=dict(color="orange", size=10),
        name="Total Variants"
    ))

    fig.update_layout(
        title="Dumbbell Chart: Unique Products vs Total Variants by Category",
        xaxis_title="Product Amount",
        yaxis_title="Category",
        xaxis_tickvals = [0, 250, 500, 1000, 1500, 2000, 2500],
        height=600
    )
    return fig

def avg_price_by_product_category(df):
    # Calculate mean price for each main product category
    df = df[["main_category", "price"]]
    df = df.groupby("main_category")["price"].mean().reset_index()
    # Sort categories by descending average price for clearer visualization
    df = df.sort_values("price", ascending=False).reset_index(drop=True)

    earth_palette = get_earth_colorscale()

    # Create horizontal bar chart of average prices by category
    fig = px.bar(df, x = "price", y ="main_category", color="main_category", color_discrete_sequence=earth_palette, title = "Average Price by Product Category")

    fig.update_layout(
        xaxis_title = "Average Price (USD)",
        yaxis_title = "Product Category",
        yaxis={"categoryorder":"total descending"},
    )
    # Manually add price labels outside bars for clarity
    price_labels = ["116.65", "91.67", "76.5", "73.09", "48.02", "23.25"]
    for i, trace in enumerate(fig.data):
        trace.text = price_labels[i]
        trace.textposition = "outside"
    return fig

def price_distribution_by_category(df):
    # Create violin plot showing price distribution across product categories
    fig = px.violin(df, y="main_category", x="price", color="main_category", box=True,
        points="outliers",  # Display only outlier points for clarity
        title="Violin Plot: Price Distribution by Category",
        hover_data={"price": ":.2f", "main_category": True},  # Format price hover info
        color_discrete_sequence=px.colors.qualitative.Set2  # Set consistent color scheme
    )

    # Refine marker appearance and point distribution
    fig.update_traces(
        jitter=0.2,  # Moderate horizontal spread of points
        pointpos=0.0,  # Center points within violins
        marker=dict(size=3, opacity=0.7)  # Smaller, semi-transparent points
    )

    # Customize layout for clarity and visual appeal
    fig.update_layout(
        xaxis_title="Price ($)",
        yaxis_title="Product Category",
        title_x=0.5,  # Center plot title
        font=dict(family="Arial", size=12, color="white")
    )
    return fig

def price_distribution_by_category_binned(df):
    # Categorize product prices into defined ranges for segmentation analysis
    df = df[["main_category", "price"]].copy()
    bins = [0, 15, 30, 50, 75, 100, float("inf")]
    labels = ["<15", "15-30", "30-50", "50-75", "75-100", ">100"]
    df["price_range"] = pd.cut(df["price"], bins=bins, labels=labels, right=False)

    # Calculate count of products in each price range within each main category
    df = df.groupby(["main_category", "price_range"], observed = True).size().reset_index(name="count")
    # Visualize price distribution across categories using a grouped bar chart with log scale

    earth_palette = get_earth_colorscale()

    fig = px.bar(df, x = "main_category", y = "count", log_y=True, color = "price_range", color_discrete_sequence=earth_palette, barmode="group", title="Product Count Distribution Across Price Bins by Category (Log Scale)")

    fig.update_layout(
        xaxis_title = "Product Category",
        yaxis_title = "Product Amount",
        yaxis_tickvals = ["1", "3", "10", "25", "50", "100", "200", "500", "1000"]
    )
    return fig

def mismatch_comparison(df):
    earth_palette = get_earth_colorscale()
    fig = px.bar(df, x="Score", y=["Total review amount", "Mismatched review amount"], barmode="group", labels={"variable": "Metric"}, 
                title="Mismatch Metrics by Review Score(Log scale)", log_y = True, color_discrete_sequence=earth_palette,)
    fig.update_layout(
    yaxis_title = "Review Amount",
    yaxis_tickvals = ["250", "500", "1000", "1700", "3000", "5500", "10000"],
    yaxis_ticktext = ["250", "500", "1000", "1700", "3000", "5500", "10000"]
    )
    return fig

def user_pie(df):
    user_rating = df["score"].value_counts().reset_index()
    user_rating.columns = ["Score", "Count"]
    earth_palette = get_earth_colorscale()
    # Visualize BERT sentiment distribution across rating with pie chart
    fig = px.pie(user_rating , values="Count", names="Score", title="Users Score Distribution", color_discrete_sequence=earth_palette)
    # Adjust legend position
    fig.update_layout(
        legend=dict(
            orientation="h",            
            yanchor="bottom", y=-0.2,   
            xanchor="center", x=0.5     
        )
    )
    return fig

def bert_pie(df):
    bert_rating = df["bert_sentiment"].value_counts().reset_index()
    bert_rating.columns = ["Score", "Count"]
    earth_palette = get_earth_colorscale()
    # Visualize BERT sentiment distribution across rating with pie chart
    fig = px.pie(bert_rating , values="Count", names="Score", title="BERT Score Distribution", color_discrete_sequence=earth_palette)
    # Adjust legend position
    fig.update_layout(
        legend=dict(
            orientation="h",            
            yanchor="bottom", y=-0.2,   
            xanchor="center", x=0.5     
        )
    )
    return fig

def mismatch_by_category(df):
    # Select relevant columns and create a copy for mismatch analysis by category
    dfc = df[["main_category", "review_id", "score", "bert_sentiment"]].copy()

    # Flag mismatches between user rating and BERT sentiment
    dfc["mismatches"] = dfc["score"] != dfc["bert_sentiment"]

    # Drop raw score columns to focus on counts and mismatches
    dfc = dfc.drop(["score", "bert_sentiment"], axis=1)
    dfc = dfc.rename(columns = {"review_id" : "Review Amount", "mismatches" : "Mismatches"})
    # Group by product category and aggregate total reviews and mismatch counts
    dfc = dfc.groupby("main_category").agg({
        "Review Amount": "count",
        "Mismatches": "sum"
    }).reset_index()

    earth_palette = get_earth_colorscale()

    # Plot grouped bar chart of total reviews vs mismatches per product category
    fig = px.bar(dfc, x="main_category", y=["Review Amount", "Mismatches"], barmode="group", title="Review vs Mismatch Count by Product Category(Log scale for Review Amount)", log_y=True,
        color_discrete_sequence=earth_palette
    )

    # Update axis labels and customize tick values for clarity
    fig.update_layout(
        xaxis_title="Product Category",
        yaxis_title="Review Amount",
        yaxis_tickvals=["30", "100", "250", "500", "1000", "2000", "5000", "10000"],
        yaxis_ticktext=["30", "100", "250", "500", "1000", "2000", "5000", "10000"]
    )

    return fig

def mismatch_heatmap(df):
    # Create confusion matrix
    confusion = pd.crosstab(df["score"], df["bert_sentiment"])
    # Apply log scaling for better color distribution
    confusion_log = np.log1p(confusion)

    # Plot the heatmap
    fig = px.imshow(
        confusion_log,
        labels=dict(x="BERT Star Rating", y="User Star Rating"),
        x=confusion.columns.astype(str),
        y=confusion.index.astype(str),
        color_continuous_scale="Viridis",
    )

    # Add raw counts as annotations
    fig.update_traces(
        text=confusion.values,
        texttemplate="%{text}",
        textfont={"size": 12, "color": "black"}
    )
    fig.update_layout(
        title={
            "text": "User Ratings vs BERT Predicted Ratings",
            "x": 0.45,  
            "xanchor": "center"
        }
    )
    return fig

def mismatch_size_feedback(df):
    # Calculate the average user rating by size feedback category
    size_users = df.groupby("size")["score"].mean().reset_index()

    # Calculate the average BERT-predicted sentiment by size feedback category
    size_bert = df.groupby("size")["bert_sentiment"].mean().reset_index()

    # Merge both sets of averages into one DataFrame
    size_analisys = pd.merge(size_users, size_bert, on="size", how="right")

    # Rename columns for clarity
    size_analisys.columns = ["Size Feedback", "User Score", "BERT Score"]

    # Calculate the difference (drift) between user score and BERT sentiment
    size_analisys["Drift (Score - Sentiment)"] = size_analisys["User Score"] - size_analisys["BERT Score"]

    # Reshape the DataFrame for grouped bar plotting
    size_analisys = size_analisys.melt(
        id_vars="Size Feedback",
        value_vars=["User Score", "BERT Score"],
        var_name="Metric",
        value_name="Value"
    )
    earth_palette = get_earth_colorscale()
    # Plot average scores by size feedback
    fig = px.bar(
        size_analisys,
        x="Size Feedback",
        y="Value",
        color="Metric",
        barmode="group",
        color_discrete_sequence=earth_palette,
        title="Users Rate Higher Than They Feel: Size Feedback vs. Review Sentiment"
    )
    fig.update_layout(
        xaxis_title = "Metric",
        yaxis_title = "Average Score"
    )
    return fig

def review_length(df):
    # Aggregate average review length, BERT sentiment, user score, and count by size feedback
    dft = df.groupby("size").agg({
        "review_length": "mean",
        "bert_sentiment": "mean",
        "score": "mean",
        "review_id": "count"
    }).reset_index()

    # Plot review length distribution across size categories
    fig = px.box(dft, y="review_length",  color_discrete_sequence=["#F1B555"], title = "Review Length")
    fig.update_layout(
        yaxis_title="Review Length",
        width=400
    )
    return fig

def review_length_by_category(df):
    # Group by product category and compute total review length and number of reviews
    dfc = df[["main_category", "review_length", "review_id"]]
    dfc = dfc.groupby("main_category").agg({
        "review_length": "sum",
        "review_id": "count"
    }).reset_index()

    # Calculate average review length per review
    dfc["avg_review_length"] = round(dfc["review_length"] / dfc["review_id"], 2)

    # Sort by average length for better visual ranking
    dfc = dfc.sort_values("avg_review_length", ascending=True)

    earth_palette= get_earth_colorscale()
    # Plot horizontal bar chart of average review length by category
    fig = px.bar(
        dfc,
        x="avg_review_length",
        y="main_category",
        orientation="h",
        text="avg_review_length",
        color="main_category",
        title="Average Review Length by Product Category",
        color_discrete_sequence=earth_palette
    )

    fig.update_traces(textposition="outside")
    fig.update_layout(
        xaxis_title="Avg. Review Length (Characters)",
        yaxis_title="Product Category",
        showlegend=False,
        width=1100
    )
    return fig

def aspects_raw(df):
    # Group the exploded review data by aspect and sentiment, count occurrences, pivot to wide format
    df_aspects_raw = df.melt(id_vars="detected_aspects", value_vars=["Negative", "Neutral", "Positive"], var_name="Aspects", value_name="Count")
    earth_palette = get_earth_colorscale()
    # Create a bar chart showing sentiment count per aspect (log-scaled Y axis for readability)
    fig = px.bar(df_aspects_raw, x="detected_aspects", y="Count", color="Aspects", log_y=True,color_discrete_sequence=earth_palette,
        barmode="group", title="Aspect-Level Breakdown of Sentiment in Reviews"
    )

    # Update chart layout for clarity and consistent scaling
    fig.update_layout(
        xaxis_title="Product Aspect Category",
        yaxis_title="Review Count (Log Scale)",
        yaxis_tickvals=[0, 5, 10, 20, 50, 100, 250, 500, 1000, 1700, 3000, 5000, 8000]
    )
    return fig

def aspects_percentage(df):
    df_aspects_percentages = df.melt(id_vars="detected_aspects", value_vars=["Negative_%", "Neutral_%", "Positive_%"], var_name="Aspects", value_name="Percentage")
    earth_palette = get_earth_colorscale()
    # Create a bar chart showing sentiment percentage per aspec
    fig = px.bar(df_aspects_percentages, x="detected_aspects", y="Percentage", color="Aspects", color_discrete_sequence=earth_palette,
    barmode="group", title="Aspect-Level Breakdown of Sentiment in Reviews")

    fig.update_layout(
        xaxis_title="Product Aspect Category",
        yaxis_title="Review Percentage"
    )
    return fig

def aspects_shoe(df):
    # Convert absolute sentiment counts and sentiment percentages to long format for plotting
    earth_palette = get_earth_colorscale()
    # Create a bar chart showing sentiment count per aspect (log-scaled Y axis for readability)
    fig = px.bar(df, x="detected_shoe_type", y="Count", color="Aspects",log_y=True,
        color_discrete_sequence=earth_palette, barmode="group", title="Aspect-Level Breakdown of Sentiment in Reviews"
    )

    # Update chart layout for clarity and consistent scaling
    fig.update_layout(
        xaxis_title="Product Aspect Category",
        yaxis_title="Review Count (Log Scale)",
        yaxis_tickvals=[0, 2, 5, 10, 25, 50, 100, 250, 500, 1000]
    )

    return fig

def bertopic_plot(df):
    # Select top 10 topics for plotting
    df = df.head(10)

    earth_palette = get_earth_colorscale()
    # Create grouped bar chart of sentiment percentages
    fig = px.bar(df, x="Topic Name", y=["Positive %", "Neutral %", "Negative %"], barmode="group",
                color_discrete_sequence=earth_palette, title="BERTopic Sentiment Analysis")
    return fig

def poly_ridge_prediction(df):
    fig, ax = plt.subplots(figsize=(8, 8))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#1e1e1e")

    sns.scatterplot(x=df["Actual Sales"], y=df["Predicted Sales"], ax=ax)
    ax.plot([df["Actual Sales"].min(), df["Actual Sales"].max()],
            [df["Actual Sales"].min(), df["Actual Sales"].max()],
            "r--", label="Perfect Prediction (y=x)")

    ax.set_xlabel("Actual Sales", color="#f0f0f0")
    ax.set_ylabel("Predicted Sales", color="#f0f0f0")
    ax.set_title("Poly Ridge Regression: Actual vs Predicted Sales", color="#f0f0f0")
    ax.tick_params(colors="#f0f0f0")
    ax.grid(True, color="#444444")
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig)

def random_forest_prediction(df):
    fig, ax = plt.subplots(figsize=(6, 6))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#1e1e1e")

    sns.scatterplot(data=df, x="Actual Sales", y="Predicted Sales", ax=ax)
    ax.plot([df["Actual Sales"].min(), df["Actual Sales"].max()],
            [df["Actual Sales"].min(), df["Actual Sales"].max()],
            "r--", label="Perfect Prediction")

    ax.set_xlabel("Actual Sales", color="#f0f0f0")
    ax.set_ylabel("Predicted Sales", color="#f0f0f0")
    ax.set_title("Random Forest: Actual vs Predicted Sales", color="#f0f0f0")
    ax.tick_params(colors="#f0f0f0")
    ax.grid(True, color="#444444")
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig)

def xgboost_prediction(df):
    fig, ax = plt.subplots(figsize=(8, 8))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#1e1e1e")

    sns.scatterplot(data=df, x="Actual Sales", y="Predicted Sales", ax=ax)
    ax.plot([df["Actual Sales"].min(), df["Actual Sales"].max()],
            [df["Actual Sales"].min(), df["Actual Sales"].max()],
            "r--", label="Perfect Prediction")

    ax.set_xlabel("Actual Sales", color="#f0f0f0")
    ax.set_ylabel("Predicted Sales", color="#f0f0f0")
    ax.set_title("XGBoost with Native Categorical: Actual vs Predicted Sales", color="#f0f0f0")
    ax.tick_params(colors="#f0f0f0")
    ax.grid(True, color="#444444")
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig)

def plot_poly_ridge_features(df):
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#1e1e1e")

    ax.barh(df['Feature'], df['Coefficient'], color='skyblue')
    ax.set_xlabel("Model Coefficient", color="#f0f0f0")
    ax.set_title("Feature Importances (Poly Ridge Regression)", color="#f0f0f0")
    ax.tick_params(colors="#f0f0f0")
    ax.invert_yaxis()
    ax.grid(True, color="#444444")
    fig.tight_layout()
    st.pyplot(fig)

def plot_random_forest_features(df):
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#1e1e1e")

    ax.barh(df['feature'], df['importance'], color='skyblue')
    ax.set_title("Feature Importances (Random Forest)", color="#f0f0f0")
    ax.set_xlabel("Importance", color="#f0f0f0")
    ax.set_ylabel("Feature", color="#f0f0f0")
    ax.tick_params(colors="#f0f0f0")
    ax.invert_yaxis()
    ax.grid(True, color="#444444")
    fig.tight_layout()
    st.pyplot(fig)

def plot_xg_boost_features(df):
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#1e1e1e")

    ax.barh(df['feature'], df['importance'], color='skyblue')
    ax.set_xscale('log')
    ax.set_xticks([100, 250, 500, 1000, 2000, 5000, 10000, 25000, 50000, 100000, 200000, 400000, 700000])
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())

    ax.set_xlabel("Feature Importance", color="#f0f0f0")
    ax.set_ylabel("Feature Name", color="#f0f0f0")
    ax.set_title("Feature Importances (XGBoost)", color="#f0f0f0")
    ax.tick_params(colors="#f0f0f0")
    ax.invert_yaxis()
    ax.grid(True, color="#444444")
    fig.tight_layout()
    st.pyplot(fig)