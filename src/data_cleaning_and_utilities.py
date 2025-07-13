import sqlite3
import pandas as pd
import numpy as np
from plotly.colors import get_colorscale

### Data cleaning and preparation
# We load raw product and variant data from an SQLite database. Then, we:
# - Replace placeholders ("N/A") with NaN
# - Drop rows with missing critical fields
# - Fix inconsistent category naming
# - Extract main product categories
# - Normalize numeric and text fields

# Load raw data
path = ("../data/product_data.db")
conn = sqlite3.connect(path)
df_main = pd.read_sql("SELECT * FROM product_main_data", conn)
df_variants = pd.read_sql("SELECT * FROM product_variants_data", conn)
df_reviews = pd.read_sql("SELECT * FROM product_review_data", conn)

# Replace "N/A" to NaN 
df_main.replace("N/A", np.nan, inplace= True)
df_variants.replace("N/A", np.nan, inplace= True)
df_reviews.replace("N/A", np.nan, inplace= True)

# Drop rows where there is NaN in category or Name
df_main.dropna(subset = ["category", "master_name"], inplace=True)
df_variants.dropna(subset = ["category", "master_name"], inplace = True)
df_reviews.dropna(subset = ["category", "master_name"], inplace = True)

# Remove unneeded size feedback(1-10 entires per value)
df_reviews = df_reviews[~df_reviews["size"].isin(["Good", "Satisfactory", "Very Good", "Weak"])]
df_reviews["review_length"] = df_reviews["text"].str.len()


# Ensure numerical categories are int
columns_to_convert = [
    "True to size (size_ag)", "Too Big (size_ag)", "Small (size_ag)", "Too Small (size_ag)", "Not specified (size_ag)",
    "Big (size_ag)", "Not specified (experience)", "Quality (experience)", "Stylish (experience)", "Comfortable (experience)",
    "Verified Buyer", "Verified Reviewer", "Unverified"
]

df_main[columns_to_convert] = df_main[columns_to_convert].astype(float)

# Ensure correct category name and formatting
fix_categories = {
    "Loafers-flats": "Womens-loafers-flats",
    "Boots-booties": "Womens-boots-booties",
    "Trends-slides": "Womens-slip-on-slides",
    "Trends-mules": "Womens-mules",
    "Mini-me": "Kids-mini-me",
    "Flower-girl-shoes": "Kids-flower-girl-shoes",
    "Bottoms": "Clothing-bottoms",
    "Dresses": "Clothing-dresses",
    "Tops-and-shirts" : "Clothing-tops-shirts",
    "Blazers": "Clothing-blazers",
    "Jackets-and-coats": "Clothing-jackets-coats",
    "Socks-tights": "Accessories-socks-tights",
    "Hats-gloves-and-scarves": "Accessories-hats-gloves-scarves"
    
}
df_main["category"] = df_main["category"].replace(fix_categories)
df_variants["category"] = df_variants["category"].replace(fix_categories)
df_reviews["category"] = df_reviews["category"].replace(fix_categories)
# Change review values to int
df_main["rating"] = df_main["rating"].replace(r" star rating", "", regex= True).astype("float")

# Ensure price values are in int and dont have currency symbols
df_variants["price"] = df_variants["price"].replace(r"From |[\$]", "", regex = True).astype("float")

# Remove bundles(several products from different categories combined)
df_main = df_main[df_main["category"] != "Clothing-matching-sets"]
df_reviews = df_reviews[df_reviews["category"] != "Clothing-matching-sets"]
rows_to_remove = df_variants["full_product_name"].str.contains("bundle", case=False, na=False)
df_variants = df_variants[~rows_to_remove]


# Add main category column
category_map = {
    # Womans
    "Womens-sandals": "Woman's",
    "Womens-heels": "Woman's",
    "Womens-sneakers": "Woman's",
    "Womens-loafers-flats": "Woman's",
    "Womens-boots-booties": "Woman's",
    "Womens-platforms": "Woman's",
    "Womens-mules": "Woman's",
    "Womens-wedges": "Woman's",
    "Womens-slip-on-slides": "Woman's",

    # Accessories
    "Accessories-hats-scarves": "Accessories",
    "Accessories-socks-tights": "Accessories",
    "Accessories-fashion-jewelry": "Accessories",
    "Accessories-sunglasses": "Accessories",
    "Accessories-shoe-care": "Accessories",
    "Accessories-hats-gloves-scarves": "Accessories",

    # Clothing
    "Clothing-jackets-coats": "Clothing",
    "Clothing-bottoms": "Clothing",
    "Clothing-tops-shirts": "Clothing",
    "Clothing-dresses-jumpsuits": "Clothing",
    "Clothing-matching-sets": "Clothing",
    "Clothing-faux-leather": "Clothing",
    "Clothing-blazers": "Clothing",
    "Clothing-dresses": "Clothing",

    # Handbags
    "Handbags-wallets-charms": "Handbags",
    "Handbags-belt-bags-backpacks": "Handbags",
    "Handbags-clutches-mini-bags": "Handbags",
    "Handbags-crossbody-bags": "Handbags",
    "Handbags-belt-bags": "Handbags",
    "Handbags-clutches": "Handbags",

    # Kids
    "Kids-sporty-chic": "Kids",
    "Kids-pretty-in-pink": "Kids",
    "Kids-special-occasion-shoes": "Kids",
    "Kids-mini-me": "Kids",
    "Kids-flower-girl-shoes": "Kids",
    "Kids-beach-days": "Kids",

    # Men
    "Mens-casual": "Men's",
    "Mens-sandals": "Men's",
    "Mens-loafers": "Men's",
    "Mens-boots": "Men's",
    "Mens-sneakers": "Men's",
    "Mens-dress-shoes": "Men's"
}
df_main["main_category"] = df_main["category"].map(category_map)
df_variants["main_category"] = df_variants["category"].map(category_map)
df_reviews["main_category"] = df_reviews["category"].map(category_map)
# Rename catgory column
df_main.rename(columns = {"category": "subcategory"}, inplace=True)
df_variants.rename(columns = {"category": "subcategory"}, inplace=True)
df_reviews.rename(columns = {"category": "subcategory"}, inplace=True)
# Get the index of the 'sub_category' column

subcat_index = df_main.columns.get_loc("subcategory")
# Insert the new column at that position
df_main.insert(subcat_index, "main_category", df_main.pop("main_category"))

# Repeat for df_variants dataframe
subcat_index = df_variants.columns.get_loc("subcategory")
df_variants.insert(subcat_index, "main_category", df_variants.pop("main_category"))

# Export for use in streamlit
output_path = "../data/cleaned_main.csv"
df_main.to_csv(output_path, index=False)
output_path = "../data/cleaned_variants.csv"
df_variants.to_csv(output_path, index=False)
output_path = "../data/cleaned_reviews.csv"
df_reviews.to_csv(output_path, index=False)


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
