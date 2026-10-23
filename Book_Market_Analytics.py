import os
import re
import time
import warnings
from urllib.parse import urljoin

import requests
import pandas as pd
import matplotlib.pyplot as plt

from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")


# ============================================================
# 1. SETTINGS
# ============================================================

BASE_URL = "https://books.toscrape.com/"
TOTAL_PAGES = 50

OUTPUT_FOLDER = "book_market_outputs"
GRAPH_FOLDER = os.path.join(OUTPUT_FOLDER, "graphs")

os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(GRAPH_FOLDER, exist_ok=True)

RAW_FILE = os.path.join(OUTPUT_FOLDER, "../books_raw_dataset.csv")
CLEAN_FILE = os.path.join(OUTPUT_FOLDER, "../books_cleaned_dataset.csv")
PREDICTIONS_FILE = os.path.join(
    OUTPUT_FOLDER, "book_price_predictions.csv"
)

print("=" * 60)
print("BOOK MARKET ANALYTICS")
print("=" * 60)


# ============================================================
# 2. CREATE A SESSION WITH RETRIES
# ============================================================

session = requests.Session()

retry_strategy = Retry(
    total=3,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["GET"]
)

adapter = HTTPAdapter(max_retries=retry_strategy)

session.mount("https://", adapter)
session.mount("http://", adapter)

session.headers.update({
    "User-Agent": (
        "Mozilla/5.0 "
        "(Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 "
        "(KHTML, like Gecko) "
        "Chrome/130.0 Safari/537.36"
    )
})


def get_soup(url):

    try:
        response = session.get(url, timeout=20)
        response.raise_for_status()

        return BeautifulSoup(response.text, "html.parser")

    except requests.RequestException as error:
        print(f"Could not access {url}: {error}")
        return None


# ============================================================
# 3. HELPER FUNCTIONS
# ============================================================

def extract_rating(book_element):

    rating_element = book_element.select_one(".star-rating")

    if rating_element:
        rating_word = next(
            (
                class_name
                for class_name in rating_element.get("class", [])
                if class_name in [
                    "One", "Two", "Three", "Four", "Five"
                ]
            ),
            "Unknown"
        )
    else:
        rating_word = "Unknown"

    rating_numbers = {
        "One": 1,
        "Two": 2,
        "Three": 3,
        "Four": 4,
        "Five": 5
    }

    return rating_word, rating_numbers.get(rating_word)


def extract_price(price_text):

    if not price_text:
        return None

    match = re.search(r"(\d+(?:\.\d+)?)", price_text)

    if match:
        return float(match.group(1))

    return None


def extract_stock(stock_text):

    if not stock_text:
        return "Unknown", None

    stock_text = " ".join(stock_text.split())

    quantity_match = re.search(
        r"\((\d+)\s+available\)",
        stock_text,
        flags=re.IGNORECASE
    )

    quantity = (
        int(quantity_match.group(1))
        if quantity_match
        else None
    )

    if "in stock" in stock_text.lower():
        status = "In stock"
    elif "out of stock" in stock_text.lower():
        status = "Out of stock"
        quantity = 0
    else:
        status = "Unknown"

    return status, quantity


# ============================================================
# 4. SCRAPE THE BOOK LISTING PAGES
# ============================================================

print("\nSTEP 1: SCRAPING BOOK LISTINGS")
print("=" * 60)

books = []
failed_pages = []

for page_number in range(1, TOTAL_PAGES + 1):

    if page_number == 1:
        page_url = BASE_URL
    else:
        page_url = urljoin(
            BASE_URL,
            f"catalogue/page-{page_number}.html"
        )

    print(f"Scraping listing page {page_number}/{TOTAL_PAGES}")

    soup = get_soup(page_url)

    if soup is None:
        failed_pages.append(page_url)
        continue

    book_cards = soup.select("article.product_pod")

    for card in book_cards:

        title_element = card.select_one("h3 a")
        price_element = card.select_one(".price_color")
        stock_element = card.select_one(".availability")

        if title_element is None:
            continue

        title = title_element.get("title") or title_element.get_text(
            strip=True
        )

        relative_url = title_element.get("href", "")

        # Product links on listing pages are relative to /catalogue/
        product_url = urljoin(
            urljoin(BASE_URL, "catalogue/"),
            relative_url
        )

        price_text = (
            price_element.get_text(strip=True)
            if price_element
            else None
        )

        rating_word, rating_number = extract_rating(card)

        stock_text = (
            stock_element.get_text(" ", strip=True)
            if stock_element
            else None
        )

        stock_status, stock_quantity = extract_stock(stock_text)

        books.append({
            "Title": title,
            "Price_Text": price_text,
            "Rating_Word": rating_word,
            "Rating_Number": rating_number,
            "Stock_Text": stock_text,
            "Stock_Status": stock_status,
            "Stock_Quantity": stock_quantity,
            "Product_URL": product_url
        })

    time.sleep(0.1)


raw_df = pd.DataFrame(books)

if raw_df.empty:
    raise RuntimeError(
        "No books were scraped. Check your internet connection "
        "and whether the website is accessible."
    )

# Save the raw listing data before cleaning.
raw_df.to_csv(RAW_FILE, index=False, encoding="utf-8-sig")

print(f"\nBooks collected from listing pages: {len(raw_df)}")
print(f"Raw dataset saved to: {RAW_FILE}")

if failed_pages:
    print(f"Listing pages that failed: {len(failed_pages)}")


# ============================================================
# 5. SCRAPE ADDITIONAL PRODUCT DETAILS
# ============================================================

print("\nSTEP 2: COLLECTING PRODUCT DETAILS")
print("=" * 60)

product_details = []

for index, row in raw_df.iterrows():

    product_url = row["Product_URL"]

    print(
        f"Product details {index + 1}/{len(raw_df)}: "
        f"{row['Title']}"
    )

    soup = get_soup(product_url)

    if soup is None:
        product_details.append({
            "Product_URL": product_url,
            "Genre": "Unknown",
            "Description": "",
            "Product_Type": "Unknown",
            "Review_Count": None,
            "UPC": None
        })
        continue

    # Extract genre from the breadcrumb links.
    breadcrumb_links = soup.select(
        "ul.breadcrumb li a"
    )

    if len(breadcrumb_links) >= 3:
        genre = breadcrumb_links[-1].get_text(strip=True)
    else:
        genre = "Unknown"

    # Extract description.
    description_element = soup.select_one(
        "#product_description + p"
    )

    if description_element:
        description = description_element.get_text(
            " ", strip=True
        )
    else:
        description = ""

    # Extract product information from the table.
    product_info = {}

    for table_row in soup.select("table.table-striped tr"):

        heading = table_row.select_one("th")
        value = table_row.select_one("td")

        if heading and value:
            key = heading.get_text(strip=True)
            val = value.get_text(strip=True)
            product_info[key] = val

    product_type = product_info.get(
        "Product Type", "Unknown"
    )

    review_count_text = product_info.get(
        "Number of reviews", ""
    )

    review_match = re.search(
        r"\d+",
        review_count_text
    )

    review_count = (
        int(review_match.group())
        if review_match
        else None
    )

    upc = product_info.get("UPC")

    product_details.append({
        "Product_URL": product_url,
        "Genre": genre,
        "Description": description,
        "Product_Type": product_type,
        "Review_Count": review_count,
        "UPC": upc
    })

    time.sleep(0.1)


details_df = pd.DataFrame(product_details)

# Merge listing data with product detail data.
raw_df = raw_df.merge(
    details_df,
    on="Product_URL",
    how="left"
)

# Save the expanded raw dataset.
raw_df.to_csv(RAW_FILE, index=False, encoding="utf-8-sig")

print("\nProduct details collected.")
print(f"Total raw records: {len(raw_df)}")


# ============================================================
# 6. DATA CLEANING AND PREPARATION
# ============================================================

print("\nSTEP 3: CLEANING THE DATASET")
print("=" * 60)

clean_df = raw_df.copy()

# Remove duplicate products based on their URL.
clean_df = clean_df.drop_duplicates(
    subset=["Product_URL"]
).copy()

# Clean book titles and genre names.
clean_df["Title"] = (
    clean_df["Title"]
    .fillna("")
    .astype(str)
    .str.strip()
)

clean_df["Genre"] = (
    clean_df["Genre"]
    .fillna("Unknown")
    .astype(str)
    .str.strip()
)

clean_df["Description"] = (
    clean_df["Description"]
    .fillna("")
    .astype(str)
    .str.strip()
)

# Convert prices into numeric GBP values.
clean_df["Price_GBP"] = clean_df["Price_Text"].apply(
    extract_price
)

# Convert ratings into numeric values.
rating_mapping = {
    "One": 1,
    "Two": 2,
    "Three": 3,
    "Four": 4,
    "Five": 5
}

clean_df["Rating_Number"] = (
    clean_df["Rating_Word"]
    .map(rating_mapping)
)

# Clean stock quantities.
clean_df["Stock_Quantity"] = pd.to_numeric(
    clean_df["Stock_Quantity"],
    errors="coerce"
)

# Clean review counts.
clean_df["Review_Count"] = pd.to_numeric(
    clean_df["Review_Count"],
    errors="coerce"
)

# Create a description word count.
clean_df["Description_Word_Count"] = (
    clean_df["Description"]
    .apply(lambda text: len(text.split()))
)

# Prepare a simple searchable text field.
# This is derived from descriptions; it is not a set of
# official website tags.
clean_df["Description_Words"] = (
    clean_df["Description"]
    .str.lower()
    .str.replace(r"[^a-z0-9\s]", " ", regex=True)
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
)

# Record whether a description was available.
clean_df["Has_Description"] = (
    clean_df["Description"].str.len() > 0
)

# Replace inconsistent empty text values.
text_columns = [
    "Rating_Word",
    "Stock_Status",
    "Product_Type"
]

for column in text_columns:
    clean_df[column] = (
        clean_df[column]
        .fillna("Unknown")
        .astype(str)
        .str.strip()
    )

# Remove records without a usable price or rating.
clean_df = clean_df.dropna(
    subset=["Price_GBP", "Rating_Number"]
).copy()

# Keep realistic positive prices.
clean_df = clean_df[
    clean_df["Price_GBP"] > 0
].copy()

# Reset row numbering.
clean_df.reset_index(drop=True, inplace=True)

# Save the final cleaned dataset.
clean_df.to_csv(
    CLEAN_FILE,
    index=False,
    encoding="utf-8-sig"
)

print("\nCleaning completed.")
print(f"Clean dataset rows: {len(clean_df)}")
print(f"Clean dataset columns: {len(clean_df.columns)}")
print(f"Clean dataset saved to: {CLEAN_FILE}")

print("\nFirst five cleaned records:")
print(clean_df.head().to_string(index=False))

print("\nMissing values in cleaned dataset:")
print(clean_df.isnull().sum())

print("\nDataset information:")
clean_df.info()

if len(clean_df) < 200:
    print(
        "\nWARNING: Fewer than 200 usable books were collected. "
        "Check the failed requests and rerun the script."
    )


# ============================================================
# 7. DESCRIPTIVE STATISTICS
# ============================================================

print("\nSTEP 4: EXPLORATORY DATA ANALYSIS")
print("=" * 60)

print("\nPrice summary:")
print(clean_df["Price_GBP"].describe())

print("\nRating counts:")
print(
    clean_df["Rating_Word"]
    .value_counts()
)

print("\nBooks by genre:")
print(
    clean_df["Genre"]
    .value_counts()
)

print("\nAverage price by rating:")
print(
    clean_df.groupby("Rating_Word")["Price_GBP"]
    .mean()
    .sort_values(ascending=False)
)

print("\nAverage price by genre:")
print(
    clean_df.groupby("Genre")["Price_GBP"]
    .mean()
    .sort_values(ascending=False)
)

print("\nStock status counts:")
print(
    clean_df["Stock_Status"]
    .value_counts()
)

print("\nCorrelation between numeric variables:")
print(
    clean_df[
        [
            "Price_GBP",
            "Rating_Number",
            "Stock_Quantity",
            "Review_Count",
            "Description_Word_Count"
        ]
    ].corr(numeric_only=True)
)


# ============================================================
# 8. VISUALISATIONS
# ============================================================

print("\nSTEP 5: CREATING GRAPHS")
print("=" * 60)


# GRAPH 1: DISTRIBUTION OF BOOK PRICES
plt.figure(figsize=(10, 6))

plt.hist(
    clean_df["Price_GBP"].dropna(),
    bins=20,
    edgecolor="black"
)

plt.title("Distribution of Book Prices")
plt.xlabel("Price (£)")
plt.ylabel("Number of Books")
plt.tight_layout()

plt.savefig(
    os.path.join(GRAPH_FOLDER, "01_price_distribution.png"),
    dpi=300
)

plt.show()
plt.close()


# GRAPH 2: AVERAGE PRICE BY STAR RATING
rating_order = [
    "One", "Two", "Three", "Four", "Five"
]

rating_prices = (
    clean_df.groupby("Rating_Word")["Price_GBP"]
    .mean()
    .reindex(rating_order)
    .dropna()
)

plt.figure(figsize=(9, 6))

plt.bar(
    rating_prices.index,
    rating_prices.values,
    edgecolor="black"
)

plt.title("Average Book Price by Star Rating")
plt.xlabel("Star Rating")
plt.ylabel("Average Price (£)")
plt.tight_layout()

plt.savefig(
    os.path.join(GRAPH_FOLDER, "02_average_price_by_rating.png"),
    dpi=300
)

plt.show()
plt.close()


# GRAPH 3: NUMBER OF BOOKS BY GENRE
genre_counts = clean_df["Genre"].value_counts()

plt.figure(figsize=(12, 9))

genre_counts.sort_values().plot(
    kind="barh",
    edgecolor="black"
)

plt.title("Number of Books by Genre")
plt.xlabel("Number of Books")
plt.ylabel("Genre")
plt.tight_layout()

plt.savefig(
    os.path.join(GRAPH_FOLDER, "03_books_by_genre.png"),
    dpi=300
)

plt.show()
plt.close()


# GRAPH 4: AVERAGE PRICE BY GENRE
genre_prices = (
    clean_df.groupby("Genre")["Price_GBP"]
    .mean()
    .sort_values()
)

plt.figure(figsize=(12, 9))

genre_prices.plot(
    kind="barh",
    edgecolor="black"
)

plt.title("Average Book Price by Genre")
plt.xlabel("Average Price (£)")
plt.ylabel("Genre")
plt.tight_layout()

plt.savefig(
    os.path.join(GRAPH_FOLDER, "04_average_price_by_genre.png"),
    dpi=300
)

plt.show()
plt.close()


# GRAPH 5: STAR RATING AGAINST PRICE
plt.figure(figsize=(9, 6))

plt.scatter(
    clean_df["Rating_Number"],
    clean_df["Price_GBP"],
    alpha=0.7
)

plt.title("Relationship Between Star Rating and Price")
plt.xlabel("Star Rating (1–5)")
plt.ylabel("Price (£)")
plt.tight_layout()

plt.savefig(
    os.path.join(GRAPH_FOLDER, "05_rating_vs_price.png"),
    dpi=300
)

plt.show()
plt.close()


# GRAPH 6: STOCK AVAILABILITY
stock_counts = clean_df["Stock_Status"].value_counts()

plt.figure(figsize=(8, 6))

plt.bar(
    stock_counts.index,
    stock_counts.values,
    edgecolor="black"
)

plt.title("Book Stock Availability")
plt.xlabel("Stock Status")
plt.ylabel("Number of Books")
plt.tight_layout()

plt.savefig(
    os.path.join(GRAPH_FOLDER, "06_stock_availability.png"),
    dpi=300
)

plt.show()
plt.close()


# GRAPH 7: DESCRIPTION LENGTH AGAINST PRICE
plt.figure(figsize=(9, 6))

plt.scatter(
    clean_df["Description_Word_Count"],
    clean_df["Price_GBP"],
    alpha=0.7
)

plt.title("Description Length vs Book Price")
plt.xlabel("Description Word Count")
plt.ylabel("Price (£)")
plt.tight_layout()

plt.savefig(
    os.path.join(GRAPH_FOLDER, "07_description_length_vs_price.png"),
    dpi=300
)

plt.show()
plt.close()


# GRAPH 8: REVIEW COUNT AGAINST PRICE
review_df = clean_df.dropna(
    subset=["Review_Count", "Price_GBP"]
)

plt.figure(figsize=(9, 6))

plt.scatter(
    review_df["Review_Count"],
    review_df["Price_GBP"],
    alpha=0.7
)

plt.title("Number of Reviews vs Book Price")
plt.xlabel("Number of Reviews")
plt.ylabel("Price (£)")
plt.tight_layout()

plt.savefig(
    os.path.join(GRAPH_FOLDER, "08_reviews_vs_price.png"),
    dpi=300
)

plt.show()
plt.close()

print(f"\nGraphs saved in: {GRAPH_FOLDER}")


# ============================================================
# 9. LINEAR REGRESSION MODEL
# ============================================================

print("\nSTEP 6: BUILDING THE LINEAR REGRESSION MODEL")
print("=" * 60)

# Select features that may help estimate book prices.
# Price_GBP is the target variable.
numeric_features = [
    "Rating_Number",
    "Stock_Quantity",
    "Review_Count",
    "Description_Word_Count"
]

categorical_features = [
    "Genre",
    "Stock_Status"
]

feature_columns = numeric_features + categorical_features

model_df = clean_df[
    feature_columns + ["Price_GBP"]
].copy()

# Ensure numerical columns are numeric.
for column in numeric_features:
    model_df[column] = pd.to_numeric(
        model_df[column],
        errors="coerce"
    )

# A target price is required to train the model.
model_df = model_df.dropna(
    subset=["Price_GBP"]
).copy()

X = model_df[feature_columns]
y = model_df["Price_GBP"]

if len(model_df) < 10:
    raise ValueError(
        "Not enough usable records to build a reliable model. "
        "Collect more book records first."
    )

# Separate numeric and categorical processing.
numeric_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        )
    ]
)

categorical_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "onehot",
            OneHotEncoder(handle_unknown="ignore")
        )
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            numeric_transformer,
            numeric_features
        ),
        (
            "categorical",
            categorical_transformer,
            categorical_features
        )
    ]
)

# Create the complete modelling pipeline.
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression())
    ]
)

# Split the dataset into training and testing groups.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# Train the model.
model.fit(X_train, y_train)

# Predict prices for the test dataset.
y_pred = model.predict(X_test)

# Calculate evaluation metrics.
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

print("\nMODEL EVALUATION RESULTS")
print("-" * 40)
print(f"Training records: {len(X_train)}")
print(f"Testing records: {len(X_test)}")
print(f"Mean Absolute Error (MAE): £{mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): £{rmse:.2f}")
print(f"R-squared (R²): {r2:.3f}")


# ============================================================
# 10. PLOT ACTUAL VS PREDICTED PRICES
# ============================================================

plt.figure(figsize=(9, 6))

plt.scatter(
    y_test,
    y_pred,
    alpha=0.7
)

# Draw a reference line for perfect predictions.
minimum_value = min(y_test.min(), y_pred.min())
maximum_value = max(y_test.max(), y_pred.max())

plt.plot(
    [minimum_value, maximum_value],
    [minimum_value, maximum_value],
    linestyle="--"
)

plt.title("Actual Prices vs Predicted Prices")
plt.xlabel("Actual Price (£)")
plt.ylabel("Predicted Price (£)")
plt.tight_layout()

plt.savefig(
    os.path.join(GRAPH_FOLDER, "09_actual_vs_predicted.png"),
    dpi=300
)

plt.show()
plt.close()


# ============================================================
# 11. SAVE MODEL PREDICTIONS
# ============================================================

predictions_df = pd.DataFrame({
    "Actual_Price_GBP": y_test.to_numpy(),
    "Predicted_Price_GBP": y_pred,
    "Prediction_Error_GBP": y_test.to_numpy() - y_pred
})

predictions_df.to_csv(
    PREDICTIONS_FILE,
    index=False,
    encoding="utf-8-sig"
)

print(f"\nPredictions saved to: {PREDICTIONS_FILE}")

print("\nFirst ten predictions:")
print(predictions_df.head(10).round(2).to_string(index=False))


# ============================================================
# 12. AUTOMATIC SUMMARY FOR YOUR REPORT
# ============================================================

print("\nSTEP 7: PROJECT SUMMARY")
print("=" * 60)

print(f"Total cleaned books: {len(clean_df)}")
print(f"Number of genres: {clean_df['Genre'].nunique()}")

print(
    f"Average book price: "
    f"£{clean_df['Price_GBP'].mean():.2f}"
)

print(
    f"Median book price: "
    f"£{clean_df['Price_GBP'].median():.2f}"
)

print(
    f"Lowest book price: "
    f"£{clean_df['Price_GBP'].min():.2f}"
)

print(
    f"Highest book price: "
    f"£{clean_df['Price_GBP'].max():.2f}"
)

print(
    f"Average star rating: "
    f"{clean_df['Rating_Number'].mean():.2f}/5"
)

print(
    f"Most common genre: "
    f"{clean_df['Genre'].mode().iloc[0]}"
)

print(
    f"Mean Absolute Error: £{mae:.2f}"
)

print(
    f"R-squared score: {r2:.3f}"
)

print("\nPROJECT FILES")
print(f"Raw dataset: {RAW_FILE}")
print(f"Clean dataset: {CLEAN_FILE}")
print(f"Predictions: {PREDICTIONS_FILE}")
print(f"Graphs folder: {GRAPH_FOLDER}")

print("\nProject completed.")
print("=" * 60)