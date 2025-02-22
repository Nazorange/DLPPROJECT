# H&M Fashion Recommender System

## Overview

This project focuses on building a fashion recommendation system for H&M, leveraging customer purchase history, product information, and image embeddings to provide personalized recommendations. The system uses a two-stage approach: **retrieval** and **ranking**, to efficiently find and rank relevant fashion items for each customer. The project involves data preprocessing, feature engineering, model training, and evaluation to ensure high-quality recommendations.

## Table of Contents

1. [Introduction](#introduction)
2. [Dataset Description](#dataset-description)
3. [Methods](#methods)
   - [Step 1: Data Loading and Initial Processing](#step-1-data-loading-and-initial-processing)
   - [Step 1.2: Image Embedding](#step-12-image-embedding)
   - [Step 2: Data Splitting](#step-2-data-splitting)
   - [Step 3: Feature Engineering](#step-3-feature-engineering)
   - [Step 4: Model Data Preparation](#step-4-model-data-preparation)
   - [Step 5: Model Architecture](#step-5-model-architecture)
   - [Step 6: Inference](#step-6-inference)
   - [Step 7: Evaluation](#step-7-evaluation)
4. [Results and Evaluation](#results-and-evaluation)

---

## Introduction

The goal of this project is to build a recommendation system for H&M that provides personalized fashion recommendations to customers based on their purchase history, product metadata, and visual features of the items. The system uses a **two-tower neural network architecture** to map customers and articles into a shared embedding space, allowing for efficient retrieval and ranking of relevant items.

The project involves:
- **Data preprocessing** to handle large datasets and ensure consistency.
- **Feature engineering** to create meaningful features for customers, articles, and their interactions.
- **Model training** using a two-stage approach (retrieval and ranking).
- **Evaluation** of the system using metrics like Precision@K, Recall@K, and NDCG@K.

---

## Dataset Description

The dataset consists of three main components:

1. **Articles (Products):**
   - Contains detailed metadata for each fashion item, including product categories, color information, and visual attributes.
   - **Size:** 105,542 items.
   - **Key Features:** `article_id`, `product_code`, `product_type`, `color_group`, `graphical_appearance`, etc.

2. **Customers:**
   - Contains demographic and engagement information for 1.37 million customers.
   - **Key Features:** `customer_id`, `age`, `postal_code`, `club_member_status`, `fashion_news_frequency`, etc.

3. **Transactions:**
   - Records customer purchases, including timestamps, article IDs, and sales channels.
   - **Size:** 28 million transactions.
   - **Key Features:** `customer_id`, `article_id`, `t_dat` (purchase date), `price`, `sales_channel_id`.

4. **Image Embeddings:**
   - ResNet152 embeddings for product images, stored as 2048-dimensional vectors.
   - Used to capture visual similarity between fashion items.

---

## Methods

### Step 1: Data Loading and Initial Processing

- **Purpose:** Load and preprocess customer, article, and transaction data.
- **Challenges:** Handling large datasets, ensuring data consistency, and processing high-dimensional image data.
- **Approach:** 
  - Batch processing and parallel loading techniques.
  - Data type standardization (e.g., converting IDs to strings, dates to datetime format).
  - Image embeddings are processed using PCA to reduce dimensionality while maintaining 95% variance.

### Step 1.2: Image Embedding

- **Purpose:** Convert product images into numerical representations using ResNet152.
- **Process:**
  - Images are resized to 224x224 pixels and preprocessed using ResNet's `preprocess_input`.
  - ResNet152 generates 2048-dimensional embeddings, which are stored as `.npy` files.
  - PCA is applied to reduce dimensionality while preserving 95% of the variance.

### Step 2: Data Splitting

- **Purpose:** Split the data into training, validation, and test sets while respecting the temporal nature of purchases.
- **Approach:**
  - Transactions are split chronologically: 70% for training, 15% for validation, and 15% for testing.
  - This ensures that the model is trained on older data and tested on newer data, mimicking real-world scenarios.

### Step 3: Feature Engineering

- **Purpose:** Create meaningful features for customers, articles, and their interactions.
- **Customer Features:**
  - Purchase behavior (e.g., total transactions, average purchase price, purchase frequency).
  - Recent activity (e.g., number of purchases in the last 30 days).
- **Article Features:**
  - Popularity metrics (e.g., total sales, number of unique customers).
  - Recent performance (e.g., sales in the last 30 days).
- **Interaction Features:**
  - Temporal features (e.g., day of the week, month, weekend indicator).

### Step 4: Model Data Preparation

- **Purpose:** Prepare the data for model training by generating negative samples and preprocessing features.
- **Negative Sampling:**
  - For each positive interaction (customer purchase), 4 negative samples are generated by randomly sampling articles.
  - This helps the model learn to distinguish between items customers like and dislike.
- **Feature Preprocessing:**
  - Standardize numerical features and encode categorical variables.
  - Convert customer and article IDs to integer indices.
  - Maintain image embeddings for visual similarity.

### Step 5: Model Architecture

- **Purpose:** Build a two-tower neural network for retrieval and ranking.
- **Retrieval Model:**
  - **Customer Tower:** Maps customer features into a 32-dimensional embedding.
  - **Article Tower:** Maps article features into a 32-dimensional embedding.
  - Both towers use ReLU activation and layer normalization.
- **Ranking Model:**
  - Combines customer and article embeddings to predict the likelihood of a purchase.
  - Uses dense layers with ReLU activation and a sigmoid output layer.
- **Training:**
  - Optimizer: Adam with a learning rate of 0.001.
  - Loss: Combined retrieval and ranking loss.
  - Metrics: AUC, Precision, Recall.

### Step 6: Inference

- **Purpose:** Generate recommendations for customers using the trained model.
- **Process:**
  - Retrieve potential candidates using the retrieval model.
  - Rank candidates using the ranking model.
  - Filter out previously purchased items.
  - Return the top K recommendations for each customer.

### Step 7: Evaluation

- **Purpose:** Evaluate the recommendation system using multiple metrics.
- **Metrics:**
  - **Precision@K:** Measures the accuracy of the top K recommendations.
  - **Recall@K:** Measures the coverage of customer interests.
  - **NDCG@K:** Measures the ranking quality of the recommendations.
- **Results:**
  - The model achieves high precision and AUC scores, indicating strong performance in recommending relevant items.

---

## Results and Evaluation

- **Training:**
  - The model shows consistent improvement over 500 epochs.
  - **Precision:** Ranges between 0.85 and 0.925.
  - **AUC:** Consistently above 0.85, indicating strong performance.
- **Inference:**
  - The system successfully generates personalized recommendations for customers.
  - Recommendations are filtered to exclude previously purchased items, ensuring novelty.

---

## Conclusion

This project successfully builds a fashion recommendation system for H&M, leveraging customer purchase history, product metadata, and image embeddings. The two-tower neural network architecture allows for efficient retrieval and ranking of relevant items, while the evaluation metrics demonstrate the system's effectiveness in providing personalized recommendations.

