# 🏨 Hotel Booking Cancellation Prediction

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange)
![Pandas](https://img.shields.io/badge/Library-Pandas-150458)
![Status](https://img.shields.io/badge/Status-Completed-green)

## 📌 Project Overview
This project aims to build a machine learning model to predict whether a customer will cancel their hotel booking. By analyzing various features such as lead time, pricing, and booking segment, the model helps hotel management forecast cancellations, allowing for better inventory management and revenue optimization.

This solution was developed as part of a Kaggle competition, implementing a full Data Science pipeline from data cleaning to hyperparameter tuning.

## ❓ Problem Statement
Hotel cancellations can cause significant revenue loss and operational challenges. The objective is to predict the binary target `booking_status`:
* **0**: Booking not cancelled
* **1**: Booking cancelled

## 📂 Dataset Description
The dataset consists of booking records with the following features:

| Feature | Description |
| :--- | :--- |
| **id** | Unique identifier for the booking |
| **adults/children** | Number of adults and children |
| **weekends/weekdays** | Number of weekend and weekday nights booked |
| **meal_type** | Type of meal selected |
| **room_type** | Room category selected |
| **arrival** | Date of arrival |
| **lead_time** | Days between booking and arrival |
| **segment** | Market segment (e.g., Online, Offline) |
| **repeat** | Whether the customer is a repeat guest |
| **price** | Average price per room |
| **requests** | Number of special requests made |
| **booking_status** | **Target Variable** (0 or 1) |

## 🛠️ Methodology

### 1. Data Preprocessing & Cleaning
* **Missing Values:** * Categorical features (`room_type`, `segment`, `meal_type`) were imputed with the label `'Unknown'`.
    * Numerical features (`lead_time`, `price`) were imputed using the **median**.
* **Invalid Dates:** Handling `NaT` values in the `arrival` column by imputing with the mode.
* **Duplicate Removal:** Checked for and removed duplicate entries to ensure data integrity.

### 2. Feature Engineering
* **Date Extraction:** The `arrival` date column was decomposed into usable numerical features: `Arrival Year`, `Month`, `Day`, `Weekday`, and `Week Number`.
* **Outlier Analysis:** Analyzed `price` and `lead_time` using the IQR method. Outliers were retained as they represented genuine high-value or long-term bookings important for model generalization.

### 3. Exploratory Data Analysis (EDA)
Key insights derived from the data:
* **Booking Status Balance:** Visualized the class distribution between cancelled and non-cancelled bookings.
* **Price Distribution:** Observed a right-skewed distribution for room prices.
* **Segment Analysis:** Certain market segments (e.g., Online bookings) showed significantly higher cancellation rates than others.

### 4. Data Transformation
A `ColumnTransformer` pipeline was established:
* **Numerical Features:** Scaled using `StandardScaler`.
* **Categorical Features:** Encoded using `OneHotEncoder` (ignoring unknown categories during inference).

## 🤖 Model Development

### Model Selection
Seven different classification algorithms were trained and evaluated using **F1-Score** (harmonic mean of precision and recall) as the primary metric.

| Model | Accuracy | F1 Score |
| :--- | :--- | :--- |
| **Random Forest** | **0.89** | **0.83** |
| XGBoost | 0.88 | 0.82 |
| Decision Tree | 0.85 | 0.77 |
| Gradient Boosting | 0.85 | 0.75 |
| SVC | 0.84 | 0.73 |
| KNN | 0.83 | 0.73 |
| Logistic Regression | 0.80 | 0.68 |

### Hyperparameter Tuning
`RandomizedSearchCV` was used to tune the top performers.
* **Logistic Regression:** Tuned C values and solvers.
* **Random Forest:** Tuned estimators, max depth, and split criteria.
* **XGBoost:** Tuned learning rate, max depth, and subsample ratios.

**Final Model:** The **Random Forest Classifier** was selected as the final model due to its superior F1 score and stability on the validation set.

## 🚀 Installation & Usage

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/yourusername/hotel-booking-prediction.git](https://github.com/yourusername/hotel-booking-prediction.git)
   cd hotel-booking-prediction
