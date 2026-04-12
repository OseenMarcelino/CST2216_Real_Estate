# Real Estate Price Prediction Project

A machine learning application for predicting real estate prices using Linear Regression and Random Forest models. This project demonstrates data science fundamentals including data loading, model training, evaluation, and a web-based interactive dashboard.

## Project Overview

This project analyzes real estate property data and builds predictive models to estimate property prices based on various features such as:
- Property characteristics (bedrooms, bathrooms, square footage)
- Location features (popular area)
- Financial indicators (property tax, insurance)
- Market conditions (recession status, year sold)

### Models Used
- **Linear Regression**: Simple, interpretable baseline model
- **Random Forest**: Ensemble method for improved predictions

## Project Structure

```
CST2216_Real_Estate/
├── data/
│   └── final.csv                 # Real estate dataset
├── src/
│   ├── __init__.py              # Package initialization
│   ├── data_loader.py           # Data loading and validation
│   ├── model.py                 # Model training functions
│   ├── evaluation.py            # Model evaluation metrics
│   └── utils.py                 # Logging and utility functions
├── app.py                       # Streamlit web application
├── requirements.txt             # Python dependencies
├── README.md                    
```

## Getting Started

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Installation

1. **Navigate to the project folder:**
   ```bash
   cd CST2216_Real_Estate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## How to Run
Start the Streamlit app to interact with the models through a user-friendly interface:

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501` in your browser.

**Features:**
- **Dashboard**: View data overview and statistics
- **Model Training**: Train both models and compare performance
- **Price Predictor**: Input property details and get price predictions

## Data Description

**File:** `data/final.csv`

**Columns (14 features + 1 target):**
- `price`: Property price (target variable) - USD
- `year_sold`: Year the property was sold
- `property_tax`: Annual property tax - USD
- `insurance`: Annual insurance cost - USD
- `beds`: Number of bedrooms
- `baths`: Number of bathrooms
- `sqft`: Square footage of the property
- `year_built`: Year the property was built
- `lot_size`: Size of the lot - square feet
- `basement`: Whether property has a basement (0/1)
- `popular`: Whether in a popular area (0/1)
- `recession`: Whether sold during recession (0/1)
- `property_age`: Age of the property (calculated)
- `property_type_Condo`: Type of property - 1 for Condo, 0 for House

## Code Organization

### `src/data_loader.py`
- `load_data()`: Loads CSV file with error handling
- `validate_data()`: Ensures all required columns exist
- `prepare_features()`: Separates features from target variable
- `get_data_info()`: Returns dataset statistics

### `src/model.py`
- `split_data()`: Splits data into train/test sets
- `train_linear_regression()`: Trains Linear Regression model
- `train_random_forest()`: Trains Random Forest model
- `make_prediction()`: Makes price predictions

### `src/evaluation.py`
- `calculate_metrics()`: Computes MAE, RMSE, and R² scores
- `evaluate_model()`: Evaluates model on train and test sets
- `compare_models()`: Compares multiple models

### `src/utils.py`
- `setup_logging()`: Configures logging to file and console
- Provides centralized error handling and logging

## Model Performance

The models are evaluated using three metrics:

1. **Mean Absolute Error (MAE)**: Average prediction error in dollars
2. **Root Mean Squared Error (RMSE)**: Penalizes larger errors more heavily
3. **R² Score**: Proportion of variance explained (0-1, higher is better)

Target: Achieve MAE < $70,000 for reliable predictions

## Deployment on Streamlit Cloud

### Steps to Deploy:

1. **Push code to GitHub:**

2. **Visit [Streamlit Cloud](https://streamlit.io/cloud):**
   - Sign in with your GitHub account
   - Click "New app"
   - Select your repository
   - Set main file: `app.py`
   - Click "Deploy"

3. **Your app will be live at:** `https://<your-username>-realestate-<name>.streamlit.app`

### Streamlit Cloud Requirements:
- GitHub account with repository
- `requirements.txt` in project root
- `app.py` as main file

## Logging

The application logs all operations to:
- **Console**: INFO level and above
- **File**: `logs/application.log` (created automatically)

Logs include:
- Data loading operations
- Model training progress
- Predictions made
- Any errors encountered

## Error Handling

The application includes robust error handling for:
- Missing or corrupted data files
- Invalid input values
- Model training failures
- Prediction errors

All errors are logged and displayed to the user in the Streamlit app.

## Libraries Used

| Library | Purpose |
|---------|---------|
| pandas | Data manipulation and analysis |
| numpy | Numerical computations |
| scikit-learn | Machine learning models and metrics |
| streamlit | Web application framework |
| Pillow | Image processing |



