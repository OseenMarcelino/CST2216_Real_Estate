"""Module for training machine learning models"""
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from .utils import logger


def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split data into training and testing sets.
    
    Args:
        X (pd.DataFrame): Features
        y (pd.Series): Target
        test_size (float): Proportion of test set
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    try:
        logger.info(f"Splitting data with test_size={test_size}")
        
        # Use property_type_Condo for stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=random_state,
            stratify=X['property_type_Condo']
        )
        
        logger.info(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
        return X_train, X_test, y_train, y_test
    
    except Exception as e:
        logger.error(f"Error splitting data: {str(e)}")
        raise


def train_linear_regression(X_train, y_train):
    """
    Train a Linear Regression model.
    
    Args:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training target
        
    Returns:
        LinearRegression: Trained model
    """
    try:
        logger.info("Training Linear Regression model...")
        model = LinearRegression()
        model.fit(X_train, y_train)
        logger.info("Linear Regression model trained successfully")
        return model
    
    except Exception as e:
        logger.error(f"Error training Linear Regression: {str(e)}")
        raise


def train_random_forest(X_train, y_train, n_estimators=200):
    """
    Train a Random Forest Regressor model.
    
    Args:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training target
        n_estimators (int): Number of trees
        
    Returns:
        RandomForestRegressor: Trained model
    """
    try:
        logger.info(f"Training Random Forest model with {n_estimators} trees...")
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            criterion='absolute_error',
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        logger.info("Random Forest model trained successfully")
        return model
    
    except Exception as e:
        logger.error(f"Error training Random Forest: {str(e)}")
        raise


def make_prediction(model, X):
    """
    Make predictions using a trained model.
    
    Args:
        model: Trained model
        X (pd.DataFrame): Features for prediction
        
    Returns:
        np.array: Predictions
    """
    try:
        predictions = model.predict(X)
        logger.info(f"Predictions made for {len(predictions)} samples")
        return predictions
    
    except Exception as e:
        logger.error(f"Error making predictions: {str(e)}")
        raise
