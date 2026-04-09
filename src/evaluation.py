"""Module for evaluating model performance"""
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
from .utils import logger


def calculate_metrics(y_true, y_pred):
    """
    Calculate evaluation metrics for regression models.
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        
    Returns:
        dict: Dictionary containing MAE, RMSE, and R2 score
    """
    try:
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        
        logger.info(f"MAE: ${mae:,.2f}, RMSE: ${rmse:,.2f}, R²: {r2:.4f}")
        
        return {
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2
        }
    
    except Exception as e:
        logger.error(f"Error calculating metrics: {str(e)}")
        raise


def evaluate_model(model, X_train, y_train, X_test, y_test, model_name="Model"):
    """
    Evaluate a model on training and test sets.
    
    Args:
        model: Trained model
        X_train: Training features
        y_train: Training target
        X_test: Test features
        y_test: Test target
        model_name: Name of the model for logging
        
    Returns:
        dict: Dictionary with train and test metrics
    """
    try:
        logger.info(f"Evaluating {model_name}...")
        
        # Get predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # Calculate metrics
        train_metrics = calculate_metrics(y_train, y_train_pred)
        test_metrics = calculate_metrics(y_test, y_test_pred)
        
        logger.info(f"{model_name} - Train MAE: ${train_metrics['MAE']:,.2f}")
        logger.info(f"{model_name} - Test MAE: ${test_metrics['MAE']:,.2f}")
        
        return {
            'train': train_metrics,
            'test': test_metrics,
            'train_pred': y_train_pred,
            'test_pred': y_test_pred
        }
    
    except Exception as e:
        logger.error(f"Error evaluating model: {str(e)}")
        raise


def compare_models(models_dict, X_test, y_test):
    """
    Compare multiple models based on test MAE.
    
    Args:
        models_dict (dict): Dictionary of model_name: model pairs
        X_test: Test features
        y_test: Test target
        
    Returns:
        dict: Comparison results sorted by MAE (best first)
    """
    try:
        results = {}
        
        for model_name, model in models_dict.items():
            y_pred = model.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)
            results[model_name] = mae
        
        # Sort by MAE (ascending)
        sorted_results = dict(sorted(results.items(), key=lambda x: x[1]))
        
        logger.info("Model Comparison Results:")
        for name, mae in sorted_results.items():
            logger.info(f"  {name}: MAE = ${mae:,.2f}")
        
        return sorted_results
    
    except Exception as e:
        logger.error(f"Error comparing models: {str(e)}")
        raise
