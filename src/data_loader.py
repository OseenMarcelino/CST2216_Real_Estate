"""Module for loading and validating real estate data"""
import pandas as pd
import os
from .utils import logger


def load_data(file_path):
    """
    Load real estate data from CSV file.
    
    Args:
        file_path (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: Loaded data
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the file is empty or corrupted
    """
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        logger.info(f"Loading data from {file_path}")
        df = pd.read_csv(file_path)
        
        if df.empty:
            raise ValueError("The CSV file is empty")
        
        logger.info(f"Data loaded successfully. Shape: {df.shape}")
        return df
    
    except FileNotFoundError as e:
        logger.error(str(e))
        raise
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise ValueError(f"Error loading data: {str(e)}")


def validate_data(df):
    """
    Validate that the data has required columns.
    
    Args:
        df (pd.DataFrame): Data to validate
        
    Returns:
        bool: True if valid
        
    Raises:
        ValueError: If required columns are missing
    """
    required_columns = ['price', 'year_sold', 'property_tax', 'insurance', 
                       'beds', 'baths', 'sqft', 'year_built', 'lot_size', 
                       'basement', 'popular', 'recession', 'property_age', 
                       'property_type_Condo']
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    logger.info("Data validation passed")
    return True


def prepare_features(df):
    """
    Prepare features and target for model training.
    
    Args:
        df (pd.DataFrame): Raw data
        
    Returns:
        tuple: (X, y) where X is features and y is target
    """
    try:
        validate_data(df)
        
        # Separate features and target
        X = df.drop('price', axis=1)
        y = df['price']
        
        logger.info(f"Features shape: {X.shape}, Target shape: {y.shape}")
        return X, y
    
    except Exception as e:
        logger.error(f"Error preparing features: {str(e)}")
        raise


def get_data_info(df):
    """
    Get basic information about the dataset.
    
    Args:
        df (pd.DataFrame): Data
        
    Returns:
        dict: Dictionary with data info
    """
    return {
        'num_samples': len(df),
        'num_features': len(df.columns) - 1,
        'target_column': 'price',
        'mean_price': df['price'].mean(),
        'min_price': df['price'].min(),
        'max_price': df['price'].max()
    }
