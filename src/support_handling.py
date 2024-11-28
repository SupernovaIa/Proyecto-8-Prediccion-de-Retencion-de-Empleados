# Tratamiento de datos
# -----------------------------------------------------------------------
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer


def impute_knn(df, tv, n=5):
    """
    Imputes missing values in a DataFrame using the K-Nearest Neighbors (KNN) algorithm.

    Parameters:
    - df (pd.DataFrame): The input DataFrame with missing values.
    - tv (str): The target variable column name. This column will not be used for imputing if numeric.
    - n (int): The number of neighbors to consider for imputing missing values. Default is 5.

    Returns:
    - (pd.DataFrame): A DataFrame with missing values imputed using KNN.
    """

    # Select numeric variables
    df_num = df.select_dtypes(include=np.number)

    if tv in df_num.columns:
        df_num = df_num.drop(columns=[tv])

    # KNN algorithm for imputing
    imputer_knn = KNNImputer(n_neighbors=n)
    knn_imputed = imputer_knn.fit_transform(df_num)
    df_imputed = pd.DataFrame(knn_imputed, columns=df_num.columns, index=df.index)

    # Restore target variable, if it was numeric
    if tv in df.columns and tv not in df_imputed.columns:
        df_imputed[tv] = df[tv]

    df_knn = df.copy()
    df_knn.update(df_imputed)

    return df_knn


# Data Treatment
# -----------------------------------------------------------------------
import numpy as np
import pandas as pd

# Imputation of Missing Values Using Advanced Statistical Methods
# -----------------------------------------------------------------------
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import KNNImputer


class MissingValuesHandler:
    """
    A class for handling and imputing missing values in a dataframe using various methods.

    Attributes:
    - dataframe (pandas.DataFrame): The dataframe to be processed for missing values.

    Methods:
    - __init__(dataframe): Initializes the object with a dataframe.
    - get_missing_values_percentages(): Calculates the percentage of missing values for each column.
    - select_missing_values_columns(): Selects columns with missing values that are of numeric data type.
    - use_knn(list_columns=None, n=5): Imputes missing values using the K-Nearest Neighbors (KNN) algorithm.
    - use_imputer(list_columns=None): Imputes missing values using the Iterative Imputer.
    - method_comparison(): Compares the imputed and original columns with missing values.
    """

    def __init__(self, dataframe):
        """
        Initializes the object with a dataframe.

        Parameters:
        - dataframe (pandas.DataFrame): The dataframe to be stored as an attribute.
        """
        self.dataframe = dataframe
    

    def get_missing_values_percentages(self):
        """
        Calculates the percentage of missing values for each column in the dataframe.

        Returns:
        - (pandas.Series): A series containing the percentage of missing values for columns with missing data.
        """
        df_missing_values = (self.dataframe.isna().sum() / self.dataframe.shape[0]) * 100
        return df_missing_values[df_missing_values > 0]
    

    def select_missing_values_columns(self):
        """
        Selects columns with missing values that are of numeric data type.

        Returns:
        - (pandas.Index): An index object containing the names of numeric columns with missing values.
        """
        filter_missing_values = self.dataframe.columns[self.dataframe.isna().any()]
        cols = self.dataframe[filter_missing_values].select_dtypes(include=np.number).columns
        return cols


    def use_knn(self, list_columns=None, n=5):
        """
        Imputes missing values in the specified columns using the K-Nearest Neighbors (KNN) algorithm.

        Parameters:
        - list_columns (list, optional): A list of column names to impute. Defaults to numeric columns with missing values.
        - n (int, optional): The number of neighbors to use for the KNN algorithm. Defaults to 5.

        Returns:
        - (pandas.DataFrame): The dataframe with new columns containing the imputed values, named with the original column names appended with "_knn".
        """
        if list_columns == None:
            list_columns = self.select_missing_values_columns().to_list()

        imputer = KNNImputer(n_neighbors= n)
        imputed = imputer.fit_transform(self.dataframe[list_columns])

        new_columns = [col + "_knn" for col in list_columns]
        self.dataframe[new_columns] = imputed

        return self.dataframe
    

    def use_iterative(self, list_columns=None):
        """
        Imputes missing values in the specified columns using the Iterative Imputer.

        Parameters:
        - list_columns (list, optional): A list of column names to impute. Defaults to numeric columns with missing values.

        Returns:
        - (pandas.DataFrame): The dataframe with new columns containing the imputed values, named with the original column names appended with "_iterative".
        """
        if list_columns == None:
            list_columns = self.select_missing_values_columns().to_list()

        imputer = IterativeImputer(max_iter=20, random_state=42)
        imputed = imputer.fit_transform(self.dataframe[list_columns])

        new_columns = [col + "_iterative" for col in list_columns]
        self.dataframe[new_columns] = imputed

        return self.dataframe
    

    def method_comparison(self):
        """
        Compares the imputed columns generated by different imputation methods and the original columns with missing values.

        Returns:
        - (pandas.DataFrame): A summary statistics dataframe for the original and imputed columns, ordered alphabetically by column name.
        """
        columns = self.dataframe.columns[self.dataframe.columns.str.contains("_knn|_iterative")].tolist() + self.select_missing_values_columns().tolist()
        results = self.dataframe.describe()[columns].reindex(sorted(columns), axis=1)
        return results
    
    def drop_columns(self, list_colums):
        """
        Drops specified columns from the dataframe.

        Parameters:
        - list_colums (list): List of column names to be dropped from the dataframe.

        Returns:
        - None: Modifies the dataframe in place.

        Raises:
        - ValueError: If list_colums is not a list.
        - KeyError: If any of the columns in list_colums do not exist in the dataframe.
        """
        if not isinstance(list_colums, list):
            raise ValueError("list_colums must be a list of column names.")

        missing_columns = [col for col in list_colums if col not in self.dataframe.columns]
        if missing_columns:
            raise KeyError(f"The following columns do not exist in the dataframe: {missing_columns}")
        
        self.dataframe.drop(list_colums, axis=1, inplace=True)
