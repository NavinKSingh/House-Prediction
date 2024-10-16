California Housing Price Prediction
This project utilizes the California housing dataset to predict housing prices using the XGBoost regression algorithm. The dataset includes various features related to housing, and the objective is to build a predictive model that can estimate the price of a house based on these features.

Table of Contents
Technologies Used
Dataset
Getting Started
Model Evaluation
Visualization
License
Technologies Used
Python
NumPy
Pandas
Matplotlib
Seaborn
Scikit-learn
XGBoost
Dataset
The dataset used in this project is the California housing dataset, which can be fetched using the sklearn.datasets.fetch_california_housing function. It contains various features such as median income, house age, and more, along with the target variable, which is the median house value.

Getting Started
To run this code, ensure you have the required libraries installed. You can install them using pip:

bash
Copy code
pip install numpy pandas matplotlib seaborn scikit-learn xgboost
Then, you can execute the script in your Python environment. The code performs the following steps:

Data Loading: Fetch the California housing dataset.
Data Preparation: Convert the dataset into a Pandas DataFrame and handle the target variable.
Exploratory Data Analysis (EDA): Check for missing values and visualize the correlation matrix.
Train-Test Split: Split the dataset into training and testing sets.
Model Training: Train an XGBoost regressor on the training data.
Model Evaluation: Evaluate the model using R-squared and Mean Absolute Error metrics.
Visualization: Plot the actual prices against the predicted prices for the training data.
Model Evaluation
After training the model, it evaluates performance using the following metrics:

R-squared Score: Measures the proportion of variance in the target variable that is predictable from the independent variables.
Mean Absolute Error (MAE): The average of absolute errors between predicted and actual values.
Results for both the training and testing datasets are printed out.

Visualization
The code includes a heatmap to visualize the correlation matrix of the features, helping to understand relationships between them. Additionally, a scatter plot is generated to compare actual prices against predicted prices for the training data.

License
This project is licensed under the MIT License.
