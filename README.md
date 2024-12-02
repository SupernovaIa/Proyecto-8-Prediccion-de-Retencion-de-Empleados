## 📜 Project Overview

This project aims to address one of the biggest challenges in Human Resources departments: employee retention. Through data analysis and Machine Learning model development, we seek to identify patterns, trends, and key factors that influence employees' decisions to stay or leave a fictitious company.

To achieve this, we will analyze a dataset that includes satisfaction surveys, performance metrics, and working hours. Additionally, we will develop a predictive model to anticipate the likelihood of employee turnover. This analysis will help propose practical strategies to improve employee retention and optimize HR decision-making.


## 🎯 Specific Objectives

1. **Data Exploration:**

- Understand and analyze the dataset variables.
- Identify patterns, trends, and potential outliers.

2. **Data Preprocessing:**

- Data cleaning and transformation.
- Encode categorical variables.
- Scale numerical variables.
- Handle missing values and class imbalances.

3. **Model Building:**

- Train predictive models such as Decision Trees, Logistic Regression, and Random Forest.
- Evaluate performance using metrics like F1-score, accuracy, and confusion matrix.

4. **Visualization:**

- Create charts explaining variable importance and model predictions.

5. **Optimization:**

- Tune hyperparameters to improve model performance.

6. **Recommendations:**

- Propose practical strategies to improve employee retention based on the results obtained.

## 🛠️ Project Structure

```
Proyecto-8-Prediccion-de-Retencion-de-Empleados
├── data/                               # Folder containing datasets
├── model/                              # Saved trained models
├── notebook/                           # Jupyter Notebooks for EDA and modeling
├── src/                                # Source code
├── .gitignore                          # Git ignore file
├── README.md                           # Project documentation
├── requirements.txt                    # Project dependencies
```

## 🔧 Installation and Requirements

This project was developed in Python 3.12. To set it up, follow these steps:

1. **Clone the repository:**

   ```bash
   git clone https://github.com/SupernovaIa/Proyecto-8-Prediccion-de-Retencion-de-Empleados
   ```

2. **Navigate to the project directory:**

   ```bash
   cd Proyecto-8-Prediccion-de-Retencion-de-Empleados
   ```

3. **Install the necessary dependencies:**

   Required libraries include:

   - **Pandas:** Data manipulation and analysis ([Documentation](https://pandas.pydata.org/docs/)).
   - **NumPy:** Numerical data processing ([Documentation](https://numpy.org/doc/)).
   - **Matplotlib:** Data visualization ([Documentation](https://matplotlib.org/stable/contents.html)).
   - **Seaborn:** Statistical visualizations ([Documentation](https://seaborn.pydata.org/)).
   - **Scikit-learn:** Machine Learning algorithms ([Documentation](https://scikit-learn.org/stable/)).
   - **Category Encoders:** Encoding categorical variables ([Documentation](https://contrib.scikit-learn.org/category_encoders/)).
   - **SciPy:** Scientific computations ([Documentation](https://scipy.org/docs.html)).
   - **XGBoost:** Advanced gradient boosting algorithms ([Documentation](https://xgboost.readthedocs.io/)).
   - **SHAP:** Explainability for ML models ([Documentation](https://shap.readthedocs.io/en/latest/)).


   To install everything, run:

   ```bash
   pip install -r requirements.txt
   ```

4. **Run the notebooks:**

   Perform exploratory analysis and modeling by executing the notebooks in the `notebook/` folder.


## 📊 Results and Conclusions

### Exploratory Analysis

`Attrition` appears to be higher in proportion among people who are:  

* Younger.  

* Have spent little time at the company or have fewer years of work experience overall.  

* Have low job, environmental, or work-life balance satisfaction.  

### Model Performance

We are evaluating a model that predicts whether an employee will leave the company. In this case, we want to minimize the number of false negatives, i.e., we aim to miss as few positives as possible. False positives are also undesirable, but raising alarms for an employee who is not going to leave while thinking they will is less critical than the opposite.

Therefore, we are particularly interested in the `recall` metric over `precision`. We will also aim to maximize the `f1_score` as much as possible.

Let us recall that the `recall` metric tells us, out of all the **actually** positive cases, how many we have correctly identified.

![Model comparison](https://github.com/user-attachments/assets/9c8c3bcd-2a69-431b-be03-51620b56cb2c)

The model with the best `recall` is the `random_forest`, although all of them are quite similar. Due to computation time and overall metrics, we will stick with this model.

We also observe that, in general, there is a bit of overfitting, which we could try to reduce by adding more data if possible or by attempting to reduce the number of predictor variables. Next, we will proceed to build a model that handles data preprocessing differently so that we have a smaller number of columns.

### Feature importance

![f_imp](https://github.com/user-attachments/assets/f45e6ebc-758a-4454-83bc-d71ee1929f64)

![f_imp_shap](https://github.com/user-attachments/assets/4f6148d2-d914-43df-9a9d-6ec823c5277d)

Similar to what was concluded in the exploratory analysis, we see that age and related variables such as total years worked and total years at the company have a strong impact on the model. Salary, raises, and satisfaction also seem to play an important role.

### Confusion matrix

![conf_matrix](https://github.com/user-attachments/assets/80f7f29c-0816-42a8-919b-e45953c705c6)



## 📊 Recommendations

- **Job Satisfaction:** Implement regular surveys and wellness programs to increase job satisfaction.

- **Training and Promotion:** Design clear career paths and provide internal growth opportunities.

- **Ensuring data quality:** Properly identifying unique employees to avoid duplicates that could bias the models.


## 🧬 Next Steps

1. **Test other preprocessing methods:** Try different combinations of encoding and scaling for better data preparation before training the models.

2. **Explore other training techniques:** Use stacking techniques to improve resulting metrics and achieve a better final model.

3. **Attempt data balancing:** The data had significant imbalance, so rebalancing techniques could be applied to improve the model. However, this often introduces biases into the data, so it should be done carefully.

4. **Develop a web interface:** Create a web interface where new employee data can be manually entered to predict the risk of them leaving the company.

## 🤝 Contributions

Contributions are welcome! Follow these steps to collaborate:

1. Fork the repository.
2. Create a new branch for your changes.
3. Submit a pull request when ready.

If you have suggestions or improvements, feel free to open an issue.


## ✍️ Author

**Javier Carreira**  - *Lead Developer*  
GitHub: [https://github.com/SupernovaIa](https://github.com/SupernovaIa)

