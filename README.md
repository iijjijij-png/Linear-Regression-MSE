# Simple Linear Regression Model

This project is a Python script that demonstrates how to create and evaluate a simple linear regression model using the `pandas` and `scikit-learn` libraries. The script uses a sample dataset of years of experience and corresponding salaries to train and test the model.

## How to Use

### Prerequisites

- Python 3.x installed on your system.
- The following Python libraries installed:
  - `pandas`
  - `scikit-learn`
You can install these libraries using pip:

    ```bash
    pip install pandas scikit-learn
    ```
### Running the Script

1. **Save the script as linear_regression.py or any name you prefer.**
2. **Open a terminal or command prompt.**
3. **Navigate to the directory where the script is saved.**
4. **Run the script by executing the following command:**
    ```bash
    python linear_regression.py
    ```
## Script Details

The script performs the following steps:
### Data Preparation

- A sample dataset is created using years of experience and corresponding salaries. This can be replaced with your actual dataset.

### Feature Selection

- The YearsExperience column is selected as the feature (X).
- The Salary column is selected as the target variable (y).

### Train-Test Split

- The dataset is split into training and testing sets using an 80-20 split.

### Model Training

- A linear regression model is created and trained on the training set.

### Prediction

- The model predicts salaries for the test set based on years of experience.

### Model Evaluation

- The Mean Squared Error (MSE) is calculated to evaluate the model's performance on the test set.

### Example
```bash
Here is a sample output of the script:
python
Mean Squared Error on the test set: 25691478.63
