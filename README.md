# PERFORMANCE EVALUATION OF MACHINE LEARNING-BASED MONEY LAUNDERING DETECTION IN BITCOIN
### Final Year Project (FYP)

## 📌 Project Overview
This project focuses on the performance evaluation of machine learning models for detecting illicit money laundering activities within the Bitcoin network. As cryptocurrency adoption grows, so does its misuse for illegal transactions. This project aims to identify and classify these illicit transactions effectively.

The project implements and compares tree-based ensemble methods to distinguish between licit and illicit transactions.

## 🧠 Models Implemented
The following algorithms were implemented and tuned for optimal performance:
1.  **Random Forest**
2.  **XGBoost** (Extreme Gradient Boosting)
3.  **LightGBM** (Light Gradient Boosting Machine)

Hyperparameters were optimized (e.g., `max_depth`, `learning_rate`) to balance bias and variance and identify the most significant features driving the predictions.

## 📂 Dataset
The model is trained on the **Elliptic Data Set**, a graph network of Bitcoin transactions.
* **Dataset Name:** Elliptic Data Set
* **Source:** [Kaggle / UCI Machine Learning Repository](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set)
* **Description:** The dataset maps Bitcoin transactions to real entities belonging to licit categories (exchanges, wallet providers, miners, licit services, etc.) versus illicit ones (scams, malware, terrorist organizations, ransomware, Ponzi schemes, etc.).
* *Note: Due to GitHub's file size limits, the raw dataset is not included in this repository. Please download it from the source above.*

## 🛠️ Tech Stack & Requirements
The project is built using **Python** in a **Jupyter Notebook** environment.

**Key Libraries:**
* `pandas` & `numpy` (Data Manipulation)
* `scikit-learn` (Model building & evaluation)
* `xgboost` (Advanced gradient boosting)
* `lightgbm` (Efficient gradient boosting)
* `matplotlib` & `seaborn` (Visualization)

## 🚀 How to Run
1.  **Clone the repository:**
    *(Replace the link below with your actual repository URL)*
    ```bash
    git clone https://github.com/royliee/fyp_final_project_code.git
    ```
2.  **Navigate to the project folder:**
    ```bash
    cd fyp_final_project_code
    ```
3.  **Install dependencies:**
    ```bash
    pip install pandas numpy scikit-learn xgboost lightgbm matplotlib
    ```
4.  **Run the Notebook:**
    ```bash
    jupyter notebook "Untitled24.ipynb"
    ```
    *(Note: If you renamed your file, replace "Untitled24.ipynb" with your new file name)*

## 📜 License
This project is for educational purposes as part of the Final Year Project curriculum.
