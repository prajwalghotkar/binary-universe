# Binary Universe

https://binary-universe-by-prajwal.streamlit.app/

A Streamlit app and a Jupyter notebook that convert categorical columns into numbers, train a logistic regression on customer data, and predict whether a customer will make a purchase. The app runs on a black background with tilted rain of 1s and 0s.

## Core concept

Machine learning models do math on numbers. Columns such as Gender or Membership hold words, so they must be encoded before training. The right encoder depends on the kind of column.

| Kind | Columns | Encoder | Result |
|---|---|---|---|
| Nominal (no order) | Gender, City | OneHotEncoder | One 0/1 column per value |
| Ordinal (has order) | Education, Membership | OrdinalEncoder with explicit order | Ranks 0, 1, 2, 3 |
| Numerical | Age, Annual_Income | StandardScaler | Mean 0, spread 1 |

Why the split matters: giving cities the numbers 0, 1, 2 would tell the model that one city is "more" than another. One-hot columns avoid that. For Education and Membership the order is real information, so the order is passed explicitly: School < Graduate < Postgraduate < PhD and Bronze < Silver < Gold < Platinum.

## How the original code works

1. `training_lg.py` loads the CSV, drops Purchased to form X, and keeps Purchased as y.
2. It splits 80 percent train and 20 percent test with a fixed seed and stratification so both sets keep the Yes/No ratio.
3. A ColumnTransformer applies one-hot encoding to nominal columns and ordinal encoding to ordinal columns. Numerical columns pass through untouched.
4. The transformer is fitted on training data only, then applied to test data. This prevents test information from leaking into training.
5. A LogisticRegression is fitted and predictions are printed.

## What this project changes

- Numerical columns are standardized. Annual_Income is in the tens of thousands while Age is in the tens, which slows or breaks logistic regression convergence when left unscaled.
- Encoding and model are combined in one Pipeline, so prediction on new data repeats exactly the same steps.
- Accuracy, a confusion matrix, and coefficients are reported, which the original script did not do.
- Data is validated before use, because OrdinalEncoder raises an error on a value outside the stated order, while OneHotEncoder ignores unseen values.

The sample file has only 30 rows, so accuracy on a 6 row test set changes a lot with the seed. Treat it as a demonstration, not a reliable estimate.

## Files

- `app.py` Streamlit interface with Encode, Train, and Predict tabs
- `pipeline.py` column groups, validation, encoders, and training
- `rain.js` the falling binary animation
- `styles.css` page styling
- `.streamlit/config.toml` dark theme
- `customer_purchase_notebook.ipynb` the same workflow step by step
- `dataset/customer_purchase_prediction.csv` sample data
- `requirements.txt` Python dependencies

## Setup

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows activate with `.venv\Scripts\activate`.

## Run the app

```
streamlit run app.py
```

## Run the notebook

```
jupyter notebook customer_purchase_notebook.ipynb
```

Start both from the project folder so the relative `dataset/` path resolves.

## Using your own data

Upload a CSV with these columns: Age, Annual_Income, Gender, City, Education, Membership, Purchased. Education must use School, Graduate, Postgraduate, PhD. Membership must use Bronze, Silver, Gold, Platinum. Purchased must be Yes or No.

## Tuning the rain

Open `rain.js` and change `size` for digit size, `angle` for tilt in radians, and `interval` for fall speed in milliseconds per step.
