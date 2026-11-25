import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

NUMERICAL = ['Age', 'Annual_Income']
NOMINAL = ['Gender', 'City']
ORDINAL = ['Education', 'Membership']
ORDINAL_ORDER = [
    ['School', 'Graduate', 'Postgraduate', 'PhD'],
    ['Bronze', 'Silver', 'Gold', 'Platinum'],
]
FEATURES = NUMERICAL + NOMINAL + ORDINAL
TARGET = 'Purchased'


def validate(df):
    problems = []
    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        problems.append('Missing columns: ' + ', '.join(missing))
        return problems
    for col, order in zip(ORDINAL, ORDINAL_ORDER):
        unknown = sorted(set(df[col].dropna().unique()) - set(order))
        if unknown:
            problems.append(col + ' has unknown values: ' + ', '.join(map(str, unknown)))
    if df[FEATURES + [TARGET]].isna().any().any():
        problems.append('The data contains empty cells. Fill or remove them first.')
    return problems


def build_preprocessor():
    return ColumnTransformer(
        [
            ('num', StandardScaler(), NUMERICAL),
            ('nom', OneHotEncoder(handle_unknown='ignore'), NOMINAL),
            ('ord', OrdinalEncoder(categories=ORDINAL_ORDER), ORDINAL),
        ],
        sparse_threshold=0,
        verbose_feature_names_out=False,
    )


def encode_preview(df):
    pre = ColumnTransformer(
        [
            ('nom', OneHotEncoder(), NOMINAL),
            ('ord', OrdinalEncoder(categories=ORDINAL_ORDER), ORDINAL),
        ],
        remainder='passthrough',
        sparse_threshold=0,
        verbose_feature_names_out=False,
    )
    out = pre.fit_transform(df[FEATURES])
    return pd.DataFrame(out, columns=pre.get_feature_names_out(), index=df.index).astype(float)


def train(df, test_size=0.2, seed=42):
    X, y = df[FEATURES], df[TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    model = Pipeline([
        ('pre', build_preprocessor()),
        ('clf', LogisticRegression(max_iter=1000)),
    ])
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)
    names = model.named_steps['pre'].get_feature_names_out()
    coefs = pd.Series(model.named_steps['clf'].coef_[0], index=names).sort_values()
    return {
        'model': model,
        'accuracy': accuracy_score(y_te, pred),
        'table': X_te.assign(Actual=y_te, Predicted=pred),
        'coefs': coefs,
        'rows': (len(X_tr), len(X_te)),
    }
