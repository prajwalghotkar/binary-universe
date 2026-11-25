from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

import pipeline as pl

BASE = Path(__file__).parent
DEFAULT_DATA = BASE / 'dataset' / 'customer_purchase_prediction.csv'

st.set_page_config(page_title='Binary Universe', layout='wide')
st.markdown('<style>' + (BASE / 'styles.css').read_text() + '</style>', unsafe_allow_html=True)
components.html('<script>' + (BASE / 'rain.js').read_text() + '</script>', height=0)

st.title('Turn categories into numbers')
st.markdown(
    '<p class="lead">Models read numbers, not words. Load customer data, see how every text column '
    'is encoded, train a logistic regression, and predict whether a new customer will buy.</p>',
    unsafe_allow_html=True,
)

upload = st.file_uploader('Upload a CSV with the same columns, or use the sample data', type='csv')
df = pd.read_csv(upload) if upload else pd.read_csv(DEFAULT_DATA)

problems = pl.validate(df)
if problems:
    for p in problems:
        st.error(p)
    st.stop()

tab_encode, tab_train, tab_predict = st.tabs(['Encode', 'Train', 'Predict'])

with tab_encode:
    left, right = st.columns(2)
    with left:
        st.subheader('Raw data')
        st.dataframe(df.head(10))
    with right:
        st.subheader('Encoded data')
        st.dataframe(pl.encode_preview(df).head(10))

    a, b, c = st.columns(3)
    a.markdown(
        '**Nominal: ' + ', '.join(pl.NOMINAL) + '**\n\nNo natural order, so each value gets its own '
        '0 or 1 column. A city is never ranked above another.'
    )
    b.markdown(
        '**Ordinal: ' + ', '.join(pl.ORDINAL) + '**\n\nThe order matters, so values become '
        'ranks from 0 upward.\n\n'
        + '\n\n'.join(' < '.join(o) for o in pl.ORDINAL_ORDER)
    )
    c.markdown(
        '**Numerical: ' + ', '.join(pl.NUMERICAL) + '**\n\nKept as numbers and standardized '
        'during training so income does not outweigh age.'
    )

with tab_train:
    test_size = st.slider('Share of rows held back for testing', 0.1, 0.4, 0.2, 0.05)
    if st.button('Train model', type='primary'):
        st.session_state['result'] = pl.train(df, test_size)

    res = st.session_state.get('result')
    if res is None:
        st.info('Press Train model to fit the pipeline on your data.')
    else:
        m1, m2, m3 = st.columns(3)
        m1.metric('Test accuracy', f"{res['accuracy']:.0%}")
        m2.metric('Training rows', res['rows'][0])
        m3.metric('Test rows', res['rows'][1])
        st.subheader('Actual and predicted')
        st.dataframe(res['table'])
        st.subheader('What pushed the model toward Yes')
        st.bar_chart(res['coefs'])

with tab_predict:
    res = st.session_state.get('result')
    if res is None:
        st.info('Train a model first, then come back to predict.')
    else:
        c1, c2 = st.columns(2)
        age = c1.number_input('Age', 18, 90, 30)
        income = c2.number_input('Annual income', 0, 10_000_000, 50000, step=1000)
        gender = c1.selectbox('Gender', sorted(df['Gender'].unique()))
        city = c2.selectbox('City', sorted(df['City'].unique()))
        education = c1.selectbox('Education', pl.ORDINAL_ORDER[0])
        membership = c2.selectbox('Membership', pl.ORDINAL_ORDER[1])

        if st.button('Predict purchase', type='primary'):
            row = pd.DataFrame([{
                'Age': age, 'Annual_Income': income, 'Gender': gender, 'City': city,
                'Education': education, 'Membership': membership,
            }])[pl.FEATURES]
            model = res['model']
            proba = model.predict_proba(row)[0]
            label = model.predict(row)[0]
            p_yes = proba[list(model.classes_).index('Yes')]
            st.metric('Prediction', label, f'{p_yes:.0%} chance of Yes', delta_color='off')
            pre = model.named_steps['pre']
            st.subheader('What the model received')
            st.dataframe(pd.DataFrame(pre.transform(row), columns=pre.get_feature_names_out()))
