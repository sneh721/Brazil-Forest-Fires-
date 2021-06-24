"""
Gracie Gibbons
Machine Learning model
ML plots the trend of fires in Brazil between 1998 and 2017 in its function
trend() and develops a machine learning model to predict the trend of
forest fires in Brazil from the years 2018 to 2023.
"""


import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from pycaret.regression import predict_model
from pycaret.regression import finalize_model
from pycaret.regression import compare_models
from pycaret.regression import setup


def trend(fires):
    '''
    Trend() will generate a visualization of the trend of forest
    fires in Brazil over the years 1998-2017. Both a static image
    of the visualization is generates as well as an interactive version
    that indicates the exact dates of each plotted fire.
    '''
    fires_per_year = fires.groupby('date', as_index=False)['number'].sum()
    fig, ax = plt.subplots(figsize=(15, 7))
    fig = px.scatter(fires_per_year, x='date', y='number', trendline='ols',
                     template='plotly_dark')
    fig.update_layout(
        title='Trend of Amazon Fires Over Time',
        yaxis_title='Number of Fires',
        xaxis_title='Time'
    )
    fig.show()
    fig.write_image(
        'scatter_fires.png'
        )


def fit_model(fires):
    '''
    fit_model() uses the pycaret library to predict the trend of fires
    in the years 2018 to 2023. Both a static image of the model fit and
    prediction are generated as well as an interactive version of the
    visualizations that will indicate the exact date of each fire as
    well as the number of fires on that day.
    '''
    # training model
    fires = fires.groupby('date', as_index=False)['number'].sum()
    setup(data=fires, data_split_shuffle=False,
          target='number', fold=3, fold_strategy='timeseries')
    best = compare_models(sort='MAE')
    predictions = predict_model(best, data=fires)
    fig = px.line(predictions, x='date', y=["number", "Label"],
                  template='plotly_dark')
    fig.add_vrect(x0="2014-0-1", x1="2017-11-01", fillcolor="grey",
                  opacity=0.25, line_width=0)
    fig.update_layout(
        title='Prediction on Test Set of Amazon Fires Over Time',
        yaxis_title='Number of Fires',
        xaxis_title='Time'
    )
    fig.show()
    fig.write_image(
        'fit_fires.png'
        )
    # training best model on entire dataset
    final_best = finalize_model(best)
    # predicting fires
    future_dates = pd.date_range(start='2018-01-01', end='2023-01-01',
                                 freq='MS')
    future_df = pd.DataFrame()
    future_df['month'] = [i.month for i in future_dates]
    future_df['year'] = [i.year for i in future_dates]
    future_df['date'] = [i for i in future_dates]
    predictions_future = predict_model(final_best, data=future_df)
    concat_df = pd.concat([fires, predictions_future], axis=0)
    concat_df_i = pd.date_range(start='1998-02-01', end='2023-01-01',
                                freq='MS')
    concat_df.set_index(concat_df_i, inplace=True)
    fig = px.line(concat_df, x=concat_df.index, y=["number", "Label"],
                  template='plotly_dark')
    fig.update_layout(
        title='Prediction of Amazon Fires From 2018 to 2023',
        yaxis_title='Number of Fires',
        xaxis_title='Time'
    )
    fig.show()
    fig.write_image('D:/Textbooks/cse163/Project/predict_fires.png')


def test_trend(fires):
    fires = fires.groupby('date', as_index=False)['number'].sum()
    fires1998 = fires[fires['date'] == '1998-01-01']
    fires2003 = fires[fires['date'] == '2003-06-01']
    fires2008 = fires[fires['date'] == '2008-08-01']
    assert int(fires1998['number']) == 0
    assert int(fires2003['number']) == 6506
    assert int(fires2008['number']) == 14528
