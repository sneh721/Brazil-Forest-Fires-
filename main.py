"""
Sneh Gupta and Gracie Gibbons
Brazil Forest Fires Project
Visualizing forest fires from 1998 to 2017 in Brazil via
geospatial visualizations and regression line.
Also making predictive machine learning models.
"""

import geopandas as gpd
import data_visualization as dv
import ml


def main():
    url = 'https://raw.githubusercontent.com/deltalite/Brazil-Forest-Fires\
/master/data/amazon_with_date.csv'
    fires = dv.process(url)
    states = gpd.read_file('D:/Textbooks/cse163/Project/br-states.json')
    merged, states = dv.merge_new(fires, states)
    dv.graph_tot(merged, states)
    dv.time_lapse(merged, states)
    print(merged['number'].max())
    ml.trend(fires)
    ml.fit_model(fires)
    ml.test_trend(fires)


if __name__ == '__main__':
    main()
