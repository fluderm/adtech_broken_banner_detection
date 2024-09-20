import argparse
from itertools import product
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import pickle as pk
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

if __name__ == '__main__':
    # accepts commandline arguments

    parser = argparse.ArgumentParser(description='Process grid ID and number of components.')

    # Add arguments
    parser.add_argument('grid_id', type=int, help='The grid ID')
    parser.add_argument('n_components', type=int, help='The number of components')
    parser.add_argument('data_directory', type=str, help='Directory of the data that you want to preprocess')

    # Parse the arguments
    args = parser.parse_args()

    # Now you can use args.grid_id and args.n_components in your script
    grid_id = args.grid_id
    n_components = args.n_components
    data_directory = args.data_directory
    print("Grid ID: ", grid_id)
    print("Nb components: ", n_components)
    print("Data directory: ", data_directory)

    data = pd.read_csv(data_directory)


    scale = StandardScaler()

    cb_333519 = ['ID_1184', 'ID_1281', 'ID_1305', 'ID_1353', 'ID_1448', 'ID_1522',
        'ID_1544', 'ID_162', 'ID_1682', 'ID_1690', 'ID_1824', 'ID_1888',
        'ID_1929', 'ID_2076', 'ID_2097', 'ID_2226', 'ID_2249', 'ID_2268',
        'ID_2331', 'ID_2339', 'ID_2386', 'ID_2396', 'ID_2438', 'ID_258',
        'ID_2609', 'ID_2680', 'ID_2863', 'ID_2883', 'ID_2908', 'ID_3061',
        'ID_3243', 'ID_3250', 'ID_3314', 'ID_3382', 'ID_3397', 'ID_3402',
        'ID_3420', 'ID_3459', 'ID_3470', 'ID_3540', 'ID_398', 'ID_409',
        'ID_484', 'ID_489', 'ID_526', 'ID_549', 'ID_580', 'ID_665',
        'ID_810', 'ID_84', 'ID_857', 'ID_86', 'ID_905', 'ID_927', 'ID_934',
        'ID_962', 'ID_986']

    cb_333346 = ['ID_1247', 'ID_162', 'ID_2534', 'ID_2742', 'ID_526', 'ID_2201',
        'ID_1165', 'ID_743', 'ID_199', 'ID_2145', 'ID_2569', 'ID_643',
        'ID_1305', 'ID_3180', 'ID_3158', 'ID_136', 'ID_293', 'ID_1753',
        'ID_1849', 'ID_2226', 'ID_1462', 'ID_626', 'ID_2863', 'ID_3243',
        'ID_3250', 'ID_1708', 'ID_1238', 'ID_580', 'ID_84', 'ID_2568',
        'ID_2340', 'ID_1803', 'ID_3470', 'ID_139', 'ID_2619', 'ID_2908',
        'ID_1281', 'ID_3308', 'ID_2883', 'ID_1320', 'ID_1333', 'ID_1062',
        'ID_149', 'ID_260', 'ID_599', 'ID_1513', 'ID_3402', 'ID_1888',
        'ID_2972', 'ID_398', 'ID_2339', 'ID_1030', 'ID_3382', 'ID_2076',
        'ID_1646', 'ID_1077', 'ID_10', 'ID_1153', 'ID_1533', 'ID_2609',
        'ID_1214', 'ID_810', 'ID_2097', 'ID_2386', 'ID_1585', 'ID_549',
        'ID_3420', 'ID_3397', 'ID_2006', 'ID_561', 'ID_1542', 'ID_1570',
        'ID_1819', 'ID_1413', 'ID_1240', 'ID_793', 'ID_831', 'ID_3188',
        'ID_1437', 'ID_556', 'ID_1134', 'ID_314', 'ID_2836', 'ID_1483',
        'ID_2676', 'ID_665', 'ID_15', 'ID_2755', 'ID_2194', 'ID_1682',
        'ID_2775', 'ID_1923', 'ID_1846', 'ID_1262', 'ID_3314', 'ID_1448',
        'ID_1770', 'ID_2626', 'ID_3296', 'ID_489', 'ID_1963', 'ID_1268',
        'ID_1568', 'ID_1094', 'ID_2267', 'ID_409', 'ID_1650', 'ID_1522',
        'ID_927', 'ID_2302', 'ID_1929', 'ID_1435', 'ID_1353']

    # Labelling points with ground truth
    if grid_id == 333346:
        cb = cb_333346
    elif grid_id == 333519:
        cb = cb_333519
    else: 
        raise ValueError("grid_id needs to be 333346 or 333519")

    # %%

    # %%
    # expand table so that each row corresponds to 1 click:

    data_heatmap_expanded = data.loc[data.index.repeat(data['clicks'])].reset_index(drop=True)
    data_heatmap_expanded['clicks'] = 1

    click_stat = data_heatmap_expanded.groupby(['click_x','click_y'])['clicks'].count().reset_index()

    # %%
    # Binning clicks in nr_of_x_bins, nr_of_y_bins:

    NR_OF_X_BINS = 61
    NR_OF_Y_BINS = 51

    max_width = data['display_width'].max()
    max_height = data['display_height'].max()

    width_bins = np.linspace(1, max_width, NR_OF_X_BINS)
    height_bins = np.linspace(1, max_height, NR_OF_Y_BINS)

    width_bins_max = len(width_bins)-2 # start at 0
    height_bins_max = len(height_bins)-2

    data_heatmap_expanded['click_x_bin'] = pd.cut(data_heatmap_expanded['click_x'], 
                                        bins = width_bins, 
                                        labels=False, 
                                        include_lowest=True)

    data_heatmap_expanded['click_y_bin'] = pd.cut(data_heatmap_expanded['click_y'], 
                                        bins=height_bins, 
                                        labels=False, 
                                        include_lowest=True)


    aggregated_clicks = data_heatmap_expanded.groupby(['grid_id',
                                            'domain', 
                                            'click_x_bin', 
                                            'click_y_bin']).size().reset_index(name='clicks_sum')


    #aggregated_clicks['clicks_sum'].astype('int64');

    # %%
    # More binning: add empty bins --> easier to generate vectors
    # takes a few seconds

    domains_grids = data[['domain', 'grid_id']].drop_duplicates()

    aux = pd.DataFrame(list(product(range(0,width_bins_max+1), 
                                    range(0,height_bins_max+1))), 
                    columns=['click_x_bin', 'click_y_bin'])

    domains_grids['key'] = 1
    aux['key'] = 1

    expanded_set = pd.merge(domains_grids, aux, on='key').drop('key', axis=1)

    data_binned = pd.merge(expanded_set, aggregated_clicks, 
                    on = ['domain', 'grid_id', 'click_x_bin', 'click_y_bin'], 
                    how = 'left').fillna(0)
    data_binned['clicks_sum'] = data_binned['clicks_sum'].astype('int64')
    # data_binned = data_binned[data_binned['grid_id'] == 333519]

    # %%
    data_binned = data_binned[data_binned['grid_id'] == grid_id]

    # %%
    # create a wide-format DataFrame such that each row corresponds to one heatmap
    pivot_df = pd.pivot_table(data_binned, values='clicks_sum', index=['domain', 'grid_id'],
                            columns=['click_x_bin', 'click_y_bin'], aggfunc='sum', fill_value=0)

    # Reset index to make 'domain' and 'grid_id' regular columns
    pivot_df.reset_index(inplace=True)

    # Rename columns to match the desired format
    pivot_df.columns = ['domain', 'grid_id'] + [f'clicks_sum_for_{x}_{y}' for x, y in pivot_df.columns[2:]]

    # Uncomment to view pivot_df
    #print(pivot_df)

    # %%
    # Version 1 of normalizing
    # Normalizing such that the total number of clicks in one heatmap sums to one. We do not standardize because standardization assumes the clicks follow a normal distribution, which is not the case
    input = pivot_df.copy()

    normalized_input = input.copy()

    normalized_input.iloc[:,2:] = (input.iloc[:,2:].transpose()/input.iloc[:,2:].sum(axis=1)).transpose()

    normalized_input.head()

    # %%
    normalized_input['label'] = 0
    normalized_input.loc[normalized_input.domain.isin(cb),'label'] = 1
    # normalized_input.head(10)
    X_train = normalized_input.iloc[:,2:-1]
    y_train = normalized_input.iloc[:,-1]

    # %%
    features = normalized_input.columns[2:] #input.columns[1:]  # Exclude the first column, which is the index

    pipeline = Pipeline([('scaling', StandardScaler()), ('pca', PCA(n_components=n_components))])

    # Fit and transform the data
    pca_result = pipeline.fit_transform(normalized_input[features])

    # Create a DataFrame with the PCA results
    pca_df = pd.DataFrame(data=pca_result, columns=[f'PC{i+1}' for i in range(n_components)])

    # Concatenate the first column (index) from pca_input to pca_df
    pca_df['domain'] = normalized_input['domain']

    pca_df['label'] = 0

    pca_df.loc[pca_df.domain.isin(cb),'label'] = 1

    # %%
    pca_df['label'] = 0
    pca_df.loc[pca_df.domain.isin(cb),'label'] = 1
    pca_df['label'].sum()

    # %%
    # Plotting pca_df with ground truth labels
    pca_df1 = pca_df
    plt.figure(figsize=(10, 6))
    plt.scatter(pca_df1['PC1'], pca_df1['PC2'], c=pca_df1['label'], cmap='viridis', alpha=0.75)
    plt.title(f'{grid_id}_PCA scatterplot all-all')
    plt.xlabel('Principal Component 1 (PC1)')
    plt.ylabel('Principal Component 2 (PC2)')
    plt.colorbar(label='Index')

    plt.savefig('temp/pca_df.jpg')
    pca_df.to_csv('temp/pca_df.csv')
    normalized_input.to_csv('temp/normalized_input.csv')