import numpy as np
import pandas as pd
import utils.utils as utils
from data.data_structures import GSM_Dataset
from sklearn.preprocessing import OneHotEncoder
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, FunctionTransformer, MinMaxScaler
from data.data_structures import GSM_Dataset
from sklearn.model_selection import train_test_split

def load_data_real(config):
    file_path = utils.get_project_path() + "/data/data_covid19/Merged_panel_data.csv"
    data_raw = pd.read_csv(file_path)
    data_filtered = data_raw[["canton_codes", "canton_pop", "date", "wd", "eth_ban_5", "total_trips_2020", "bag_falle"]].rename(columns={'eth_ban_5': 'a', 'total_trips_2020': 'm', 'bag_falle': 'y'})
    #data_filtered['sunday'] = data_filtered['sunday'].fillna(0).astype(bool).astype(int)
    data_filtered = data_filtered.dropna(subset=['canton_pop'])
    data_filtered['a'] = data_filtered['a'].astype(int)
    #data_filtered['cl_border'] = data_filtered['cl_border'].astype(int)

    #Check for mondays
    data_filtered['wd'] = (data_filtered['wd'] == 'Monday').astype(int)

    #Drop the first 10 days (infections from other regions)
    data_filtered = data_filtered.groupby("canton_codes").apply(lambda x: x.iloc[10:])
    data_filtered.reset_index(drop=True, inplace=True)

    #One hot encode canton codes
    one_hot_encoded = pd.get_dummies(data_filtered['canton_codes'])
    data_filtered = pd.concat([data_filtered, one_hot_encoded], axis=1)
    data_filtered.drop('canton_codes', axis=1, inplace=True)

    #Take outcomes from 10 days in the future (infection delays)
    grouped = data_filtered.groupby('canton_pop')
    data_filtered['y'] = grouped['y'].shift(-10)
    data_filtered.loc[grouped.tail(10).index, 'y'] = np.nan
    data_filtered = data_filtered.dropna(subset=['y'])

    #Impute missing mediators
    imputer = IterativeImputer(random_state=0)
    data_filtered['m'] = imputer.fit_transform(data_filtered[['m']])

    #Threshold mediators
    data_filtered['m'] = (data_filtered['m'] > data_filtered['m'].median()).astype(int)
    #Create GMSM_Dataset

    #Standardize canton population
    scaler = StandardScaler()
    data_filtered['canton_pop'] = scaler.fit_transform(data_filtered[['canton_pop']])

    # Shuffle the DataFrame
    shuffled_data = data_filtered.sample(frac=1, random_state=42)
    #Train validation split
    if config["validation"]:
        # Split the shuffled DataFrame into train and validation sets
        train_data, val_data = train_test_split(shuffled_data, test_size=1 - config["train_val_ratio"], random_state=42)
    else:
        train_data = shuffled_data
        val_data = None

    #data_filtered.hist(figsize=(10, 8))
    #plt.tight_layout()
    #plt.show()
    x_train = train_data.copy()
    x_train.drop(['a', "m", "y", "date"], axis=1, inplace=True)
    d_train = GSM_Dataset(x=x_train.values, a=train_data['a'].values.reshape(-1, 1),
                          m={"m1": train_data['m'].values.reshape(-1, 1)}, y=train_data['y'].values.reshape(-1, 1),
                          x_type="continuous", a_type="binary", y_type="continuous")
    if config["validation"]:
        x_val = val_data.copy()
        x_val.drop(['a', "m", "y", "date"], axis=1, inplace=True)
        d_val = GSM_Dataset(x=x_val.values, a=val_data['a'].values.reshape(-1, 1),
                          m={"m1": val_data['m'].values.reshape(-1, 1)}, y=val_data['y'].values.reshape(-1, 1),
                          x_type="continuous", a_type="binary", y_type="continuous")
    else:
        d_val = None
    causal_graph = {"nodes": ["a", "m1", "y"],
                    "edges": [("a", "m1"), ("a", "y"), ("m1", "y")]}

    return {"d_train": d_train, "d_val": d_val, "causal_graph": causal_graph}