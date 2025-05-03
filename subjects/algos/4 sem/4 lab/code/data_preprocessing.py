import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

def preprocess_data(datasets: list[pd.DataFrame]) -> list[pd.DataFrame]:
    processed_datasets = []
    
    ohe_train = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    ohe_stations = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    le_seat = LabelEncoder()
    
    all_trains = []
    all_from_stations = []
    all_to_stations = []
    all_seats = []
    
    for df in datasets:
        all_trains.extend(df['train_number'].astype(str).values)
        all_from_stations.extend(df['from_station'].astype(str).values)
        all_to_stations.extend(df['to_station'].astype(str).values)
        all_seats.extend(df['wagon_seat'].astype(str).str.replace('1-', '', regex=False).str.strip().values)
    
    ohe_train.fit(pd.DataFrame({'train_number': all_trains}))
    ohe_stations.fit(pd.DataFrame({'from_station': all_from_stations, 'to_station': all_to_stations}))
    le_seat.fit(all_seats)
    
    for df in datasets:
        df = df.copy()
        
        df['full_name'] = df['full_name'].apply(lambda x: hash(x) & 0xffffffff)
        df['pasport_number'] = df['pasport_number'].astype(str).str.replace(' ', '').apply(lambda x: hash(x) & 0xffffffff)
        df['card_number'] = df['card_number'].astype(str).apply(lambda x: hash(x) & 0xffffffff)
        
        df['wagon_seat'] = le_seat.transform(
            df['wagon_seat'].astype(str).str.replace('1-', '', regex=False).str.strip()
        )
        
        train_encoded = ohe_train.transform(pd.DataFrame({'train_number': df['train_number'].astype(str)}))
        train_encoded_df = pd.DataFrame(
            train_encoded,
            columns=[f'train_number_{cat}' for cat in ohe_train.categories_[0]],
            index=df.index
        )
        
        stations_encoded = ohe_stations.transform(pd.DataFrame({
            'from_station': df['from_station'].astype(str),
            'to_station': df['to_station'].astype(str)
        }))
        stations_encoded_df = pd.DataFrame(
            stations_encoded,
            columns=[f'from_{cat}' for cat in ohe_stations.categories_[0]] + [f'to_{cat}' for cat in ohe_stations.categories_[1]],
            index=df.index
        )
        
        df['price'] = df['price'].astype(str).str.replace('руб', '', regex=False).str.strip()
        df['price'] = pd.to_numeric(df['price'], errors='coerce')
        df['departure'] = pd.to_datetime(df['departure']).astype('int64') // 10**9
        df['arrival'] = pd.to_datetime(df['arrival']).astype('int64') // 10**9
        
        df = df.drop(['train_number', 'from_station', 'to_station'], axis=1)
        df = pd.concat([df, train_encoded_df, stations_encoded_df], axis=1)
        
        processed_datasets.append(df)
    
    return processed_datasets
