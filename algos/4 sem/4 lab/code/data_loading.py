import pandas as pd

def read_csv_by_pd(paths: list[str], info: bool, head: bool) -> list[pd.DataFrame]:
    """Загружает и предварительно обрабатывает датасеты"""
    csv_by_pd = []
    
    for path in paths:
        dataset = pd.read_csv("data/" + path + ".csv")
        dataset = dataset.drop(columns=["Unnamed: 0"])
        dataset = dataset.dropna()
        dataset = dataset.drop_duplicates()
        dataset = dataset.reset_index(drop=True)
        
        if info:
            print(dataset.info())
        if head:
            print(dataset.head()) 
        
        csv_by_pd.append(dataset)
    
    return csv_by_pd