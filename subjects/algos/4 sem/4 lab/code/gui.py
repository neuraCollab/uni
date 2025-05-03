import gradio as gr
from data_loading import read_csv_by_pd
from data_preprocessing import preprocess_data
from evaluation import evaluate_imputation_methods

methods = [
    "mean", "median", "mode", "ffill", "hot_deck",
    "linear_regression", "stochastic_regression", "spline"
]

def process_file(file):
    datasets = read_csv_by_pd([file.name], info=True, head=True)
    datasets = preprocess_data(datasets)
    result_df = evaluate_imputation_methods(datasets[0], methods)
    return result_df

iface = gr.Interface(fn=process_file,
                     inputs=gr.File(label="Выберите CSV-файл"),
                     outputs=gr.Dataframe(label="Результаты иммутации"),
                     title="Оценка методов заполнения пропусков")

iface.launch()
