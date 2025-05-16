import os

# Конфигурация структуры проекта
project_structure = {
    "data": {
        "__files__": ["iris.csv"]
    },
    "src": {
        "clustering": {
            "__files__": [
                "single_linkage.py",
                "hierarchical.py",
                "maxmin_distance.py",
                "isodata.py",
                "cure.py",
                "forel.py"
            ]
        },
        "features": {
            "__files__": [
                "compactness.py",
                "spread.py",
                "del_method.py",
                "add_method.py",
                "spa_method.py"
            ]
        },
        "metrics": {
            "__files__": [
                "internal_indices.py",
                "external_indices.py"
            ]
        },
        "distances": {
            "__files__": [
                "euclidean.py",
                "pearson.py",
                "chebyshev.py",
                "minkowski.py"
            ]
        },
        "__files__": [
            "utils.py",
            "pipeline.py"
        ]
    },
    "notebooks": {
        "__files__": ["main.ipynb"]
    },
    "requirements.txt": "",
    "config.yaml": ""
}


def create_structure(base_path, structure):
    for name, content in structure.items():
        path = os.path.join(base_path, name)

        # Если это файл (не словарь)
        if not isinstance(content, dict):
            with open(path, "w", encoding="utf-8") as f:
                if name.endswith(".py"):
                    f.write("# TODO: Implement this module\n")
                elif name == "requirements.txt":
                    f.write("numpy\nscikit-learn\nmatplotlib\ncure-cluster\npandas\nPyYAML\n")
                elif name == "config.yaml":
                    f.write("# Configuration file for clustering pipeline\n")
                else:
                    f.write(str(content))

        else:
            # Создаем директорию
            os.makedirs(path, exist_ok=True)

            # Обрабатываем файлы в __files__
            files_to_create = content.pop("__files__", [])
            for filename in files_to_create:
                file_path = os.path.join(path, filename)
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write("# TODO: Implement this module\n")

            # Рекурсивно обрабатываем остальные вложенные структуры
            if content:
                create_structure(path, content)


if __name__ == "__main__":
    project_name = "clustering_project"
    print(f"Создание структуры проекта в директории: {project_name}")
    create_structure(project_name, project_structure)
    print("✅ Структура проекта успешно создана!")