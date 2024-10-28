import os

common_import = 'import my_cnn_models.cnn_util as cnnu'

def create_model_source(version: int, names: list[str], num_classes: list[int], dataset_paths: list[str], model_dst_path: str, epoch: int):
    folder_name = f'v{version}_model_source'
    sources = [f"""{common_import}
models = {{
    '{name}_class_v{version}' : (cnnu.create_model_v{version}({num_class}), '{dataset_path}'),
}}
cnnu.train_models(models, '{model_dst_path}/', {epoch})
    """ for name, num_class, dataset_path in zip(names, num_classes, dataset_paths)]
    for source, name in zip(sources, names):
        file_path = f'{folder_name}/{name}_model_v{version}.py'
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w+') as fp:
            fp.write(source)
    init_path = f'{folder_name}/__init__.py'
    os.makedirs(os.path.dirname(init_path), exist_ok=True)
    with open(init_path, 'w+') as fp:
        pass

model_names = ['digit', 'paren', 'operator', 'category']
num_classes = [10, 2, 5, 3]
dataset_paths = ['dataset/digit', 'dataset/paren', 'dataset/operator', 'dataset']
create_model_source(1, model_names, num_classes, dataset_paths, '../models', 20)
create_model_source(2, model_names, num_classes, dataset_paths, '../models', 20)
create_model_source(3, model_names, num_classes, dataset_paths, '../models', 20)
create_model_source(4, model_names, num_classes, dataset_paths, '../models', 20)
