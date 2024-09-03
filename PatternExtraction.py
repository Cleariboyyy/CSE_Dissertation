import os
import pandas as pd
import json


def extract_event_data(event_data):
    try:
        return json.loads(event_data.replace("'", "\""))
    except json.JSONDecodeError:
        return {}


def process_csv_file(file_path):
    # 加载CSV文件
    data = pd.read_csv(file_path)
    headers = data.iloc[:, 0].values  # 获取属性名称
    data_values = data.iloc[:, 1:]  # 获取数据
    new_data = pd.DataFrame(data_values.values.T, columns=headers)
    return new_data


def extract_features(data):
    # 打印数据的列名以确保我们正在使用正确的列引用
    print("Columns in the DataFrame:", data.columns)

    # 提取所需的特征列，确保列名与数据中的实际列名匹配
    feature_columns = ['EventID', 'SystemTime', 'Keywords']
    for col in feature_columns:
        if col not in data.columns:
            print(f"Column {col} not found in the data.")

    if all(col in data.columns for col in feature_columns):
        features = data[feature_columns].copy()
        # features.loc[:, 'EventDataDict'] = features['EventData'].apply(extract_event_data)
        return features
    else:
        print("One or more required columns are missing. Please check the data and column names.")
        return pd.DataFrame()


def process_directory(directory):
    # 创建一个空的DataFrame来存储所有处理后的数据
    all_features = pd.DataFrame()

    # 遍历目录及其所有子目录
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.csv'):
                file_path = os.path.join(root, file)
                data = process_csv_file(file_path)
                features = extract_features(data)
                if not features.empty:
                    all_features = pd.concat([all_features, features], ignore_index=True)

    return all_features


# 指定目录路径
directory_path = 'CSVevents/TA0040-Impact'

# 处理目录并提取特征
final_features = process_directory(directory_path)

# 显示或保存结果
print(final_features.head())
final_features.to_csv('ProcessedFeatures/Other.csv', index=False, mode='a')
