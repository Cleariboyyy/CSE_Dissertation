import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import os
import csv
from Evtx.Evtx import Evtx
from Evtx.Views import evtx_file_xml_view
import xml.etree.ElementTree as ET
from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization, Add


# 自定义TransformerLayer类
@tf.keras.utils.register_keras_serializable()
class TransformerLayer(tf.keras.layers.Layer):
    def __init__(self, num_heads, key_dim, **kwargs):
        super(TransformerLayer, self).__init__(**kwargs)
        self.mha = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)
        self.layernorm = LayerNormalization(epsilon=1e-6)
        self.add = Add()

    def call(self, inputs):
        attn_output = self.mha(inputs, inputs)
        out = self.add([inputs, attn_output])
        return self.layernorm(out)

    def get_config(self):
        config = super(TransformerLayer, self).get_config()
        config.update({
            "num_heads": self.mha.num_heads,
            "key_dim": self.mha.key_dim,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


# 加载模型、编码器和标准化器
model = tf.keras.models.load_model('TrainingResults/3/threat_classification_model.keras',
                                   custom_objects={'TransformerLayer': TransformerLayer})
label_encoder = joblib.load('TrainingResults/3/label_encoder.pkl')
scaler = joblib.load('TrainingResults/3/scaler.pkl')


# 解析evtx文件并转换为CSV
def parse_evtx_to_csv(evtx_file, output_csv):
    with Evtx(evtx_file) as log:
        with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['SystemTime', 'EventID', 'Keywords']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            record_count = 0
            valid_record_count = 0
            for record in evtx_file_xml_view(log):
                record_count += 1
                xml_str = record[0]  # 使用第一个元素作为 XML 数据

                root = ET.fromstring(xml_str)

                time_created_elem = root.find(".//{http://schemas.microsoft.com/win/2004/08/events/event}TimeCreated")
                if time_created_elem is not None:
                    system_time = time_created_elem.attrib.get('SystemTime', 'Unknown')
                else:
                    system_time = 'Unknown'

                event_id_elem = root.find(".//{http://schemas.microsoft.com/win/2004/08/events/event}EventID")
                event_id = event_id_elem.text if event_id_elem is not None else 'UnknownEvent'

                keywords_elem = root.find(".//{http://schemas.microsoft.com/win/2004/08/events/event}Keywords")
                keywords = keywords_elem.text if keywords_elem is not None else 'NoKeywords'

                if event_id != 'UnknownEvent' and keywords != 'NoKeywords':
                    valid_record_count += 1
                    writer.writerow({
                        'SystemTime': system_time,
                        'EventID': event_id,
                        'Keywords': keywords
                    })
            print(f"Total records processed: {record_count}")
            print(f"Valid records written to CSV: {valid_record_count}")


# 加载和预处理单个CSV文件
def load_and_preprocess(file_path):
    df = pd.read_csv(file_path)
    print(f"Initial data shape: {df.shape}")

    df = df[(df['EventID'] != 'UnknownEvent') & (df['Keywords'] != 'NoKeywords')]
    print(f"Data shape after removing unknowns: {df.shape}")

    if df.empty:
        return df, None, None

    df = df.sort_values(by='SystemTime')

    # 如果编码过程中遇到未知标签，将其映射到 "Other"
    other_class_eventid = len(label_encoder.classes_)
    other_class_keywords = len(label_encoder.classes_) + 1

    df['EventID'] = df['EventID'].apply(
        lambda x: label_encoder.transform([x])[0] if x in label_encoder.classes_ else other_class_eventid)
    df['Keywords'] = df['Keywords'].apply(
        lambda x: label_encoder.transform([x])[0] if x in label_encoder.classes_ else other_class_keywords)

    print(f"Data shape after encoding: {df.shape}")
    return df, other_class_eventid, other_class_keywords


# 创建滑动窗口
def create_sliding_window(X, window_size):
    Xs = []
    for i in range(len(X) - window_size + 1):
        Xs.append(X[i:i + window_size])
    return np.array(Xs)


# 填充数据函数
def pad_data(X, window_size):
    pad_size = window_size - len(X)
    if pad_size > 0:
        X_padded = np.pad(X, ((0, pad_size), (0, 0)), mode='constant', constant_values=0)
        X_window = np.expand_dims(X_padded, axis=0)
        X_window = np.expand_dims(X_window, axis=3)
    else:
        X_window = create_sliding_window(X, window_size)
        X_window = np.expand_dims(X_window, axis=3)
    return X_window


# 执行预测
def predict_threat(evtx_file):
    temp_csv = 'temp_output.csv'
    parse_evtx_to_csv(evtx_file, temp_csv)

    df, other_class_eventid, other_class_keywords = load_and_preprocess(temp_csv)

    if df.empty:
        print("Error: No valid data found in the provided EVTX file.")
        return

    features = ['EventID', 'Keywords']
    X = df[features]

    X_scaled = scaler.transform(X)

    window_size = 10
    if len(X_scaled) < window_size:
        print(f"Warning: Not enough data for sliding window. Padding data.")
        X_window = pad_data(X_scaled, window_size)
    else:
        X_window = create_sliding_window(X_scaled, window_size)
        X_window = np.expand_dims(X_window, axis=3)

    if X_window.size == 0:
        print("Error: Insufficient data for sliding window creation.")
        return

    y_pred_probs = model.predict(X_window)

    # 计算最可能的分类
    y_pred_labels = np.argmax(np.mean(y_pred_probs, axis=0))

    # 检查预测结果是否为 "Other" 类别
    final_prediction = label_encoder.inverse_transform([y_pred_labels])[0] if y_pred_labels < other_class_eventid else 'Other'

    print(f"Final Predicted Threat Label: {final_prediction}")

    os.remove(temp_csv)


if __name__ == "__main__":
    evtx_file = 'EVTX-ATTACK-SAMPLES-master/Defense Evasion/DE_BYOV_Zam64_CA_Memdump_sysmon_7_10.evtx'
    predict_threat(evtx_file)
