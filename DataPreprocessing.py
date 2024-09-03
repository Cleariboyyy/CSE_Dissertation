import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# 读取日志数据
df = pd.read_csv('windows_security_log.csv')

# 示例数据处理（假设日志包含时间戳、事件ID、事件描述等）
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['hour'] = df['timestamp'].dt.hour
df['day_of_week'] = df['timestamp'].dt.dayofweek

# 编码分类特征
label_encoder = LabelEncoder()
df['event_id'] = label_encoder.fit_transform(df['event_id'])
df['event_description'] = label_encoder.fit_transform(df['event_description'])

# 选择特征和标签
X = df[['hour', 'day_of_week', 'event_id', 'event_description']]
y = df['threat_level']  # 假设存在threat_level列，表示威胁级别

# 标准化特征
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)