import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, f1_score
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, Flatten, TimeDistributed, Input, MultiHeadAttention, LayerNormalization, Add
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping
import glob
import os

# 自定义TransformerLayer类，并使用装饰器进行注册
@tf.keras.utils.register_keras_serializable()
class TransformerLayer(tf.keras.layers.Layer):
    def __init__(self, num_heads, key_dim):
        super(TransformerLayer, self).__init__()
        self.mha = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)
        self.layernorm = LayerNormalization(epsilon=1e-6)
        self.add = Add()

    def call(self, inputs):
        attn_output = self.mha(inputs, inputs)
        out = self.add([inputs, attn_output])
        return self.layernorm(out)

# 函数：加载并预处理单个文件
def load_and_preprocess(file_path, label):
    df = pd.read_csv(file_path)
    df = df.sort_values(by='SystemTime')
    label_encoder = LabelEncoder()
    df['EventID'] = label_encoder.fit_transform(df['EventID'])
    df['Keywords'] = label_encoder.fit_transform(df['Keywords'])
    df['label'] = label
    return df

# 合并所有文件的数据并添加标签
all_data = []
directory = 'ProcessedFeatures'
for file_path in glob.glob(os.path.join(directory, '*.csv')):
    label = os.path.basename(file_path).replace('.csv', '')
    df = load_and_preprocess(file_path, label)
    all_data.append(df)

if not all_data:
    raise ValueError("No data loaded. Please check the file paths and names.")

df_all = pd.concat(all_data, ignore_index=True)
df_all = df_all.sort_values(by='SystemTime')

features = ['EventID', 'Keywords']
X = df_all[features]
y = df_all['label']

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

def create_sliding_window(X, y, window_size):
    Xs, ys = [], []
    for i in range(len(X) - window_size):
        Xs.append(X[i:i+window_size])
        ys.append(y[i+window_size])
    return np.array(Xs), np.array(ys)

window_size = 10
X_window, y_window = create_sliding_window(X_scaled, y_encoded, window_size)
X_window = np.expand_dims(X_window, axis=3)

# 划分训练集和最终测试集
X_train_val, X_test, y_train_val, y_test = train_test_split(X_window, y_window, test_size=0.2, random_state=42)

def create_model():
    input_layer = Input(shape=(window_size, X_train_val.shape[2], 1))
    conv_layer = TimeDistributed(Conv1D(filters=64, kernel_size=2, activation='relu'))(input_layer)
    max_pool_layer = TimeDistributed(MaxPooling1D(pool_size=1))(conv_layer)
    flatten_layer = TimeDistributed(Flatten())(max_pool_layer)
    lstm_layer = LSTM(64, return_sequences=True)(flatten_layer)

    transformer_layer = TransformerLayer(num_heads=2, key_dim=64)(lstm_layer)
    flatten_transformer = Flatten()(transformer_layer)
    dense_layer = Dense(128, activation='relu', kernel_regularizer=l2(0.01))(flatten_transformer)
    dropout_layer = Dropout(0.5)(dense_layer)
    dropout_layer_2 = Dropout(0.5)(dropout_layer)
    output_layer = Dense(len(np.unique(y_encoded)), activation='softmax')(dropout_layer_2)

    model = Model(inputs=input_layer, outputs=output_layer)

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


kf = KFold(n_splits=5, shuffle=True, random_state=42)

for train_index, val_index in kf.split(X_train_val):
    X_train, X_val = X_train_val[train_index], X_train_val[val_index]
    y_train, y_val = y_train_val[train_index], y_train_val[val_index]

    model = create_model()
    def scheduler(epoch, lr):
        if epoch < 10:
            return lr
        else:
            return lr * 0.99

    lr_scheduler = LearningRateScheduler(scheduler)
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    history = model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_val, y_val), callbacks=[lr_scheduler, early_stopping])

    val_loss, val_accuracy = model.evaluate(X_val, y_val)
    print(f'Validation Accuracy: {val_accuracy:.2f}, Validation Loss: {val_loss:.2f}')

# 在最终测试集上评估模型
y_test_pred = np.argmax(model.predict(X_test), axis=1)
cm = confusion_matrix(y_test, y_test_pred)

# 计算正负例
positive_true = np.diag(cm)
positive_false = np.sum(cm, axis=0) - positive_true
negative_true = np.sum(cm) - (positive_false + np.sum(cm, axis=1))
negative_false = np.sum(cm, axis=1) - positive_true

# 计算 F1 分数
f1 = f1_score(y_test, y_test_pred, average='weighted')

# 打印结果
print(f'Positive True: {positive_true}')
print(f'Positive False: {positive_false}')
print(f'Negative True: {negative_true}')
print(f'Negative False: {negative_false}')
print(f'F1 Score: {f1:.3f}')

# 保存模型
model.save('threat_classification_model.keras')
import joblib
joblib.dump(label_encoder, 'label_encoder.pkl')
joblib.dump(scaler, 'scaler.pkl')
