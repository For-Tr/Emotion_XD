import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D, BatchNormalization
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras import regularizers
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 读取数据
Catch_bat = pd.read_csv('fer2013.csv')

X_train, Y_train, X_test, Y_test = [], [], [], []

for index, row in Catch_bat.iterrows():
    val = row['pixels'].split(' ')
    try:
        if 'Training' in row['Usage']:
            X_train.append(np.array(val, 'float32'))
            Y_train.append(row['emotion'])
        elif 'PublicTest' in row['Usage']:
            X_test.append(np.array(val, 'float32'))
            Y_test.append(row['emotion'])
    except:
        pass

# 转化为 np.array
X_train = np.array(X_train, 'float32')
Y_train = np.array(Y_train, 'float32')
X_test = np.array(X_test, 'float32')
Y_test = np.array(Y_test, 'float32')

# 计算训练集全局均值和标准差
TRAIN_MEAN = np.mean(X_train)
TRAIN_STD = np.std(X_train)
print(f"训练集全局均值: {TRAIN_MEAN:.4f}, 标准差: {TRAIN_STD:.4f}")

# 保存到本地，方便预测阶段使用
np.save("train_mean.npy", TRAIN_MEAN)
np.save("train_std.npy", TRAIN_STD)

# 数据标准化
X_train = (X_train - TRAIN_MEAN) / TRAIN_STD
X_test = (X_test - TRAIN_MEAN) / TRAIN_STD

# 标签 one-hot 编码
labels = 7
Y_train = to_categorical(Y_train, num_classes=labels)
Y_test = to_categorical(Y_test, num_classes=labels)

# 调整形状
width, height = 48, 48
X_train = X_train.reshape(X_train.shape[0], width, height, 1)
X_test = X_test.reshape(X_test.shape[0], width, height, 1)

features = 64
batch_size = 64
epochs = 80

print("创建数据增强器...")

# 2. 自定义无限生成器
def create_infinite_generator(X, y, batch_size=64):
    datagen = ImageDataGenerator(
        horizontal_flip=True,      # 水平翻转
        rotation_range=10,         # 旋转±10度
        width_shift_range=0.1,     # 水平平移 ±10%
        height_shift_range=0.1,    # 垂直平移 ±10%
        brightness_range=[0.8,1.2],# 亮度变化 0.8~1.2
        fill_mode='nearest'        # 填充方式
    )

    generator = datagen.flow(X, y, batch_size=batch_size, shuffle=True)

    while True:
        try:
            batch_x, batch_y = next(generator)
            yield batch_x, batch_y
        except StopIteration:
            generator = datagen.flow(X, y, batch_size=batch_size, shuffle=True)
            batch_x, batch_y = next(generator)
            yield batch_x, batch_y

train_generator = create_infinite_generator(X_train, Y_train, batch_size)
print("数据增强设置：水平翻转 + 旋转 + 平移 + 亮度变化")

# 3. 构建 CNN
model = Sequential()

# 第一部分
model.add(Conv2D(features, kernel_size=(3, 3), activation='relu',
                 input_shape=X_train.shape[1:],
                 kernel_regularizer=regularizers.l2(0.0005)))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Conv2D(features, kernel_size=(3, 3), activation='relu',
                 kernel_regularizer=regularizers.l2(0.0005)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.3))

# 第二部分
model.add(Conv2D(features, kernel_size=(3, 3), activation='relu',
                 kernel_regularizer=regularizers.l2(0.0005)))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Conv2D(features, kernel_size=(3, 3), activation='relu',
                 kernel_regularizer=regularizers.l2(0.0005)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.3))

# 第三部分
model.add(Conv2D(2*features, kernel_size=(3, 3), activation='relu',
                 kernel_regularizer=regularizers.l2(0.0005)))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Conv2D(2*features, kernel_size=(3, 3), activation='relu',
                 kernel_regularizer=regularizers.l2(0.0005)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# 全连接层
model.add(Flatten())
model.add(Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
model.add(BatchNormalization())
model.add(Dropout(0.3))

# 输出层
model.add(Dense(labels, activation='softmax'))

model.summary()

# 4. 编译模型
model.compile(
    loss=categorical_crossentropy,
    optimizer=Adam(learning_rate=0.0003),
    metrics=['accuracy']
)

# 5. 回调函数
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, min_delta=0.001, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_delta=0.001, min_lr=0.00001)
]

# 6. 训练模型
steps_per_epoch = len(X_train) // batch_size
print(f"开始训练，每轮步数: {steps_per_epoch}")

history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=epochs,
    validation_data=(X_test, Y_test),
    callbacks=callbacks
)

# 7. 保存模型
fer_json = model.to_json()
with open('fer.json', 'w') as json_file:
    json_file.write(fer_json)

model.save_weights('fer.weights.h5')

print("="*50)
print("训练完成！")
print(f"模型已保存: 'fer.json' 和 'fer.weights.h5'")
print("="*50)

# 8. 模型效果评估
# 情绪类别
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

print("\n开始评估模型在测试集上的性能...")

# 预测测试集
Y_pred_prob = model.predict(X_test, batch_size=64)
Y_pred = np.argmax(Y_pred_prob, axis=1)
Y_true = np.argmax(Y_test, axis=1)

# 分类报告
print("\n===== 分类性能报告 =====")
print(classification_report(Y_true, Y_pred, target_names=emotion_labels))

# 混淆矩阵
cm = confusion_matrix(Y_true, Y_pred)

print("\n===== 混淆矩阵 =====")
print(cm)

# 归一化混淆矩阵
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

# 可视化混淆矩阵
plt.figure(figsize=(8,6))
sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=emotion_labels,
            yticklabels=emotion_labels)

plt.title("FER2013 Emotion Confusion Matrix (Normalized)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=300)
plt.show()

print("\n混淆矩阵图已保存为 confusion_matrix.png")
print("="*50)
