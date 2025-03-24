
# # Aula 3 - Construindo um difusor

#  [markdown]
# ## Vídeo 3.1 - Adicionando de ruído


import tensorflow as tf
import keras.api as keras
from keras.api import layers
from keras.api.layers import Conv2D, LeakyReLU, Flatten, Dropout, Dense
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os 
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from tqdm.auto import trange, tqdm

# 
# Carregando o dataset Fashion MNIST
#(X_treino, y_treino), (X_teste, y_teste) = tf.keras.datasets.fashion_mnist.load_data()
# Normalizando as imagens para o intervalo [-1, 1]
#X_treino = (X_treino / 127.5) - 1.0

n_dim = 8
tamanho_imagem = (n_dim, n_dim)
canal = 1  # Imagem em escala de cinza
BUFFER_SIZE = 60000
BATCH_SIZE = 16

def normalize(image):
    image = tf.cast(image, tf.float32) / 127.5 - 1.0
    return image

dataset_path = "data/train/"
train_ds = keras.preprocessing.image_dataset_from_directory(
    dataset_path,
    label_mode=None,
    color_mode='grayscale',
    image_size=tamanho_imagem,
    batch_size=BATCH_SIZE
)

train_ds = train_ds.map(normalize)

X_treino = np.concatenate([x for x in train_ds], axis=0)


# Adicionando um canal extra para as imagens de 28x28
#X_treino = np.expand_dims(X_treino, axis=-1)

timesteps = 16    # Quantidade de passos para uma imagem ruidosa se tornar clara
time_bar = 1 - np.linspace(0, 1.0, timesteps + 1) # linspace para timesteps

# 
time_bar

# 
def cvtImg(img):
    img = img - img.min()
    img = (img / img.max())
    return img.astype(np.float32)

# 
def show_examples(x):
    x = (x + 1) / 2  # Convert [-1,1] to [0,1] for display
    num_images = x.shape[0]
    plt.figure(figsize=(10, 10))
    for i in range(10):  # Mostra no máximo 25 imagens
        plt.subplot(5, 5, i+1)
        img = cvtImg(x[i])
        plt.imshow(img.squeeze(), cmap='gray')  # Exibe como imagem em escala de cinza
        plt.axis('off')

# 
show_examples(X_treino)

# 
def forward_noise(x, t):
    a = time_bar[t]      # imagem no tempo t
    b = time_bar[t + 1]  # imagem em t + 1

    ruido = np.random.normal(size=x.shape)  # Gera máscara de ruído
    a = a.reshape((-1, 1, 1, 1))
    b = b.reshape((-1, 1, 1, 1))
    img_a = x * (1 - a) + ruido * a
    img_b = x * (1 - b) + ruido * b
    return img_a, img_b

def generate_ts(num):
    return np.random.randint(0, timesteps, size=num)

#
# Gera exemplos de treino
t = generate_ts(3)  # Gera timesteps para 25 exemplos
# a, b = forward_noise(X_treino[:3], t)
# show_examples(a)

#
# ## Vídeo 3.2 - Implementando uma U-net

# 
def block(x):
    x = layers.Conv2D(128, kernel_size=3, padding='same')(x)
    x = layers.LayerNormalization()(x)
    x = layers.Activation('relu')(x)
    return x

#
def make_model():
    x = x_input = layers.Input(shape=(8, 8, 1))
    x_ts = layers.Input(shape=(1,))
    
    # Downsample
    x = layers.Conv2D(64, 3, padding='same')(x)
    x = layers.MaxPool2D(2)(x)  # 8x8 -> 4x4
    
    # Bottleneck
    x = layers.Flatten()(x)
    x = layers.Concatenate()([x, layers.Dense(128)(x_ts)])
    x = layers.Dense(4*4*64)(x)
    x = layers.Reshape((4,4,64))(x)
    
    # Upsample
    x = layers.Conv2DTranspose(64, 3, strides=2, padding='same')(x)  # 4x4 -> 8x8
    x = layers.Conv2D(1, 1, padding='same')(x)
    
    return tf.keras.Model([x_input, x_ts], x)

train_ds = train_ds.shuffle(BUFFER_SIZE).prefetch(tf.data.AUTOTUNE)


# 
model = make_model()

# 
model.compile(loss=tf.keras.losses.MeanAbsoluteError(), optimizer=tf.keras.optimizers.Adam(learning_rate=0.0008))

# 
# ## Vídeo 3.3 - Criando funções de previsão

# 
def predict(model, timesteps=50, batch_size=BATCH_SIZE):
    # Inicialize com ruído gaussiano
    x = np.random.normal(size=(batch_size, 8, 8, 1))

    for i in trange(timesteps):
        t = np.full((batch_size, 1), i)  # Tempo como um vetor coluna
        x = model.predict([x, t], verbose=0)

    # Normalize as imagens para o intervalo [0, 1]
    x = (x - x.min()) / (x.max() - x.min())

    show_examples(x)

# 

test_input = [np.random.randn(BATCH_SIZE, 8, 8, 1), 
              np.random.randint(0, timesteps, (BATCH_SIZE, 1))]
model(test_input)
predict(model)

# 
def predict_step(model, timesteps=50, num_samples=BATCH_SIZE):
    xs = []
    x = np.random.normal(size=(num_samples, 8, 8, 1))  # Ajustado para Fashion MNIST

    for i in trange(timesteps):
        t = np.full((num_samples, 1), i)  # Tempo como vetor coluna
        x = model.predict([x, t], verbose=0)
        if i % 5 == 0:  # Salva a cada 5 passos para reduzir o número de imagens
            xs.append(x[0])

    # Normaliza as imagens para o intervalo [0, 1]
    xs = [(x - x.min()) / (x.max() - x.min()) for x in xs]

    plt.figure(figsize=(20, 3))
    for i, img in enumerate(xs):
        plt.subplot(1, len(xs), i+1)
        plt.imshow(cvtImg(img), cmap='gray')
        plt.title(f'Step {i*5}')
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# 
predict_step(model)

# ## Vídeo 3.4 - Treinando a U-Net

# 
def train_one(x_img):
    x_ts = generate_ts(len(x_img))
    x_a, x_b = forward_noise(x_img, x_ts)
    loss = model.train_on_batch([x_a, x_ts], x_b)
    return loss

# 
def train(R=50):
    total = 100
    for epoch in range(R):
        for batch in train_ds:
            x_img = batch
            x_ts = generate_ts(len(x_img))  # Gera timesteps aleatórios
            x_a, x_b = forward_noise(x_img, x_ts)
            loss = model.train_on_batch([x_a, x_ts], x_b)
        print(f"Epoch {epoch}, Loss: {loss:.4f}")


train()



predict(model)
predict_step(model)


