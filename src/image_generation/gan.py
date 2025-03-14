import tensorflow as tf
import keras.api as keras
from keras.api import layers
from keras.api.layers import Conv2D, LeakyReLU, Flatten, Dropout, Dense
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

# Parâmetros
tamanho_imagem = (128, 128)
canal = 1  # Imagem em escala de cinza
BUFFER_SIZE = 60000
BATCH_SIZE = 64

# Função para normalizar as imagens
def normalize(image):
    image = tf.cast(image, tf.float32) / 255.0  # Converte para float e normaliza
    image = (image - 0.5) * 2  # Ajusta para o intervalo [-1, 1]
    return image

# Carregando dataset
dataset_path = "data/train"
train_ds = keras.preprocessing.image_dataset_from_directory(
    dataset_path,
    label_mode=None,
    color_mode='grayscale',
    image_size=tamanho_imagem,
    batch_size=BATCH_SIZE
)

train_ds = train_ds.map(lambda x: (normalize(x),))

def plot_a_few_images(dataset):
    for batch in dataset:
        for image in batch:
            plt.imshow(image[0, :, :, 0] * 127.5 + 127.5, cmap='gray')
            plt.show()
        
plot_a_few_images(train_ds)

# Criando o gerador
def constroi_gerador():
    input = layers.Input((100,))
    x = layers.Dense(4*4*256, use_bias=False)(input)
    x = layers.LeakyReLU()(x)
    x = layers.Reshape((4, 4, 256))(x)

    x= layers.Conv2DTranspose(256, (5, 5), strides=(2, 2), padding='same', use_bias=False)(x)
    x= layers.BatchNormalization()(x)
    x= layers.LeakyReLU()(x)
    

    x= layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False)(x)
    x= layers.BatchNormalization()(x)
    x= layers.LeakyReLU()(x)
    
    x= layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False)(x)
    x= layers.LeakyReLU()(x)

    x= layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False)(x)
    x= layers.LeakyReLU()(x)
    
    output = layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh')(x)

    model = keras.Model(input,output)
    
    return model



# Criando o discriminador
def constroi_discriminador():
    input = layers.Input((128,128,1))
    x = layers.Conv2D(64, (3, 3), strides=(2, 2), padding='same')(input)
    x = layers.LeakyReLU()(x)
    
    x = layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same')(x)
    x = layers.LeakyReLU()(x)
    
    x = layers.Conv2D(128, (2, 2), strides=(2, 2), padding='same')(x)
    x = layers.LeakyReLU()(x)
    
    x = layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same')(x)
    x = layers.LeakyReLU()(x)

    x = layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same')(x)
    x = layers.LeakyReLU()(x)
    
    x = layers.Flatten()(x)
    x = layers.Dropout(0.5)(x)
    output = layers.Dense(1)(x)

    modelo = keras.Model(input,output)
    
    return modelo

def constroi_discriminador_2():
    model = Sequential()
	# normal
    model.add(Conv2D(64, (3,3), padding='same', input_shape=(128,128,1)))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.5))
	# downsample
    model.add(Conv2D(128, (3,3), strides=(2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.5))
	# downsample
    model.add(Conv2D(128, (3,3), strides=(2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    # downsample
    model.add(Conv2D(128, (3,3), strides=(2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.4))
	# downsample
    model.add(Conv2D(256, (3,3), strides=(2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2D(128, (3,3), strides=(2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.3))
	# classifier
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(1))
    return model

# Definição das funções de perda
def loss_discriminador(real_output, fake_output):
    real_loss = keras.losses.BinaryCrossentropy(from_logits=True)(tf.ones_like(real_output), real_output)
    fake_loss = keras.losses.BinaryCrossentropy(from_logits=True)(tf.zeros_like(fake_output), fake_output)
    return real_loss, fake_loss

def loss_gerador(fake_output):
    return keras.losses.BinaryCrossentropy(from_logits=True)(tf.ones_like(fake_output), fake_output)

# Criando os modelos
gerador = constroi_gerador()
discriminador = constroi_discriminador_2()

# Otimizadores
optimizer_gerador = keras.optimizers.Adam(2.5e-4, beta_1 = 0.5)
optimizer_discriminador = keras.optimizers.Adam(.5e-4, beta_1 = 0.5)

# Função de treinamento@tf.function
def passo_treino_2(imagens):
    batch_size = tf.shape(imagens)[0]
    ruido = tf.random.normal([batch_size, 100])
    
    with tf.GradientTape() as tape_g, tf.GradientTape() as tape_d:
        imagens_falsas = gerador(ruido, training=True)
        output_real = discriminador(imagens, training=True)
        output_falso = discriminador(imagens_falsas, training=True)
        
        loss_g = loss_gerador(output_falso)
        loss_d_r, loss_d_f = loss_discriminador(output_real, output_falso)
        loss_d = loss_d_f+loss_d_r
        
    gradientes_g = tape_g.gradient(loss_g, gerador.trainable_variables)
    gradientes_d = tape_d.gradient(loss_d, discriminador.trainable_variables)
    
    optimizer_gerador.apply_gradients(zip(gradientes_g, gerador.trainable_variables))
    optimizer_discriminador.apply_gradients(zip(gradientes_d, discriminador.trainable_variables))
    
    return loss_g, loss_d, loss_d_r, loss_d_f

# Função para gerar e salvar imagens
def gera_e_salva_imagens(modelo, epoca, seed):
    imagens_geradas = modelo(seed, training=False)
    plt.imshow(imagens_geradas[0, :, :, 0] * 127.5 + 127.5, cmap='gray')
    
    # for i in range(imagens_geradas.shape[0]):
    #     plt.subplot(4, 4, i + 1)
    #     plt.imshow(imagens_geradas[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
    #     plt.axis('off')
    
    plt.savefig(f'imagem_128_epoca_{epoca}.png')
    #plt.show()

# Loop de treinamento
def treino(dataset, epocas):
    seed = tf.random.normal([1, 100])
    count = 0
    
    for epoca in range(epocas):
        for imagem_batch in dataset:
            loss_g, loss_d, loss_d_r, loss_d_f = passo_treino_2(imagem_batch[0])
        epoca+=1
        print(f"Epoca {epoca}:\nLoss do gerador = {loss_g}\nLoss do discriminador = {loss_d}\nLoss do discriminador fake: {loss_d_f}\nLoss do discriminador real: {loss_d_r}\n\n")
        count+=1
        if count % 100 == 0:
            gera_e_salva_imagens(gerador, epoca, seed)
            print(f'Época {epoca} concluída')

# Executando o treinamento
treino(train_ds, epocas=2000)

gerador.save("gerador_modelo_3.keras")
discriminador.save("discriminador_modelo_3.keras")