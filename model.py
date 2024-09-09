import h5py
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Input, Activation, Dense, LSTM, Subtract, Dropout, Lambda, GRU, RepeatVector, TimeDistributed, Concatenate,PReLU, LSTM

from tensorflow.keras import backend as K

def load_preprocessed_data(save_path):
    with h5py.File(save_path, 'r') as hf:
        x1 = hf['x1'][:]
        x2 = hf['x2'][:]
        x3 = hf['x3'][:]
        y = hf['y'][:]
    return [x1,x2,x3], y

class VAEEncoder:
    def __init__(self, timesteps, data_dim, latent_dim):
        self.timesteps = timesteps
        self.data_dim = data_dim
        self.latent_dim = latent_dim

    def sampling(self, args):
        z_mean, z_log_variance = args
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = K.random_normal(shape=(batch, dim),mean=0.,stddev=1.)
        return z_mean + K.exp(0.5 * z_log_variance) * epsilon

    def build(self):
        encoder_input = Input(shape=(self.timesteps, self.data_dim))
        
        x = GRU(200, return_sequences=True)(encoder_input)
        x = GRU(100, return_sequences=False)(x)
   
        z_mean = Dense(self.latent_dim)(x)
        z_mean = PReLU()(z_mean)
        
        z_log_variance = Dense(self.latent_dim)(x)
        z_log_variance = PReLU()(z_log_variance)
        
        encoder_output = Lambda(self.sampling)([z_mean, z_log_variance])
        
        return Model(encoder_input, encoder_output)


class AuralNet:
    def __init__(self):
        self.timesteps = 39
        self.data_dim = 64
        self.latent_dim = 100
        self.gccphat_dim = 33
        self.dropRate = 0.2
        self.d_model = 64

    def build_model(self):
        encoder1_input = Input(shape=(self.timesteps, self.data_dim)) # (batch_size, timesteps, data_dim)
        encoder2_input = Input(shape=(self.timesteps, self.data_dim)) # (batch_size, timesteps, data_dim)
        cc_input = Input(shape=(self.gccphat_dim,))

        encoder1 = VAEEncoder(self.timesteps, self.data_dim, self.latent_dim).build()
        encoder2 = VAEEncoder(self.timesteps, self.data_dim, self.latent_dim).build()
        
        encoder1_output = encoder1(encoder1_input) # (batch_size, latent_dim)
        encoder2_output = encoder2(encoder2_input) # (batch_size, latent_dim)

        # Subtract
        Subed = Subtract()([encoder1_output, encoder2_output])

        # Attention Module
        h1_in = Concatenate(axis=1)([cc_input,encoder1_output])
        h1 = Dense(self.data_dim)(h1_in)
        h1 = PReLU()(h1)
        h1_norm = tf.keras.layers.LayerNormalization()(h1)
        q_1 = Dense(self.d_model * self.timesteps, name='dense_q1')(h1_norm)
        k_1 = Dense(self.d_model * self.timesteps, name='dense_k1')(h1_norm)
        v_1 = Dense(self.d_model * self.timesteps, name='dense_v1')(h1_norm)
        atten1_out = self.atten([q_1, k_1, v_1])
        atten1 = Dense(self.data_dim)(atten1_out)
        h1_atten = Dropout(self.dropRate)(atten1)
        h1_out = h1 + h1_atten

        # Attention Module
        h2_in = Concatenate(axis=1)([cc_input,encoder2_output])
        h2 = Dense(self.data_dim)(h2_in)
        h2 = PReLU()(h2)
        h2_norm = tf.keras.layers.LayerNormalization()(h2)
        q_2 = Dense(self.d_model * self.timesteps, name='dense_q2')(h2_norm)
        k_2 = Dense(self.d_model * self.timesteps, name='dense_k2')(h2_norm)
        v_2 = Dense(self.d_model * self.timesteps, name='dense_v2')(h2_norm)
        atten2_out = self.atten([q_2, k_2, v_2])
        atten2 = Dense(self.data_dim)(atten2_out)
        h2_atten = Dropout(self.dropRate)(atten2)
        h2_out = h2 + h2_atten 

        features = Concatenate(axis=1)([h1_out, h2_out, Subed]) # (batch_size, data_dim*2+latent_dim)

        # MLP
        body = Dense(200)(features)
        body = PReLU()(body)
        body = Dropout(0.2)(body)

        intentionNets = []
        aziNets = []
        eleNets = []
        for i in range(8):
            # intension nets
            intention_hidden = Dense(100)(body)
            intention_hidden = PReLU()(intention_hidden)
            
            intention = Dense(50)(intention_hidden)
            intention = PReLU()(intention)

            intentionNet = Dense(1, activation='sigmoid')(intention)

            features = Concatenate(axis=1)([body, intention])
            features = PReLU()(features)

            subNet = Dense(100, activation='relu')(features)
            subNet = Dropout(0.2)(subNet)

            aziNet = Dense(50, activation='relu')(subNet)
            aziNet = Dense(1, activation='sigmoid')(aziNet)

            eleNet = Dense(50, activation='relu')(subNet)
            eleNet = Dense(1, activation='tanh')(eleNet)

            intentionNets.append(intentionNet)
            aziNets.append(aziNet)
            eleNets.append(eleNet)

        intentionNet_out = tf.stack(intentionNets)
        aziNet_out = tf.stack(aziNets)
        eleNet_out = tf.stack(eleNets)
        output = Concatenate(axis=-1)([intentionNet_out, aziNet_out, eleNet_out]) # (8, batch_size, 3)
        output = tf.transpose(output, perm=[1, 0, 2]) # (batch_size, 8, 3)

        model = Model(inputs=[encoder1_input, encoder2_input,cc_input], outputs= output)
        return model
        
    def atten(self,featureIn):
        query, key, value = featureIn

        query = tf.reshape(query, shape=(-1, self.timesteps, self.d_model))
        key = tf.reshape(key, shape=(-1, self.timesteps, self.d_model))
        value = tf.reshape(value, shape=(-1,self.timesteps, self.d_model))

        d_k = tf.cast(query.shape[-1], tf.float32)
        key = tf.transpose(key, perm=[0, 2, 1])

        scores = tf.matmul(query, key )/ tf.math.sqrt(d_k)
        p_atten = tf.nn.softmax(scores, axis=-1)
        out = (tf.matmul(p_atten, value))

        return tf.reshape(out, shape=(-1, self.timesteps*self.d_model))
    
