"""
Key modules for the trimmed SchNet architecture
"""

import numpy as np
import tensorflow as tf

class ShiftedSoftplus(tf.keras.layers.Layer):
    """
    Softplus activation with shift log(2)
    """
    def __init__(self):
        super(ShiftedSoftplus, self).__init__()

    def call(self, inputs):
        return tf.math.softplus(inputs) - tf.math.log(2.0)

class AtomEmbedding(tf.keras.layers.Layer):
    """
    Atom-wise embedding layer that converts atomic numbers into feature vectors.
    """

    def __init__(self, embedding_dim, num_elements=10):
        super(AtomEmbedding, self).__init__()
        self.embedding = tf.keras.layers.Embedding(num_elements, embedding_dim)

    def call(self, atom_type_array):
        return self.embedding(atom_type_array)

class RBFConstraint(tf.keras.constraints.Constraint):
    """
    Custom constraint to ensure that RBF centers stay within a specific range.
    """
    def __init__(self, min_value=0.0, max_value=10.0):
        self.min_value = min_value
        self.max_value = max_value

    def __call__(self, w):
        return tf.clip_by_value(w, self.min_value, self.max_value)

class RBFExpansion(tf.keras.layers.Layer):
    """
    Radial Basis Function (RBF) expansion of distances.
    """
    def __init__(self, num_rbf, rbf_trainable, cutoff=15.0):
        super(RBFExpansion, self).__init__()
        self.num_rbf = num_rbf
        self.cutoff = cutoff
        self.centers = tf.Variable(np.linspace(0, cutoff, num_rbf), dtype=tf.float32,
                                   trainable=rbf_trainable,
                                   constraint=RBFConstraint(min_value=0.0, max_value=cutoff),
                                   name='rbf_center')

        self.gamma = tf.Variable(np.linspace(10, 10, num_rbf), dtype=tf.float32,
                                 trainable=rbf_trainable,
                                 constraint=RBFConstraint(min_value=1e-6, max_value=tf.float32.max),
                                 name='rbf_gamma')

    def call(self, distances):
        """
        Expand distances using RBF functions.
        """
        distances = tf.expand_dims(distances, -1)
        return tf.exp(-self.gamma * (distances - self.centers) ** 2)

def calculate_distances(positions, idx_i, idx_j):
    """
    Calculate the pairwise distances between atoms as a differentiable function of the positions.
    """
    Ri = tf.gather(positions, idx_i, axis=1)
    Rj = tf.gather(positions, idx_j, axis=1)
    distances = tf.sqrt(tf.reduce_sum(tf.square(Ri-Rj), axis=-1) + 1e-8)  # Pairwise distances
    return distances

class ContinuousFilterConvolution(tf.keras.layers.Layer):
    """
    Continuous filter convolution layer that models interactions between atoms
    based on their encoded connection
    """
    def __init__(self, embedding_dim, num_rbf, rbf_trainable):
        super(ContinuousFilterConvolution, self).__init__()
        self.embedding_dim = embedding_dim

        # RBF expansion layer
        self.rbf = RBFExpansion(num_rbf, rbf_trainable)
        self.dense1 = tf.keras.layers.Dense(embedding_dim)
        self.dense2 = tf.keras.layers.Dense(embedding_dim)
        self.activation = ShiftedSoftplus()

    def call(self, atom_features, distances, idx_j, seg_i):
        """
        Perform convolution on atom features
        """
        # RBF expansion of distances
        expanded_distances = self.rbf(distances)
        filters = self.dense1(expanded_distances)
        filters = self.activation(filters)
        filters = self.dense2(filters)
        filters = self.activation(filters)

        # Gather atom features and apply filters
        atom_features = tf.gather(atom_features, idx_j, axis=1)
        x = atom_features * filters
        x = tf.transpose(x, perm=[1, 2, 0])
        x = tf.math.segment_sum(x, seg_i)
        x = tf.transpose(x, perm=[2, 0, 1])
        return x

class InteractionBlock(tf.keras.layers.Layer):
    """
    SchNet interaction block that updates atom features with rbf encoding
    """
    def __init__(self, embedding_dim, dropout_rate, num_rbf, rbf_trainable):
        super(InteractionBlock, self).__init__()
        self.embedding_dim = embedding_dim

        # Dense layers for the interaction block
        self.dense1 = tf.keras.layers.Dense(embedding_dim)
        self.dense2 = tf.keras.layers.Dense(embedding_dim)
        self.dense3 = tf.keras.layers.Dense(embedding_dim)

        # activation function
        self.activation = ShiftedSoftplus()

        # dropout, not needed in general
        self.drop = tf.keras.layers.Dropout(dropout_rate)

        # Continuous convolutional layer
        self.continuous_conv = ContinuousFilterConvolution(embedding_dim, num_rbf, rbf_trainable)

    def call(self, atom_features, distances, idx_j, seg_i):
        """
        Update atom features based on interactions.
        """
        x = self.dense1(atom_features)
        x = self.continuous_conv(x, distances, idx_j, seg_i)
        x = self.dense2(x)
        x = self.activation(x)
        x = self.dense3(x)
        x = self.drop(x)
        x = atom_features + x
        return x
