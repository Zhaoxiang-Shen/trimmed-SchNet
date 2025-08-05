"""
A trimmed SchNet model for surrogate modeling of many-body dispersion in polymer melts.

The model can be trained for three types of polymer melts (PE, PP, PVC) and their mixed dataset.
An advanced MBD@rsSCS model is available for PE dataset.
Details of the architecture are discussed in our paper.
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
use_GPU = True
if not use_GPU:
    print('not using GPU')
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import numpy as np
import tensorflow as tf
from model_lib_tf import calculate_distances,  ShiftedSoftplus, AtomEmbedding,  InteractionBlock

polymer = 'E'
mbd = ''
mixed = False
Nat = 1000  # number of atom in the cutoff

# Unit-specified batching
batching = True
if batching:
    print('batching')

# Pre load data, in single polymer set, Np counts the smallest unit (Np_train = Np*n_group)
if polymer == 'E':
    n_group = 3
    Np = 23998
    Np_train = 20000
    data_folder = 'dataset/polyethylene/'
elif polymer == 'P':
    n_group = 9
    Np = 11892
    Np_train = 8000
    data_folder = 'dataset/polypropylene/'

elif polymer == 'VC':
    n_group = 6
    Np = 11887
    Np_train = 10000
    data_folder = 'dataset/polyvinyl_chloride/'

elif polymer == 'EP':
    mixed = True
    n_group = 3
    Np1 = 12000
    Np2 = 4000
    Np_train = 72000
    data_folder = 'dataset/p_mixed/'

elif polymer == 'EVC':
    mixed = True
    n_group = 3
    Np1 = 12000
    Np2 = 6000
    Np_train = 72000
    data_folder = 'dataset/p_mixed/'

elif polymer == 'EPVC':
    mixed = True
    n_group = 3
    Np1 = 8000
    Np2 = 2700
    Np3 = 4000
    Np_train = 72300
    data_folder = 'dataset/p_mixed/'

elif polymer == 'E_rsscs':
    n_group = 3
    Np = 23639
    Np_train = 20000
    data_folder = 'dataset/mbd_mixed/'
    mbd = '_rsscs'

elif polymer == 'E_t':
    n_group = 3
    Np = 24000
    Np_train = 20000
    data_folder = 'dataset/t_mixed/'

if not mixed:
    dataset = np.load(data_folder + 'dataset%s_Nat%d_Np%d.npz' % (mbd, Nat, Np))
    coord_full = dataset['coord'][:Np_train*n_group]
    type_full = dataset['type'][:Np_train*n_group]
    F_full = dataset['F'][:Np_train*n_group] * 1e3
    if polymer == 'E_rsscs':  # additional scaling
        F_full = F_full * 1e1

else:
    try:
        dataset = np.load(data_folder + 'dataset_P%s_Nat%d_Np%d+%d.npz' % (polymer,Nat, Np1, Np2))
    except:
        dataset = np.load(data_folder + 'dataset_P%s_Nat%d_Np%d+%d+%d.npz' % (polymer,Nat, Np1, Np2, Np3))

    coord_full = dataset['coord'][:Np_train]
    type_full = dataset['type'][:Np_train]
    F_full = dataset['F'][:Np_train] * 1e3

N_data = len(F_full)
print('P%s datasize =' % polymer, N_data)

# data division, 10% used for validation
N_valid = int(N_data * 0.1)
N_train = N_data - N_valid

F_train = F_full[N_valid:]
coord_train = coord_full[N_valid:]
type_train = type_full[N_valid:]

F_valid = F_full[:N_valid]
coord_valid = coord_full[:N_valid]
type_valid = type_full[:N_valid]

num_samples_train = len(F_train)
num_samples_valid = len(F_valid)

# data generation function support the batching strategy (shuffling included)
def train_gen():
    i = 0
    permutation = np.random.permutation(int(N_train/n_group))
    while i < num_samples_train:
        if batching:
            id = n_group * permutation[int(i/n_group)] + i % n_group
        else:
            id = i
        yield (coord_train[id], type_train[id]), F_train[id]
        i += 1

train_dataset = tf.data.Dataset.from_generator(train_gen,
                                               output_types=((tf.float32, tf.int32), tf.float32),
                                               output_shapes=(((Nat, 3), (Nat, )), (1, 3)))

# train_dataset = tf.data.Dataset.from_tensor_slices(((coord_train,type_train), F_train))
valid_dataset = tf.data.Dataset.from_tensor_slices(((coord_valid,type_valid), F_valid))

# setting
batch_size = 36
num_epochs = 100
step_save = 50
ep0 = 0
loss_func = 'mse'
lr0 = 1e-3
w_decay = 0.004
opt_reset = False  # for reset the weights of optimizer

rbf_trainable = True
num_interactions = 1
embedding_dim = 32
num_rbf = 100
drop = 0.0

# Trimmed connections, with 2 near neighbours * 50 extra connections
N_extra = 50
idx_i = np.array([0,]*(Nat-1) + list(range(1,Nat)) + [1,]*N_extra + list(range(2,2+N_extra)) + [2,]*N_extra + list(range(3,3+N_extra)), dtype=np.int32)
idx_j = np.array(list(range(1,Nat)) + [0,]*(Nat-1) + list(range(2,2+N_extra)) + [1,]*N_extra + list(range(3,3+N_extra)) + [2,]*N_extra, dtype=np.int32)
seg_i = np.array([0,]*(Nat-1) + list(range(1,Nat)) + [1,]*N_extra + list(range(2,2+N_extra)) + [2,]*N_extra + list(range(3,3+N_extra)), dtype=np.int32)

sort_id = np.argsort(seg_i)
idx_i = np.array(idx_i[sort_id], dtype=np.int32)
idx_j = np.array(idx_j[sort_id], dtype=np.int32)
seg_i = np.array(seg_i[sort_id], dtype=np.int32)

# Network
@tf.keras.saving.register_keras_serializable(package="trimmed_SchNet")
class trimmed_SchNet(tf.keras.Model):
    def __init__(self, num_interactions, embedding_dim, num_rbf, rbf_trainable, dropout_rate,
                       idx_i, idx_j, seg_i):
        super(trimmed_SchNet, self).__init__()
        self.num_interactions = num_interactions
        self.embedding_dim = embedding_dim
        self.num_rbf = num_rbf
        self.dropout_rate = dropout_rate
        self.idx_i = idx_i
        self.idx_j = idx_j
        self.seg_i = seg_i
        self.rbf_trainable = rbf_trainable
        # Atom embedding layer
        self.atom_embedding = AtomEmbedding(embedding_dim)

        # Interaction blocks
        self.interaction_blocks = [
            InteractionBlock(embedding_dim, self.dropout_rate, self.num_rbf, rbf_trainable) for _ in range(num_interactions)
        ]

        # activation function
        self.activation = ShiftedSoftplus()

        # Output layer to predict energy (scalar output)
        self.dense = tf.keras.layers.Dense(int((embedding_dim)/2))
        self.energy_output = tf.keras.layers.Dense(1, use_bias=False)

    def get_config(self):
        base_config = super().get_config()
        config = {
                "num_interactions": self.num_interactions,
                "embedding_dim": self.embedding_dim,
                "num_rbf": self.num_rbf,
                "rbf_trainable": self.rbf_trainable,
                "dropout_rate": self.dropout_rate,
                "idx_i": self.idx_i,
                "idx_j": self.idx_j,
                "seg_i": self.seg_i,
            }
        return {**base_config, **config}

    def call(self, inputs):
        positions, atom_type_array = inputs
        with tf.GradientTape() as tape:
            # Enable gradient tracking for positions (for force calculation)
            tape.watch(positions)

            # Embed atomic numbers
            atom_features = self.atom_embedding(atom_type_array)

            # Calculate pairwise distances as a function of positions
            distances = calculate_distances(positions, self.idx_i, self.idx_j)

            # Pass atom features through interaction blocks
            for interaction in self.interaction_blocks:
                atom_features = interaction(atom_features, distances, self.idx_j, self.seg_i)

            atom_features = self.dense(atom_features)
            atom_features = self.activation(atom_features)

            # Aggregate atom-wise features and predict total energy (scalar)
            energy = self.energy_output(atom_features)
            energy = tf.reduce_sum(energy, axis=1)

        # Compute forces as the negative gradient of energy with respect to positions
        forces = -tape.gradient(energy, positions)
        output = forces[:, :1, :]
        return output

# Compile the model
# Prepossess the dataset
if not batching:
    train_dataset = train_dataset.shuffle(num_samples_train,reshuffle_each_iteration=True)
    valid_dataset = valid_dataset.shuffle(num_samples_valid,reshuffle_each_iteration=True)
train_dataset = train_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
valid_dataset = valid_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

# loss function
if loss_func == 'mse':
    loss_func_formal = 'mean_squared_error'
elif loss_func == 'mae':
    loss_func_formal = 'mean_absolute_error'

# Create a MirroredStrategy.
strategy = tf.distribute.MirroredStrategy()
print("Number of devices: {}".format(strategy.num_replicas_in_sync))

# Open a strategy scope.
with strategy.scope():
    # Build the model with dynamic input and output shapes
    model = trimmed_SchNet(num_interactions, embedding_dim, num_rbf, rbf_trainable, drop, idx_i, idx_j, seg_i)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr0,weight_decay=w_decay)
    model.compile(optimizer=optimizer, loss=loss_func_formal, run_eagerly=False)

if opt_reset:
    checkpoint = tf.train.Checkpoint(model=model)
else:
    checkpoint = tf.train.Checkpoint(model=model,optimizer=optimizer)

# Train the model
file_name_para = loss_func+'_P%s_b%d_Nat%d_Np%d_BS%d_Ne%d_Ni%d_de%d_Nrbf%d_Trbf%d_d%.2f_w%.3f' % \
                 (polymer, batching, Nat, N_data, batch_size, N_extra, num_interactions, embedding_dim, num_rbf, rbf_trainable, drop, w_decay)

if ep0 > 0:
    print('loading pretrained model...')
    checkpoint.restore('training_log/check_points/'+file_name_para+'_ep%d_cp-%d' % (ep0,ep0/step_save))
    loss_history_train = np.loadtxt('training_log/loss_history_'+file_name_para+'_ep%d.txt' % ep0).tolist()
    loss_history_valid = np.loadtxt('training_log/loss_history_valid_'+file_name_para+'_ep%d.txt' % ep0).tolist()
else:
    loss_history_train = []
    loss_history_valid = []

# train by loop over model.fit()
for fit_step in range(int((num_epochs-ep0)/step_save)):
    epoch_start = fit_step*step_save + ep0
    epoch_end = epoch_start + step_save

    # Learning rate update
    if epoch_start >= 50:
        tf.keras.backend.set_value(model.optimizer.lr, 1e-4)

    history = model.fit(train_dataset, validation_data=valid_dataset,
                        initial_epoch=epoch_start, epochs=epoch_end, verbose=2)

    total_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
    print(f"Total number of parameters: {total_params}")

    # Save with manual check points
    loss_history_train = loss_history_train + history.history['loss']
    loss_history_valid = loss_history_valid + history.history['val_loss']

    np.savetxt('training_log/loss_history_'+file_name_para+'_ep%d.txt' % epoch_end, loss_history_train)
    np.savetxt('training_log/loss_history_valid_'+file_name_para+'_ep%d.txt' % epoch_end, loss_history_valid)
    checkpoint.save('training_log/check_points/'+file_name_para+'_ep%d_cp' % epoch_end)
    try:
        os.remove('training_log/loss_history_'+file_name_para+'_ep%d.txt' % (epoch_end-step_save*2))
        os.remove('training_log/loss_history_valid_'+file_name_para+'_ep%d.txt' % (epoch_end-step_save*2))
        os.remove('training_log/check_points/'+file_name_para+'_ep%d_cp-%d.index' % (epoch_end-step_save*2,(epoch_end-step_save*2)/step_save))
        os.remove('training_log/check_points/'+file_name_para+'_ep%d_cp-%d.data-00000-of-00001' %
                  (epoch_end - step_save * 2, (epoch_end - step_save * 2) /step_save))
    except:
        print('nothing to remove')
    #

