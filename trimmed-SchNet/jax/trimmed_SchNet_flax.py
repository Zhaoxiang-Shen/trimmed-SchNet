"""
A equivalent FLAX_NNX implementation of the trimmed SchNet as tensorflow/trimmed_SchNet_tf.py
Implemented for integration to JAX_MD.
"""

import jax.numpy as jnp
import jax_md.space as space
from flax import nnx
import numpy as np
import optax
import orbax.checkpoint as ocp
import os
import tensorflow as tf
from model_lib_flax import trimmed_SchNet

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

dataset = np.load(data_folder + 'dataset%s_Nat%d_Np%d.npz' % (mbd, Nat, Np))
coord_full = dataset['coord'][:Np_train*n_group]
type_full = dataset['type'][:Np_train*n_group]
F_full = dataset['F'][:Np_train*n_group] * 1e3
N_data = len(F_full)
print('P%s datasize =' % polymer, N_data)

# data division
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

# TF data management functions can be used here
def train_gen():
    i = 0
    permutation = np.random.permutation(int(N_train/3))
    while i < num_samples_train:
        if batching:
            id = 3 * permutation[int(i/3)] + i % 3
        else:
            id = i
        yield (coord_train[id], type_train[id]), F_train[id]
        i += 1

train_dataset = tf.data.Dataset.from_generator(train_gen,
                                               output_types=((tf.float32, tf.int32), tf.float32),
                                               output_shapes=(((Nat, 3), (Nat, )), (1, 3)))

# train_dataset = tf.data.Dataset.from_tensor_slices(((coord_train,type_train), F_train))
valid_dataset = tf.data.Dataset.from_tensor_slices(((coord_valid,type_valid), F_valid))


# Trimmed connections, with 2 near neighbours * 50 extra connections
N_extra = 50
idx_i = np.array([0,]*(Nat-1) + list(range(1,Nat)) + [1,]*N_extra + list(range(2,2+N_extra)) + [2,]*N_extra + list(range(3,3+N_extra)), dtype=np.int32)
idx_j = np.array(list(range(1,Nat)) + [0,]*(Nat-1) + list(range(2,2+N_extra)) + [1,]*N_extra + list(range(3,3+N_extra)) + [2,]*N_extra, dtype=np.int32)
seg_i = np.array([0,]*(Nat-1) + list(range(1,Nat)) + [1,]*N_extra + list(range(2,2+N_extra)) + [2,]*N_extra + list(range(3,3+N_extra)), dtype=np.int32)

sort_id = np.argsort(seg_i)
idx_i = np.array(idx_i[sort_id], dtype=np.int32)
idx_j = np.array(idx_j[sort_id], dtype=np.int32)
seg_i = np.array(seg_i[sort_id], dtype=np.int32)

# A displacement function is needed for handling periodicity, it is the same one used in JAX_MD
displacement_fn, _ = space.periodic_general(jnp.array([100, 100.0, 100.0]), wrapped=True, fractional_coordinates=False)

# parameters
num_interactions, embedding_dim, num_rbf, rbf_trainable = 1, 32, 100, True
ep0 = 0
num_epochs = 100
step_save = 50
batch_size = 32
model = trimmed_SchNet(Nat, num_interactions, embedding_dim, num_rbf, rbf_trainable, idx_i, idx_j, seg_i, displacement_fn)

# Set up the optimizer
optimizer = nnx.Optimizer(model, optax.adamw(1e-4))

# MAE loss function
@nnx.jit
def loss_fn(model, x, y):
    y_pred = model(x)
    return jnp.mean((y_pred - y) ** 2)

# Training step function
@nnx.jit
def train_step(model, optimizer, x, y):
  loss, grads = nnx.value_and_grad(loss_fn)(model, x, y)
  optimizer.update(grads)  # In place updates.
  return loss

#
# Prepossess the dataset
if not batching:
    train_dataset = train_dataset.shuffle(num_samples_train,reshuffle_each_iteration=True)
    valid_dataset = valid_dataset.shuffle(num_samples_valid,reshuffle_each_iteration=True)
train_dataset = train_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
valid_dataset = valid_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

steps_per_epoch_train = num_samples_train // batch_size
steps_per_epoch_valid = num_samples_valid // batch_size

# # Train the model
file_name_para = 'nnx_mse_gen_b%d_Nat%d_Np%d_BS%d_Ni%d_de%d_Nrbf%d_Trbf%d' % \
                 (batching, Nat, N_data, batch_size, num_interactions, embedding_dim, num_rbf, rbf_trainable)
curent_path = os.getcwd()
checkpointer = ocp.StandardCheckpointer()

if ep0 > 0:
    print('loading pretrained model...')
    graphdef, abstract_state = nnx.split(model)
    state_restored = checkpointer.restore(curent_path+'/_' + file_name_para+'_ep%d' % ep0, abstract_state)
    model = nnx.merge(graphdef, state_restored)
    loss_history_train = np.loadtxt('training_log/loss_history_'+file_name_para+'_ep%d.txt' % ep0).tolist()
    loss_history_valid = np.loadtxt('training_log/loss_history_valid_'+file_name_para+'_ep%d.txt' % ep0).tolist()

else:
    loss_history_train = []
    loss_history_valid = []

for epoch in range(ep0, num_epochs):
    epoch += 1
    print(f"Epoch {epoch}/{num_epochs}")
    progbar_train = tf.keras.utils.Progbar(steps_per_epoch_train, stateful_metrics=None, verbose=2)
    loss_epoch_train = 0
    for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
        x1, x2 = x_batch_train
        loss = train_step(model, optimizer, (x_batch_train[0].numpy(),x_batch_train[1].numpy()), y_batch_train.numpy())
        loss_epoch_train += loss
        progbar_train.update(step + 1, values=[('loss', loss)])

    # validation
    progbar_valid = tf.keras.utils.Progbar(steps_per_epoch_valid, stateful_metrics=None, verbose=2)
    loss_epoch_valid = 0
    for step, (x_batch_valid, y_batch_valid) in enumerate(valid_dataset):
        loss = loss_fn(model, (x_batch_valid[0].numpy(),x_batch_valid[1].numpy()), y_batch_valid.numpy())
        loss_epoch_valid += loss
        progbar_valid.update(step + 1, values=[('loss', loss)])

    # Save with manual check points

    loss_epoch_train = loss_epoch_train / steps_per_epoch_train
    loss_epoch_valid = loss_epoch_valid / steps_per_epoch_valid
    loss_history_train.append(loss_epoch_train)
    loss_history_valid.append(loss_epoch_valid)

    if epoch % step_save == 0 and epoch > ep0:
        np.savetxt('training_log/loss_history_' + file_name_para + '_ep%d.txt' % epoch, loss_history_train)
        np.savetxt('training_log/loss_history_valid_'+file_name_para+'_ep%d.txt' % epoch, loss_history_valid)

        _, state = nnx.split(model)
        checkpointer.save(curent_path+'training_log/check_points/' + file_name_para+'_ep%d' % epoch, state)

        try:
            os.remove('training_log/loss_history_'+file_name_para+'_ep%d.txt' % (epoch-step_save*2))
            os.remove('training_log/loss_history_valid_'+file_name_para+'_ep%d.txt' % (epoch-step_save*2))
        except:
            print('nothing to remove')
