import DGM
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %% Parameters

# neural network parameters
num_layers = 3
nodes_per_layer = 75
learning_rate = 0.001

# Training parameters
sampling_stages = 100  # number of times to resample new time-space domain points
steps_per_sample = 10  # number of SGD steps to take before re-sampling

# Sampling parameters
nSim_interior = 1000
nSim_terminal = 25

# Plot options
n_plot = 41  # Points on plot grid for each dimension

# Save options
saveOutput = False
saveName = 'diffusion_data'
saveFigure = False
figureName = 'diffusion_data.png'


def data_loader():
    data = pd.read_csv('Pressure_WaterSatv.txt', sep='\t', header=1)
    column_names = list(data.columns)

    time = data[['Time (day)']]

    oil = data[[column for column in data.columns if 'Oil' in column]]
    water = data[[column for column in data.columns if 'Water' in column]]
    pressure = data[[column for column in data.columns if 'Pressure' in column]]

    x_column_names = list(oil.columns)
    for column_index in range(len(x_column_names)):
        column_name = x_column_names[column_index]
        if '}-' in column_name:
            column_name = column_name.split('-')[0].replace('{', '').replace('}', '').replace('/', ',').split(',')
            major_index = column_name[0].replace(' ', '')
            minor_index = column_name[3].replace(' ', '')

            position = float(str(major_index) + '.' + str(minor_index))

            x_column_names[column_index] = position - 1

    oil.columns = x_column_names
    water.columns = x_column_names
    pressure.columns = x_column_names

    return time, oil, water, pressure


# %% Sampling function - randomly sample time-space pairs
def sampler(nSim_interior, nSim_terminal, time, quantity):
    ''' Sample time-space points from the function's domain; points are sampled
        uniformly on the interior of the domain, at the initial/terminal time points
        and along the spatial boundary at different time points.

    Args:
        nSim_interior: number of space points in the interior of the function's domain to sample
        nSim_terminal: number of space points at terminal time to sample (terminal condition)
    '''

    # Sampler #1: domain interior

    t_int_samples_indices = np.random.randint(1, quantity.shape[0] - 1, nSim_interior)
    x_int_samples_indices = np.random.randint(0, quantity.shape[1] - 1, nSim_interior)

    quantityInt = np.array(quantity)[t_int_samples_indices, x_int_samples_indices].reshape((nSim_interior, 1))
    xInt = np.array(quantity.columns)[x_int_samples_indices].reshape((nSim_interior, 1))
    tInt = np.array(time[1:])[t_int_samples_indices].reshape((nSim_interior, 1))

    # Sampler #2: spatial boundary
    # no spatial boundary condition for this problem

    # Sampler #3: initial/terminal condition
    x_ter_samples_indices = np.random.randint(0, quantity.shape[1] - 1, nSim_terminal)
    t_ter_samples_indices = np.zeros(nSim_terminal).astype('int')

    quantityTer =  np.array(quantity)[t_ter_samples_indices, x_ter_samples_indices].reshape((nSim_terminal, 1))
    tTer = np.array(time)[0][0] * np.ones((nSim_terminal, 1)).reshape((nSim_terminal, 1))
    xTer = np.array(quantity.columns)[x_ter_samples_indices].reshape((nSim_terminal, 1))

    return tInt, xInt, quantityInt, tTer, xTer, quantityTer


# %% Loss function
def loss(model, tInt, xInt, quantityInt, tTer, xTer, quantityTer):
    ''' Compute total loss for training.

    Args:
        model:      DGM model object
        tInt: sampled time points in the interior of the function's domain
        xInt: sampled space points in the interior of the function's domain
        tTer: sampled time points at terminal point (vector of terminal times)
        xTer: sampled space points at terminal time
    '''

    # Loss term #1: PDE
    qPred = model(tInt, xInt)
    qDiff = qPred - quantityInt

    # compute average L2-norm of differential operator
    L1 = tf.reduce_mean(tf.square(qDiff))

    # Loss term #2: boundary condition
    # no boundary condition for this problem

    # Loss term #3: initial/terminal condition
    qPredTer = model(tTer, xTer)

    L3 = tf.reduce_mean(tf.square(qPredTer - quantityTer))

    return L1, L3


# %% Set up network

# initialize DGM model (last input: space dimension = 1)
model = DGM.DGMNet(nodes_per_layer, num_layers, 1)

# tensor placeholders (_tnsr suffix indicates tensors)
# inputs (time, space domain interior, space domain at initial time)
tInt_tnsr = tf.placeholder(tf.float32, [None, 1])
xInt_tnsr = tf.placeholder(tf.float32, [None, 1])
qInt_tnsr = tf.placeholder(tf.float32, [None, 1])
tTer_tnsr = tf.placeholder(tf.float32, [None, 1])
xTer_tnsr = tf.placeholder(tf.float32, [None, 1])
qTer_tnsr = tf.placeholder(tf.float32, [None, 1])

# loss
L1_tnsr, L3_tnsr = loss(model, tInt_tnsr, xInt_tnsr, qInt_tnsr, tTer_tnsr, xTer_tnsr, qTer_tnsr)
loss_tnsr = L1_tnsr + L3_tnsr

# option value function
V = model(tInt_tnsr, xInt_tnsr)

# set optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_tnsr)

# initialize variables
init_op = tf.global_variables_initializer()

# open session
sess = tf.Session()
sess.run(init_op)

# load Data
time, oil, water, pressure = data_loader()

# %% Train network
# for each sampling stage
for i in range(sampling_stages):

    # sample uniformly from the required regions
    tInt, xInt, qInt, tTer, xTer, qTer = sampler(nSim_interior, nSim_terminal, time, oil)

    # for a given sample, take the required number of SGD steps
    for _ in range(steps_per_sample):
        loss, L1, L3, _ = sess.run([loss_tnsr, L1_tnsr, L3_tnsr, optimizer],
                                   feed_dict={tInt_tnsr: tInt, xInt_tnsr: xInt, qInt_tnsr: qInt,
                                              tTer_tnsr: tTer, xTer_tnsr: xTer, qTer_tnsr: qTer})

    print(loss, L1, L3, i)

# save outout
if saveOutput:
    saver = tf.train.Saver()
    saver.save(sess, './SavedNets/' + saveName)

# %% Plot results
# figure options
plt.figure()
plt.figure(figsize=(12, 10))

# time values at which to examine density
blockPositions = [0.5, 2.5, 4.5, 6.5, 8.5, 9.5]

for i, blockPosition in enumerate(blockPositions):

    # specify subplot
    plt.subplot(3, 2, i + 1)

    t_plot = np.array(time[1:2001])
    x_plot = blockPosition * np.ones(t_plot.shape[0])

    # simulate process at current t
    functionValue = oil[blockPosition][1:2001]

    # prediction
    fittedValue = sess.run([V], feed_dict={tInt_tnsr: t_plot.reshape(-1, 1), xInt_tnsr: x_plot.reshape(-1, 1)})

    plt.plot(t_plot, functionValue, color='b', label='Analytical Solution', linewidth=3, linestyle=':')
    plt.plot(t_plot, fittedValue[0], color='r', label='DGM estimate')

    # subplot options
    if i == 0:
        plt.legend(loc='upper left', prop={'size': 16})

# adjust space between subplots
plt.subplots_adjust(wspace=0.3, hspace=0.4)

if saveFigure:
    plt.savefig(figureName)
