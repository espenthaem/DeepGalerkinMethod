import DGM
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
            major_index = int(column_name[0].replace(' ', ''))
            minor_index = int(column_name[3].replace(' ', '')) - 1

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
        time: the time axis
        quantity: pressure, water saturation or oil saturation array
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


def slice_plots(time, quantity, sess, split=False):
    # figure options
    plt.figure(figsize=(12, 10))

    # time values at which to examine density
    blockPositions = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5]

    for i, blockPosition in enumerate(blockPositions):
        # specify subplot
        plt.subplot(5, 2, i + 1)

        t_plot = np.array(time[1:2601])
        x_plot = blockPosition * np.ones(t_plot.shape[0])

        # simulate process at current t
        functionValue = quantity[blockPosition][1:2601]

        # prediction
        fittedValue = sess.run([V], feed_dict={tInt_tnsr: t_plot.reshape(-1, 1), xInt_tnsr: x_plot.reshape(-1, 1)})

        plt.plot(t_plot, functionValue, color='b', label='Analytical Solution', linewidth=3, linestyle=':')
        plt.plot(t_plot, fittedValue[0], color='r', label='DGM estimate')
        plt.xlabel('time (days)')
        plt.ylabel('%s' % quantity_string)
        plt.title('x = %0.2f' % blockPosition)

        # subplot options
        # if i == 0:
        #    plt.legend(loc='upper left', prop={'size': 16})

    # adjust space between subplots
    plt.subplots_adjust(wspace=0.3, hspace=1.0)
    if split:
        plt.savefig('%s_split.png' % quantity_string)
    else:
        plt.savefig('%s.png' % quantity_string)


# Generate 2D Error plots
def plots_2d(time, quantity, out_df, split=False):
    plt.figure(figsize=(12, 10))

    yticklabels = [time.values[0][0], time.values[100][0], time.values[200][0],
                   time.values[300][0], time.values[400][0], time.values[500][0]]
    yticklabels = ['%0.1f' % ytickvalue for ytickvalue in yticklabels]

    x_columns = sorted(pressure.columns)
    ax = sns.heatmap(abs(out_df.values[:500] - quantity[x_columns].values[:500]), cmap="Spectral_r")
    adjust = ax.set(xlabel='Distance (m)', ylabel='Time',
                    xticks=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                    xticklabels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                    yticks=[0, 100, 200, 300, 400, 500], yticklabels=yticklabels,
                    title='Absolute Error %s saturation' % quantity_string
                    )
    ax.invert_yaxis()

    if split:
        ax.figure.savefig('%s_2D_error_small_split.png' % quantity_string)
    else:
        ax.figure.savefig('%s_2D_error_small.png' % quantity_string)

    plt.figure(figsize=(12, 10))
    yticklabels = [time.values[0][0], time.values[500][0], time.values[1000][0], time.values[1500][0],
                   time.values[2000][0], time.values[2500][0]]
    yticklabels = ['%0.1f' % ytickvalue for ytickvalue in yticklabels]

    x_columns = sorted(pressure.columns)
    ax = sns.heatmap(abs(out_df.values[:2500] - quantity[x_columns].values[:2500]), cmap="Spectral_r")
    adjust = ax.set(xlabel='Distance (m)', ylabel='Time',
                    xticks=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                    xticklabels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                    yticks=[0, 400, 800, 1200, 1600, 2000], yticklabels=yticklabels,
                    title='Absolute Error %s saturation' % quantity_string
                    )
    ax.invert_yaxis()

    if split:
        ax.figure.savefig('%s_2D_error_split.png' % quantity_string)
    else:
        ax.figure.savefig('%s_2D_error.png' % quantity_string)

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
pressure = pressure/10000
quantity = oil
quantity_string = "oil"

# %% Train network
# for each sampling stage
for i in range(sampling_stages):

    # sample uniformly from the required regions
    tInt, xInt, qInt, tTer, xTer, qTer = sampler(nSim_interior, nSim_terminal, time, quantity)

    # for a given sample, take the required number of SGD steps
    for _ in range(steps_per_sample):
        loss, L1, L3, _ = sess.run([loss_tnsr, L1_tnsr, L3_tnsr, optimizer],
                                   feed_dict={tInt_tnsr: tInt, xInt_tnsr: xInt, qInt_tnsr: qInt,
                                              tTer_tnsr: tTer, xTer_tnsr: xTer, qTer_tnsr: qTer})

    print(loss, L1, L3, i)

# Generate plots
slice_plots(time, quantity, sess)

# Predict function value at every grid point
x_columns = sorted(pressure.columns)

out_df = pd.DataFrame(columns=x_columns)
counter = 0
for timestep in time.values.reshape(time.values.shape[0]):
    time_array = timestep * np.ones(len(x_columns))
    fittedValue = sess.run([V], feed_dict={tInt_tnsr: time_array.reshape(-1, 1), xInt_tnsr: np.array(x_columns).reshape(-1, 1)})
    out_df.loc[counter] = fittedValue[0].flatten()
    counter += 1

# Plot 2D
plots_2d(time, quantity, out_df)

# Split data in train test set, and repeat the process
sess.close()
init_op = tf.global_variables_initializer()

# open session
sess = tf.Session()
sess.run(init_op)

# Split the data set
time_split = 20
train_time = time[time['Time (day)'] < time_split]
train_q = quantity[time['Time (day)'] < time_split]

# %% Train network
# for each sampling stage
for i in range(sampling_stages):

    # sample uniformly from the required regions
    tInt, xInt, qInt, tTer, xTer, qTer = sampler(nSim_interior, nSim_terminal, train_time, train_q)

    # for a given sample, take the required number of SGD steps
    for _ in range(steps_per_sample):
        loss, L1, L3, _ = sess.run([loss_tnsr, L1_tnsr, L3_tnsr, optimizer],
                                   feed_dict={tInt_tnsr: tInt, xInt_tnsr: xInt, qInt_tnsr: qInt,
                                              tTer_tnsr: tTer, xTer_tnsr: xTer, qTer_tnsr: qTer})

    print(loss, L1, L3, i)

# Generate plots
slice_plots(time, quantity, sess, split=True)

# Predict function value at every grid point
out_df = pd.DataFrame(columns=x_columns)
counter = 0
for timestep in time.values.reshape(time.values.shape[0]):
    time_array = timestep * np.ones(len(x_columns))
    fittedValue = sess.run([V], feed_dict={tInt_tnsr: time_array.reshape(-1, 1), xInt_tnsr: np.array(x_columns).reshape(-1, 1)})
    out_df.loc[counter] = fittedValue[0].flatten()
    counter += 1

# Plot 2D
plots_2d(time, quantity, out_df, split=True)
