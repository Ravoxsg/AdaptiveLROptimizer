#Same as train.py, but with an adaptive learning rate schedule.

import numpy as np 
import tensorflow as tf
from data import cifar10, utilities

import vgg


# Config:

BATCH_SIZE = 64
NUM_EPOCHS = 0.2
STEP = 20
DATASET_SIZE = 50000

INITIAL_LR = 0.0001
EPSILON = 0.0000001

network = 'vgg'
mode = 'adaptive_2nd_order'
logdir = 'cnn_{}_{}/train_logs/'.format(network,mode)

# Set up training data:
NUM_BATCHES = int(NUM_EPOCHS * DATASET_SIZE / BATCH_SIZE)
data_generator = utilities.infinite_generator(cifar10.get_train(), BATCH_SIZE)

# Define the placeholders:
n_input = tf.placeholder(tf.float32, shape=cifar10.get_shape_input(), name="input")
n_label = tf.placeholder(tf.int64, shape=cifar10.get_shape_label(), name="label")

# Build the model
n_output = vgg.build(n_input)

# Define the loss function
loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=n_output, labels=n_label, name="softmax"))
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(n_output, axis=1), n_label), tf.float32))

# Add summaries to track the state of training:
tf.summary.scalar('summary/loss', loss)
tf.summary.scalar('summary/accuracy', accuracy)
summaries = tf.summary.merge_all()

# Define training operations:
global_step = tf.Variable(0, trainable=False, name='global_step')
inc_global_step = tf.assign(global_step, global_step+1)

# Adaptive learning rate variables

lr = tf.Variable(INITIAL_LR, name='lr')

lr1 = tf.Variable(INITIAL_LR - EPSILON, name = 'lr1')
lr2 = tf.Variable(INITIAL_LR + EPSILON, name = 'lr2')
lr3 = tf.Variable(INITIAL_LR - 2*EPSILON, name = 'lr3')
lr4 = tf.Variable(INITIAL_LR + 2*EPSILON, name = 'lr4')

loss1 = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=n_output, labels=n_label, name="softmax"))
loss2 = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=n_output, labels=n_label, name="softmax"))
loss3 = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=n_output, labels=n_label, name="softmax"))
loss4 = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=n_output, labels=n_label, name="softmax"))

train_op_0 = tf.train.GradientDescentOptimizer(learning_rate = lr, use_locking=True).minimize(loss)
train_op = tf.train.GradientDescentOptimizer(learning_rate = lr).minimize(loss)

train_op_1_0 = tf.train.GradientDescentOptimizer(learning_rate = lr1, use_locking=True).minimize(loss1)
train_op_1 = tf.train.GradientDescentOptimizer(learning_rate = lr1).minimize(loss1)
train_op_2_0 = tf.train.GradientDescentOptimizer(learning_rate = lr2, use_locking=True).minimize(loss2)
train_op_2 = tf.train.GradientDescentOptimizer(learning_rate = lr2).minimize(loss2)
train_op_3_0 = tf.train.GradientDescentOptimizer(learning_rate = lr3, use_locking=True).minimize(loss3)
train_op_3 = tf.train.GradientDescentOptimizer(learning_rate = lr3).minimize(loss3)
train_op_4_0 = tf.train.GradientDescentOptimizer(learning_rate = lr4, use_locking=True).minimize(loss4)
train_op_4 = tf.train.GradientDescentOptimizer(learning_rate = lr4).minimize(loss4)

inc_lr = tf.assign(lr, lr - 2*EPSILON*((loss2 - loss1 + EPSILON)/(loss3 + loss4 - 2*loss + EPSILON)))

inc_lr1 = tf.assign(lr1, lr - EPSILON)
inc_lr2 = tf.assign(lr2, lr + EPSILON)
inc_lr3 = tf.assign(lr3, lr - 2*EPSILON)
inc_lr4 = tf.assign(lr4, lr + 2*EPSILON) 

# Keeping track of the loss and the learning rate
batches = []
losses = []
lrs = []

print("Loading training supervisor...")
sv = tf.train.Supervisor(logdir=logdir, global_step=global_step, summary_op=None, save_model_secs=30)
print("Done!")

with sv.managed_session() as sess:
    # Get the current global_step
    batch = sess.run(global_step)

    # Set up tensorboard logging:
    logwriter = tf.summary.FileWriter(logdir, sess.graph)
    logwriter.add_session_log(tf.SessionLog(status=tf.SessionLog.START), global_step=batch)

    print("Starting training from batch {} to {}. Saving model every {}s.".format(batch, NUM_BATCHES, 30))

    while not sv.should_stop():

        if batch >= NUM_BATCHES:
            print("Saving...")
            sv.saver.save(sess, logdir+'model.ckpt', global_step=batch)
            sv.stop()
            break

        inp, lbl = next(data_generator)

        current_loss, _ = sess.run((loss, (train_op_0)), feed_dict={n_input: inp,n_label: lbl})

        sess.run((inc_lr1, inc_lr2, inc_lr3, inc_lr4))

        loss_1, _ = sess.run((loss1, (train_op_1_0)), feed_dict={n_input: inp,n_label: lbl})
        loss_2, _ = sess.run((loss2, (train_op_2_0)), feed_dict={n_input: inp,n_label: lbl})
        loss_3, _ = sess.run((loss3, (train_op_3_0)), feed_dict={n_input: inp,n_label: lbl})
        loss_4, _ = sess.run((loss4, (train_op_4_0)), feed_dict={n_input: inp,n_label: lbl})

        summ, loss_0, _ = sess.run((summaries, loss, (train_op, inc_global_step)), feed_dict={n_input: inp,n_label: lbl})

        sess.run(inc_lr, feed_dict={n_input: inp, n_label: lbl})

        if batch > 0 and batch % 20 == 0:
            print('Step {} of {}.'.format(batch, NUM_BATCHES))
            print('Current loss: {}'.format(current_loss))
            print('Same as above: {}'.format(loss_0))
            print('loss1, loss2, loss3, loss4: {}, {}, {}, {}'.format(loss_1, loss_2, loss_3, loss_4))
            print('Learning rate: {}'.format(lr))
        
        logwriter.add_summary(summ, global_step=batch)

        batch += 1

print("Halting.")

plt.plot(np.array(batches),np.array(losses))
plt.tile('Loss over time')
plt.show()

plt.plot(np.array(batches),np.array(lrs))
plt.title('Learning rate over time')
plt.show()