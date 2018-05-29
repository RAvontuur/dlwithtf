import numpy as np
np.random.seed(456)
import  tensorflow as tf
tf.set_random_seed(456)
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# Generate synthetic data
N = 100
N2 = 50
w_true = 5
b_true = 2
noise_scale = .1
# Zeros form a Gaussian centered at (-1, -1)
x_zeros = np.random.multivariate_normal(
    mean=np.array((0, 0)), cov=.1*np.eye(2), size=(N2,))
y_zeros = np.zeros((N2,))
# Ones form a Gaussian centered at (1, 1)
x_ones = np.random.multivariate_normal(
    mean=np.array((0, 0)), cov=1.*np.eye(2), size=(N2,))
y_ones = np.ones((N2,))

x_np = np.vstack([x_zeros, x_ones])
y_np = np.concatenate([y_zeros, y_ones])


# Save image of the data distribution
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.title("FCNet Classification Data")

# Plot Zeros
plt.scatter(x_zeros[:, 0], x_zeros[:, 1], color="blue")
plt.scatter(x_ones[:, 0], x_ones[:, 1], color="red")
plt.savefig("fcnet_classification_data.png")

# Generate tensorflow graph
d = 2
n_hidden = 15
with tf.name_scope("placeholders"):
  x = tf.placeholder(tf.float32)
  y = tf.placeholder(tf.float32)
with tf.name_scope("layer-1"):
  W = tf.Variable(tf.random_normal((d, n_hidden)))
  b = tf.Variable(tf.random_normal((n_hidden,)))
  x_1 = tf.nn.relu(tf.matmul(x, W) + b)
with tf.name_scope("output"):
  W = tf.Variable(tf.random_normal((n_hidden, 1)))
  b = tf.Variable(tf.random_normal((1,)))
  y_logit = tf.squeeze(tf.matmul(x_1, W) + b)
  # the sigmoid gives the class probability of 1
  y_one_prob = tf.sigmoid(y_logit)
  # Rounding P(y=1) will give the correct prediction.
  y_pred = tf.round(y_one_prob)
with tf.name_scope("loss"):
  # Compute the cross-entropy term for each datapoint
  entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=y_logit, labels=y)
  # Sum all contributions
  l = tf.reduce_sum(entropy)

with tf.name_scope("optim"):
  train_op = tf.train.AdamOptimizer(.001).minimize(l)

with tf.name_scope("summaries"):
  tf.summary.scalar("loss", l)
  merged = tf.summary.merge_all()

train_writer = tf.summary.FileWriter('/tmp/fcnet-classification-train',
                                     tf.get_default_graph())

n_steps = 801
with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  # Train model
  for i in range(n_steps):
    feed_dict = {x: x_np, y: y_np}
    _, summary, loss = sess.run([train_op, merged, l], feed_dict=feed_dict)
    # print("step %d, loss: %f" % (i, loss))
    train_writer.add_summary(summary, i)

    if (i % 10 == 0) :
      # Make Predictions
      y_pred_np = sess.run(y_pred, feed_dict={x: x_np})
      score = accuracy_score(y_np, y_pred_np)
      print("Classification Accuracy: %f" % score)

      vec0 = np.linspace(np.min(x_np[:, 0]), np.max(x_np[:, 0]), num=100)
      vec1 = np.linspace(np.min(x_np[:, 1]), np.max(x_np[:, 1]), num=100)
      xv, yv = np.meshgrid(vec0, vec1)

      x_sim_np = np.transpose(np.vstack([np.reshape(xv,10000), np.reshape(yv,10000)]))
      y_sim_np = sess.run(y_pred, feed_dict={x: x_sim_np})

      plt.clf()
      plt.xlabel("Dimension 1")
      plt.ylabel("Dimension 2")
      plt.title("FCNet Classification Simulations")
      plt.scatter(x_zeros[:, 0], x_zeros[:, 1], color="blue")
      plt.scatter(x_ones[:, 0], x_ones[:, 1], color="red")
      plt.scatter(x_sim_np[:, 0], x_sim_np[:, 1], c=y_sim_np, s=1)
      plt.pause(0.1)

  print("ready.")
  plt.show()
  plt.savefig("fcnet_classification_sim.png")

# plt.clf()
# plt.xlabel("Dimension 1")
# plt.ylabel("Dimension 2")
# plt.title("FCNet Classification Predictions")
# plt.scatter(x_np[:, 0], x_np[:, 1], c=y_pred_np)
# plt.savefig("fcnet_classification_pred.png")
#

