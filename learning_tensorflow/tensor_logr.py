import tensorflow as tf
import os

def read_csv(batch_size, file_name, record_defaults):
    #filename_queue = tf.train.string_input_producer([os.path.dirname(__file__) + "/" + file_name])
    filename_queue = tf.train.string_input_producer([file_name])
    reader = tf.TextLineReader(skip_header_lines=1)
    key, value = reader.read(filename_queue)
    #key, value = reader.read(file_name)
    # decode_csv will convert a Tensor from type string (the text line) in
    # a tuple of tensor columns with the specified defaults, which also
    # sets the data type for each column
    decoded = tf.decode_csv(value, record_defaults=record_defaults)
    # batch actually reads the file and loads "batch_size" rows in a single tensor
    return tf.train.shuffle_batch(decoded,
                batch_size=batch_size,
                capacity=batch_size * 50,
                min_after_dequeue=5)


def combine_inputs(X):
    return tf.matmul(X, W) + b


def loss(X, Y):
    # compute loss over training data X and expected outputs Y
    #Y_predicted = inference(X)
    #return tf.reduce_sum(tf.squared_difference(Y, Y_predicted))
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(combine_inputs(X), Y))

def train(total_loss):
    # train / adjust model parameters according to computed total loss
    learning_rate = 0.01
    print("in train")
    return tf.train.GradientDescentOptimizer(learning_rate).minimize(total_loss)

W = tf.Variable(tf.zeros([5, 1]), name="weights")
b = tf.Variable(0., name="bias")

# this all tensors


with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    v0 = sess.run(W)    
    print(v0) # will show you your variable.
    v1 = sess.run(b)
    print(v1)
    
    #v2 = sess.run(pid)
    #print(v2)


    pid, survived, pclass, sex, age = read_csv(100, "train.csv", [[0.0], [0.0], [0],[""],[0.0]])

    first_class = tf.to_float(tf.equal(pclass,[1]))
    second_class = tf.to_float(tf.equal(pclass, [2]))
    third_class = tf.to_float(tf.equal(pclass, [3]))
    gender = tf.to_float(tf.equal(sex, ["female"]))

    features = tf.transpose(tf.pack([first_class, second_class, third_class, gender, age]))
    survived = tf.reshape(survived, [100, 1])
    
    X, Y = features, survived
    
    print("done X Y")
    total_loss = loss(X, Y)
    print("done total loss")
    train_op = train(total_loss)
    coord = tf.train.Coordinator()
    print("after coord")
    threads = tf.train.start_queue_runners(sess = sess,coord = coord)
    training_steps = 5000
    print("after threads")
    for step in range(training_steps):
        sess.run([train_op])
        if step % 100 == 0: print("loss:", step, sess.run([total_loss]))
    coord.request_stop()
    coord.join(threads)
    sess.close()
