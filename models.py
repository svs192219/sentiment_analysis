# models.py

import tensorflow as tf
import numpy as np
import random
from sentiment_data import *


# Returns a new numpy array with the data from np_arr padded to be of length length. If length is less than the
# length of the base array, truncates instead.
def pad_to_length(np_arr, length, unk_idx):
    result = np.ones(length, dtype=np.int32) * unk_idx
    result[0:np_arr.shape[0]] = np_arr
    return result


# Train a feedforward neural network on the given training examples, using dev_exs for development and returning
# predictions on the *blind* test_exs (all test_exs have label 0 as a dummy placeholder value). Returned predictions
# should be SentimentExample objects with predicted labels and the same sentences as input (but these won't be
# read for evaluation anyway)
def train_ffnn(train_exs, dev_exs, test_exs, word_vectors):
    # 59 is the max sentence length in the corpus, so let's set this to 60
    seq_max_len = 60
    unk_idx = word_vectors.word_indexer.index_of("UNK")
    # To get you started off, we'll pad the training input to 60 words to make it a square matrix.
    train_mat = np.asarray([pad_to_length(np.array(ex.indexed_words, dtype=np.int32), seq_max_len, unk_idx) for ex in train_exs])
    dev_mat = np.asarray([pad_to_length(np.array(ex.indexed_words, dtype=np.int32), seq_max_len, unk_idx) for ex in dev_exs])
    test_mat = np.asarray([pad_to_length(np.array(ex.indexed_words, dtype=np.int32), seq_max_len, unk_idx) for ex in test_exs])
    # Also store the sequence lengths -- this could be useful for training LSTMs
    train_seq_lens = np.array([len(ex.indexed_words) for ex in train_exs])
    dev_seq_lens = np.array([len(ex.indexed_words) for ex in dev_exs])
    test_seq_lens = np.array([len(ex.indexed_words) for ex in test_exs])
    # Labels
    train_labels_arr = np.array([ex.label for ex in train_exs])
    dev_labels_arr = np.array([ex.label for ex in dev_exs])

    feat_vec_size = word_vectors.vectors.shape[1]
    train_size = train_mat.shape[0]
    num_labels = 2

    x = tf.placeholder(tf.int32, [None, seq_max_len])
    y_ = tf.placeholder(tf.int32, [None])

    keep_prob = tf.placeholder(tf.float32)

    seq = tf.placeholder(tf.int32, [None])
    embed_mat_trainable = tf.get_variable("word_vectors", dtype=tf.float32, initializer=tf.constant(word_vectors.vectors, dtype=tf.float32))
    embeddings = tf.cast(tf.nn.embedding_lookup(embed_mat_trainable, x), tf.float32)
    ns = [tf.shape(embeddings)[0], seq_max_len, 1]
    embeddings = tf.nn.dropout(embeddings, keep_prob, noise_shape=ns)
    input_data = tf.reduce_mean(embeddings, 1)
    seq1 = tf.reshape(tf.divide(seq_max_len, tf.cast(seq, dtype=tf.float32)), [-1, 1])
    seq1 = tf.tile(seq1, [1, feat_vec_size])
    input_data = tf.multiply(input_data, seq1)
    y_one_hot = tf.one_hot(y_, depth=num_labels)

    num_h1 = 300
    num_h2 = 300

    def get_fclayer(input, in_size, out_size, names, final=False):
        W = tf.get_variable(names[0], [in_size, out_size], initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable(names[1], [out_size], initializer=tf.contrib.layers.xavier_initializer())

        out = tf.add(tf.matmul(input, W), b)
        if not final:
            out = tf.nn.dropout(out, keep_prob)
            out = tf.nn.relu(out)

        return out

    l1 = get_fclayer(input_data, feat_vec_size, num_h1, ['w1', 'b1'])
    l2 = get_fclayer(l1, num_h1, num_h2, ['w2', 'b2'])
    pred = get_fclayer(l2, num_h2, num_labels, ['w3', 'b3'], final=True)

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y_one_hot))

    train_vars = tf.trainable_variables()

    decay_steps = 100
    learning_rate_decay_factor = 0.99
    global_step = tf.train.get_or_create_global_step()
    # Smaller learning rates are sometimes necessary for larger networks
    initial_learning_rate = 0.001
    # Decay the learning rate exponentially based on the number of steps.
    lr1 = tf.train.exponential_decay(initial_learning_rate,
                                    global_step,
                                    decay_steps,
                                    learning_rate_decay_factor,
                                    staircase=True)
    lr2 = tf.train.exponential_decay(initial_learning_rate*0.5,
                                    global_step,
                                    decay_steps,
                                    learning_rate_decay_factor,
                                    staircase=True)

    optimizer1 = tf.train.AdamOptimizer(learning_rate=lr1).minimize(loss)
    # optimizer1 = tf.train.AdamOptimizer(learning_rate=lr1).minimize(loss, var_list=train_vars[1:])
    # optimizer2 = tf.train.AdamOptimizer(learning_rate=lr2).minimize(loss, var_list=train_vars[0:1])


    predictions = tf.argmax(pred, 1)
    correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y_one_hot, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    init = tf.global_variables_initializer()

    total_epochs = 50

    batch_size = 15
    splits = train_size/batch_size
    batch_sizes = (np.arange(splits-1, dtype=np.int32) * batch_size) + batch_size
    if splits*batch_size < train_size:
        batch_sizes = np.append(batch_sizes, [splits*batch_size])

    # print(batch_sizes)

    with tf.Session() as sess:
        sess.run(init)

        batches_x = np.split(train_mat, batch_sizes)
        batches_y = np.split(train_labels_arr, batch_sizes)
        batches_l = np.split(train_seq_lens, batch_sizes)
        
        for epoch in range(total_epochs):

            avg_loss = 0
            for idx in range(len(batch_sizes)):
                batch_x = batches_x[idx]
                batch_y = batches_y[idx]
                batch_l = batches_l[idx]

                _, l = sess.run([optimizer1, loss], feed_dict={x: batch_x, y_: batch_y, seq: batch_l, keep_prob: 0.7})

                avg_loss += l / len(batch_sizes)
            
                # if idx % 100 == 0:
                #     print('Epoch : %d, Batch : %d, loss : %.3f' % (epoch+1, idx, l))

            print('Epoch : %d, avg loss : %.3f' % (epoch+1, avg_loss))

            # if epoch % 5 == 0:
            t_acc = sess.run(accuracy, feed_dict={x: train_mat, y_: train_labels_arr, seq: train_seq_lens, keep_prob: 1.0})
            v_acc = sess.run(accuracy, feed_dict={x: dev_mat, y_: dev_labels_arr, seq: dev_seq_lens, keep_prob: 1.0})
            print('Train accuracy : %.3f%%, Valid accuracy : %.3f%%' % (t_acc*100, v_acc*100))

        test_pred = sess.run(predictions, feed_dict={x: test_mat, seq: test_seq_lens, keep_prob: 1.0})

    for idx, ex in enumerate(test_exs):
        ex.label = test_pred[idx]

    return test_exs


# Analogous to train_ffnn, but trains your fancier model.
def train_fancy(train_exs, dev_exs, test_exs, word_vectors):
    seq_max_len = 60
    unk_idx = word_vectors.word_indexer.index_of("UNK")
    # To get you started off, we'll pad the training input to 60 words to make it a square matrix.
    train_mat = np.asarray([pad_to_length(np.array(ex.indexed_words, dtype=np.int32), seq_max_len, unk_idx) for ex in train_exs])
    dev_mat = np.asarray([pad_to_length(np.array(ex.indexed_words, dtype=np.int32), seq_max_len, unk_idx) for ex in dev_exs])
    test_mat = np.asarray([pad_to_length(np.array(ex.indexed_words, dtype=np.int32), seq_max_len, unk_idx) for ex in test_exs])
    # Also store the sequence lengths -- this could be useful for training LSTMs
    train_seq_lens = np.array([len(ex.indexed_words) for ex in train_exs])
    dev_seq_lens = np.array([len(ex.indexed_words) for ex in dev_exs])
    test_seq_lens = np.array([len(ex.indexed_words) for ex in test_exs])
    # Labels
    train_labels_arr = np.array([ex.label for ex in train_exs])
    dev_labels_arr = np.array([ex.label for ex in dev_exs])

    feat_vec_size = word_vectors.vectors.shape[1]
    train_size = train_mat.shape[0]
    num_labels = 2

    x = tf.placeholder(tf.int32, [None, seq_max_len])
    y_ = tf.placeholder(tf.int32, [None])

    keep_prob = tf.placeholder(tf.float32)

    embed_mat_trainable = tf.get_variable("word_vectors", dtype=tf.float32, initializer=tf.constant(word_vectors.vectors, dtype=tf.float32))
    embeddings = tf.cast(tf.nn.embedding_lookup(embed_mat_trainable, x), tf.float32)
    y_one_hot = tf.one_hot(y_, depth=num_labels)


    lstmUnits = 50

    seq = tf.placeholder(tf.int32, [None])

    lstm_fw = tf.contrib.rnn.GRUCell(lstmUnits)
    lstm_fw = tf.contrib.rnn.DropoutWrapper(cell=lstm_fw, output_keep_prob=keep_prob)
    lstm_bw = tf.contrib.rnn.GRUCell(lstmUnits)
    lstm_bw = tf.contrib.rnn.DropoutWrapper(cell=lstm_bw, output_keep_prob=keep_prob)
    # rnn_outputs, _ = tf.nn.dynamic_rnn(lstm_fw, embeddings, sequence_length=seq, dtype=tf.float32)
    rnn_outputs, _, _ = tf.contrib.rnn.stack_bidirectional_dynamic_rnn([lstm_fw], [lstm_bw], embeddings, sequence_length=seq, dtype=tf.float32)


    # rnn_outputs = tf.transpose(rnn_outputs, [1, 0, 2])
    gather_idx = tf.placeholder(tf.int32, [None, 2])
    input_data = tf.gather_nd(rnn_outputs, gather_idx)
    # input_data = tf.reshape(input_data, [-1, 2, lstmUnits])
    # input_data = tf.reduce_prod(input_data, 1)

    num_h1 = 100
    num_h2 = 300

    def get_fclayer(input, in_size, out_size, names, final=False):
        W = tf.get_variable(names[0], [in_size, out_size], initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable(names[1], [out_size], initializer=tf.contrib.layers.xavier_initializer())

        out = tf.add(tf.matmul(input, W), b)
        if not final:
            out = tf.nn.relu(out)

        return out

    # l1 = get_fclayer(input_data, lstmUnits, num_h1, ['w1', 'b1'])
    # l2 = get_fclayer(l1, num_h1, num_h2, ['w2', 'b2'])
    pred = get_fclayer(input_data, lstmUnits*2, num_labels, ['w3', 'b3'], final=True)

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y_one_hot))

    trainable_var = tf.trainable_variables()

    decay_steps = 100
    learning_rate_decay_factor = 0.99
    global_step = tf.train.get_or_create_global_step()
    # Smaller learning rates are sometimes necessary for larger networks
    initial_learning_rate = 0.01
    # Decay the learning rate exponentially based on the number of steps.
    lr1 = tf.train.exponential_decay(initial_learning_rate,
                                    global_step,
                                    decay_steps,
                                    learning_rate_decay_factor,
                                    staircase=True)
    lr2 = tf.train.exponential_decay(initial_learning_rate*0.1,
                                    global_step,
                                    decay_steps,
                                    learning_rate_decay_factor,
                                    staircase=True)

    optimizer1 = tf.train.AdamOptimizer(learning_rate=lr1).minimize(loss, var_list=trainable_var[1:])
    optimizer2 = tf.train.AdamOptimizer(learning_rate=lr2).minimize(loss, var_list=trainable_var[0:1])


    predictions = tf.argmax(pred, 1)
    correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y_one_hot, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    init = tf.global_variables_initializer()

    total_epochs = 1

    batch_size = 30
    splits = train_size/batch_size
    batch_sizes = (np.arange(splits-1, dtype=np.int32) * batch_size) + batch_size
    if splits*batch_size < train_size:
        batch_sizes = np.append(batch_sizes, [splits*batch_size])

    # print(batch_sizes)

    def get_gather_indices(a):
        tmp1 = np.arange(a.shape[0])
        tmp1 = np.reshape(tmp1, [tmp1.shape[0], 1])
        tmp2 = np.reshape(a-1, [a.shape[0], 1])
        return np.concatenate((tmp1, tmp2), axis=1)

    with tf.Session() as sess:
        sess.run(init)

        batches_x = np.split(train_mat, batch_sizes)
        batches_y = np.split(train_labels_arr, batch_sizes)
        batches_l = np.split(train_seq_lens, batch_sizes)

        for epoch in range(total_epochs):

            avg_loss = 0
            for idx in range(len(batch_sizes)):
                batch_x = batches_x[idx]
                batch_y = batches_y[idx]
                batch_l = batches_l[idx]

                gather_list = get_gather_indices(batch_l)
                # print(gather_list)

                _, _, l = sess.run([optimizer1, optimizer2, loss], feed_dict={x: batch_x, y_: batch_y, \
                                                            seq: batch_l, gather_idx: gather_list, keep_prob: 1.0})

                # print(r)
                # print(i)
                avg_loss += l / len(batch_sizes)
            
                # if idx % 100 == 0:
                #     print('Epoch : %d, Batch : %d, loss : %.3f' % (epoch+1, idx, l))

            print('Epoch : %d, avg loss : %.3f' % (epoch+1, avg_loss))

            # if epoch % 5 == 0:
            t_acc = sess.run(accuracy, feed_dict={x: train_mat, y_: train_labels_arr, seq: train_seq_lens,\
                                gather_idx: get_gather_indices(train_seq_lens), keep_prob: 1.0})
            v_acc = sess.run(accuracy, feed_dict={x: dev_mat, y_: dev_labels_arr, seq: dev_seq_lens, \
                                gather_idx: get_gather_indices(dev_seq_lens), keep_prob: 1.0})
            print('Train accuracy : %.3f%%, Valid accuracy : %.3f%%' % (t_acc*100, v_acc*100))

        test_pred = sess.run(predictions, feed_dict={x: test_mat, seq: test_seq_lens, \
                                gather_idx: get_gather_indices(test_seq_lens), keep_prob: 1.0})

    for idx, ex in enumerate(test_exs):
        ex.label = test_pred[idx]

    return test_exs
