batch_size    = 32
train_steps   = 5000
learning_rate = 1e-3

num_images = len(train_images)

run_name = "vgg/" + time.strftime("%d-%b-%Y-%H-%M-%S")
print(f"Run name: {run_name}")

sess = tf.Session()
train_writer = tf.summary.FileWriter(f"{LOG_DIR}/{run_name}/train", sess.graph)
test_writer  = tf.summary.FileWriter(f"{LOG_DIR}/{run_name}/test",)

# TODO: Train the model here!

gd = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)
train_op = gd.minimize(loss)
summary_op = tf.summary.merge_all()

sess.run(tf.global_variables_initializer())

for i in tqdm.tqdm(range(train_steps)):
    
    batch_im, batch_lab = sample_batch(centered_train, train_labels, batch_size=batch_size)
    _,  train_loss, summary = sess.run([train_op, loss, summary_op], feed_dict={
            input_data: batch_im, labels: batch_lab})
    
    train_writer.add_summary(summary, global_step=i)

    if i % 10 == 0:
        batch_im, batch_lab = sample_batch(centered_train, train_labels, batch_size=batch_size)
        test_acc, test_loss, summary = sess.run([accuracy, loss, summary_op], feed_dict={
            input_data: batch_im, labels: batch_lab})

        test_writer.add_summary(summary, global_step=i)

print("Final test-set accuracy: {test_acc}.")
