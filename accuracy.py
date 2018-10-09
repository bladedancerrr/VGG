
#Evaluate on a random sample
sample_size = 500

chosen_indices = np.random.choice(len(centered_test), sample_size)

chosen_images = centered_test[chosen_indices]
chosen_labels = test_labels[chosen_indices]

results = sess.run(predictions, feed_dict={ input_data: chosen_images, labels: chosen_labels })

predicted_classes = np.argmax(results, axis=1)

acc = accuracy_score(chosen_labels, predicted_classes)

print(acc)
