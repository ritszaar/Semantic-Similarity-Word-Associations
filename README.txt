base_associations.py: Computes image-word (10 words per image) associations 
for 1000 CIFAR-100 train dataset images starting from the given index.
Results are saved in Base Associations.
Run as: ./base_associations.py <starting index>

compute_base_associations.sh: Computes all image-word (10 words per image) associations for
the CIFAR-100 train dataset images. Calls base_associations.py internally
with different indices as arguments. Results are saved in Base Associations.
Run as: ./compute_base_associations.sh

compute_words.py: Determines all the words of the Word Associations Network. Each image is associated with topIWK words each of which is in
turn associated with topWWK words. The set of all these words is computed and the result is saved in words.pickle. Note that a CIFAR-100 image
has a fine label which may or may not be included as a related word of the image depending on which the performance on test data may change.

compute_image_links.py: Saves the related topIWK words as ids for each of the CIFAR-100 training images. Results are saved in image_links.pickle.

compute_word_links.py: Determines the related topWWK words for each word in the Word Associations Network. Results are saved in word_links.pickle.

compute_test_results: Computes the predicions of the model for the CIFAR-100 test dataset. Given a test image, topK images in the CIFAR-100 train
dataset which are semantically similar to it are retrieved. Query times are also evaluted. Results are saved in test_results.pickle.

interactive_demo.py: Provides an interactive demo for the model. Requires the user to enter index of a image in the CIFAR-100 test dataset. The topk images
in the CIFAR-100 train dataset which are semantically similar to it are retrieved.

compute_retrieval_performance.py: Computes the retrieval performance from the predictions of the model saved in test_results.pickle.
