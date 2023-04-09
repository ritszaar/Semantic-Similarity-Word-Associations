base_associations.py: Computes image-word (10 words per image) associations 
for 1000 CIFAR-100 train dataset images starting from the given index.
Result saved in Base Associations.
Run as: ./base_associations.py <starting index>

compute_base_associations.sh: Computes all image-word (10 words per image) associations for
the CIFAR-100 train dataset images. Calls base_associations.py internally
with different indices as arguments.
Run as: ./compute_base_associations.sh

compute_words.py: Determines all the words of the Word Associations Network. Each image is associated with topIWK words each of which is in
turn associated with topWWK words. The set of all these words is computed and the result is saved in words.pickle. Note that a CIFAR-100 image
already has a file label which we are NOT including 

compute_image_links.py: computes the related topIWK words for each of the CIFAR-100 tr

compute_word_links.py: Computes the related topWWK words of a given word. Current value of topWWK is 4.
data saved in word_links.py.
