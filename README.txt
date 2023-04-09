base_associations.py: computes image-word associations 
for 1000 CIFAR-100 train dataset images starting from the given index.
result saved in Base Associations.
Run as: ./base_associations.py <starting index>

compute_base_associations.sh: computes all image-word associations for
the CIFAR-100 train dataset images. Calls base_associations.py internally
with different indices as arguments.
Run as: ./compute_base_associations.sh

compute_image_links.py: computes the related topIWK words for each of the CIFAR-100 train
dataset images. current value of topIWK is 4. results saved in image_links.pickle.

compute_word_links.py: computes the related topWWK words of a 
