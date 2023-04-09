# Content-Based Image Retrieval Based On a Word Associations Network

This repository contains the implementation of semantical similarity between images based a word assications network as described in the following paper:

> [**Semantic Similarity Between Images: A Novel Approach Based on a Complex Network of Free Word Associations**][1]  
> Enrico Palumbo - Physics University of Torino, via Giuria, 1, 10025 Torino, Italy [enrico.palumbo@edu.unito.it](mailto:enrico.palumbo@edu.unito.it).
> Walter Allasia - EURIX, via Carcano, 26, 10153 Torino, Italy [allasia@eurix.it](mailto:allasia@eurix.it).

The data for word assocations is taken from the follwing site:

> [**University of South Florida Free Association Norms**][2]  
> Douglas L. Nelson and Cathy L. McEvoy - University of South Florida
> Thomas A. Schreiber - University of Kansas

If you use this code, please cite the above resources.

The rest of the repository deals with Content-Based Image Retrieval (CBIR) on the CIFAR-100 dataset. Additionally, other insights have been derived from the Word Associations Network.

## Quickrun
For 

## 2. Scripts
### 2.1. base_associations.py 
Computes image-word (10 words per image) associations for 1000 CIFAR-100 train dataset images starting from the given index. Results are saved in `Base Associations`.

```python3 ./base_associations.py <staring index>```

















[1]: https://enricopal.github.io/publications/Semantic%20Similarity%20between%20Images.pdf
[2]: http://w3.usf.edu/FreeAssociation/

