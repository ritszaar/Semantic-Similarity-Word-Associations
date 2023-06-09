# Content-Based Image Retrieval Based On a Word Associations Network

This repository contains the implementation of semantical similarity between images based a word assications network as described in the following paper:

> [**Semantic Similarity Between Images: A Novel Approach Based on a Complex Network of Free Word Associations**][1]  
> Enrico Palumbo - Physics University of Torino, via Giuria, 1, 10025 Torino, Italy [enrico.palumbo@edu.unito.it](mailto:enrico.palumbo@edu.unito.it)<br>
> Walter Allasia - EURIX, via Carcano, 26, 10153 Torino, Italy [allasia@eurix.it](mailto:allasia@eurix.it)<br>

The data for word assocations is taken from the follwing site:

> [**University of South Florida Free Association Norms**][2]  
> Douglas L. Nelson and Cathy L. McEvoy - University of South Florida<br>
> Thomas A. Schreiber - University of Kansas<br>

If you use this code, please cite the above resources.

The rest of the repository deals with Content-Based Image Retrieval (CBIR) on the CIFAR-100 dataset. Additionally, other insights have been derived from the Word Associations Network.

## 1. Quickrun
Before running any scripts ensure that you have the required dependencies. To install the dependencies, use:
```
bash ./install_dependencies.sh
```
### 1.1. All Test Results
For computing the test results for all 10000 test images in the CIFAR-100 dataset, use:
```
    python ./compute_test_results.py
    python ./compute_retrieval_performance.py --all
```
### 1.2. Partial Test Results
For computing the test results for a given number of test images in the CIFAR-100 dataset, use:
```
    python ./compute_partial_test_results.py  <number of images to be tested>
    python ./compute_retrieval_performance.py --partial
```

### 1.3. Interactive Demo
For an interactive demo, use:
```
    python ./interactive_demo.py
```

## 2. Longrun
Before running any scripts ensure that you have the required dependencies. To install the dependencies, use:
```
bash ./install_dependencies.sh
```
### 2.1. Compute Image-Word Associations
Every 32x32 image in the CIFAR-100 train dataset is cubic-interpolated to 224x224. The interpolated image is passed through the ResNet-50 model. The top 10 predictions of model along with its fine label and coarse label as provided by the dataset itself are saved in `Base Associations`.
```
    bash ./compute_base_associations.sh
```

### 2.2. Compute Words
A PostgreSQL database is to be setup named `word_associations`. This database will contain only a single relation named `usf_word_associations`. The relation will have fields `id`, `cue`, `target`, `strength`. Enter the records in the `Word Associations/usf_word_associations.csv` into this relation. The relevant connection parameters are the put inside the `conn` function of `compute_words.py`. We associate every image in the CIFAR-100 train dataset to `topIWK` words, each of which is in turn associated with `topWWK` words. The set of all such words are make up the words of the Word Associations Network. Data generated is saved in `word_links.py`. A text file `Word/all_words.txt` containing a list of all the words with ids is also generated.
```
    python compute_words.py
```

### 2.3. Compute Image-Word Links
The related `topIWK` words are to be saved as ids for each of the CIFAR-100 training images. Data generated is saved in `image_links.pickle`.
```
    python compute_image_links.py
```

### 2.4. Compute Word-Word Links
For the set of all the words in the Word Associations Network, we determine the pairwise link strengths by querying the database created in step 2.2 and save the data in `word_links.pickle`. It should the ensured that the proper connection parameters are put inside the `conn` function of `compute_word_links.py`. 
```
    python compute_word_links.py
```

### 2.5. Run the Model on The Test Dataset
The model is run on the CIFAR-100 test dataset. For every test image, the `topK` related images in the CIFAR-100 train dataset which are semantically similar to it are predicted. The predictions are saved in `test_results.pickle`. 
```
    python compute_test_results.py
```

### 2.6. Evaluate The Retrieval Performance
The retrieval performance of the model is evaluted. 
```
    python compute_retrieval_performance.py
```

## 3. Model Performance
### 3.1. 500 Test Images

| Metric                            |  Value  | 
| --------------------------------- | ------- | 
| Retrieval Time (without overhead) | 0.50s   |
| Retrieval Time (with overhead)    | 2.71s   | 
| mAP@1                             | 52.10%  |
| mAP@5                             | 53.47%  |
| mAP@10                            | 52.12%  |
| mAP@15                            | 51.56%  |
| mAP@20                            | 51.11%  |
| mAHP@1                            | 59.51%  |
| mAHP@5                            | 60.27%  |
| mAHP@10                           | 59.83%  |
| mAHP@15                           | 59.65%  |
| mAHP@20                           | 59.44%  |

### 3.2. All Test Images

| Metric                            |  Value  | 
| --------------------------------- | ------- | 
| Retrieval Time (without overhead) | 0.50s   |
| Retrieval Time (with overhead)    | 2.67s   |
| mAP@1                             | 52.07%  |
| mAP@5                             | 53.04%  |
| mAP@10                            | 52.57%  |
| mAP@15                            | 52.09%  |
| mAP@20                            | 51.70%  |
| mAHP@1                            | 59.93%  |
| mAHP@5                            | 60.29%  |
| mAHP@10                           | 59.88%  |
| mAHP@15                           | 59.64%  |
| mAHP@20                           | 59.42%  |


[1]: https://enricopal.github.io/publications/Semantic%20Similarity%20between%20Images.pdf
[2]: http://w3.usf.edu/FreeAssociation/