import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def generate_template(enroll_embeddings):
    template = np.mean(enroll_embeddings, axis=0)
    return template

# def compare_with_template(test_embedding, template):
#     norm_embeddings = test_embedding / np.linalg.norm(test_embedding)
#     norm_template = template / np.linalg.norm(template)
#     # print(norm_enrolled)
#     # print(norm_verification)
#     cosine_similarity = np.dot(norm_embeddings.T, norm_template)
#     print(cosine_similarity)
#     print(cosine_similarity.shape)
#     cosine_distance = 1 - cosine_similarity
#
#     return cosine_distance


def compare_with_template(test_embeddings, template):
    norms_test = np.linalg.norm(test_embeddings, axis=1, keepdims=True)
    normed_test_embeddings = test_embeddings / norms_test

    norm_template = np.linalg.norm(template)
    normed_template = template / norm_template

    cosine_similarities = np.dot(normed_test_embeddings, normed_template)

    cosine_distances = 1 - cosine_similarities

    return cosine_distances

embeddings = np.load("embeddings_means_test.npy", allow_pickle=True)

same_distances = []

for class_id in range(1,19):
    class_embeddings = embeddings[class_id]

    enroll_embeddings = class_embeddings[:10]
    test_embeddings = class_embeddings[10:]
    template  = generate_template(enroll_embeddings)

    # print(f"template shape: {template.shape}")
    # print(f"test  shape: {test_embeddings.shape}")
    same_cosine_distances = compare_with_template(test_embeddings, template)
    same_distances.extend(same_cosine_distances)
    plt.plot(same_cosine_distances,color='green')


    print("Cosine Distances:", same_cosine_distances)
    # print("Shape of Cosine Distances:", cosine_distances.shape)
# plt.show()

diff_distances = []
for class_id in range(1,19):
    class_embeddings = embeddings[class_id]
    enroll_embeddings = class_embeddings[:10]
    template = generate_template(enroll_embeddings)

    for diff_class_id in range(class_id+1,19):
        diff_class_embeddings = embeddings[diff_class_id]

        # enroll_embeddings = diff_class_embeddings[:10]
        # test_embeddings = diff_class_embeddings[10:]


        diff_cosine_distances = compare_with_template(diff_class_embeddings, template)
        diff_distances.extend(list(diff_cosine_distances))
        plt.plot(diff_cosine_distances)


        print("Cosine Distances:", diff_cosine_distances)
        # print("Shape of Cosine Distances:", cosine_distances.shape)
plt.show()

dif = np.array(diff_distances).flatten()
same = np.array(same_distances).flatten()

print(np.array(same).shape)

FAR = len(dif[dif < 0.3]) / len(dif)
FAR = "{:.7f}".format(100 * FAR)
FRR = len(same[same > 0.3]) / len(same)
FRR = "{:.7f}".format(100 * FRR)
print(f"FAR {FAR}%")
print(f"FRR {FRR}%")
