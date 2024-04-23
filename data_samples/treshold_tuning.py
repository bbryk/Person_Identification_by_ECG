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


# test_num = 400
# margin_str = "03"
# dif = np.load(f"distances_400/{margin_str}_{test_num}_dif.npy")
# same = np.load(f"distances_400/{margin_str}_{test_num}_same.npy")

def generate_template(enroll_embeddings):
    template = np.mean(enroll_embeddings, axis=0)
    return template


def compare_with_template(test_embeddings, template):
    print(len(test_embeddings.shape))
    if len(test_embeddings.shape) > 1:
        norms_test = np.linalg.norm(test_embeddings, axis=1, keepdims=True)
    else:
        norms_test = np.linalg.norm(test_embeddings)

    normed_test_embeddings = test_embeddings / norms_test

    norm_template = np.linalg.norm(template)
    normed_template = template / norm_template

    cosine_similarities = np.dot(normed_test_embeddings, normed_template)

    cosine_distances = 1 - cosine_similarities

    return cosine_distances


def tune_threshold(embeddings, margin_str, num_train_subjects, num_test_subjects, enrollment_size, tuning_size):
    embeddings = np.load(f"embeddings/{num_train_subjects}/{margin_str}_{num_test_subjects}.npy", allow_pickle=True)
    #
    same_distances = []
    for class_id in range(1, num_test_subjects):
        class_embeddings = embeddings[class_id]

        enroll_embeddings = class_embeddings[:enrollment_size]
        tune_embeddings = class_embeddings[enrollment_size:enrollment_size + tuning_size]
        template = generate_template(enroll_embeddings)

        num_iterations = 1
        # subset_size = 5
        subset_size = 100

        indices = np.random.randint(tune_embeddings.shape[0], size=(num_iterations, subset_size))

        selected_subsets = tune_embeddings[indices]
        s1 = selected_subsets[0]

        same_cosine_distances = compare_with_template(s1, template)
        same_distances.extend(same_cosine_distances)

    print("Cosine Distances:", same_distances)
    print(len(same_distances))
    # plt.plot(same_distances, color='green')

    diff_distances = []

    for class_id in range(1, num_test_subjects):
        class_embeddings = embeddings[class_id]
        enroll_embeddings = class_embeddings[:10]
        template = generate_template(enroll_embeddings)

        for diff_class_id in range(class_id + 1, num_test_subjects):
            diff_class_embeddings = embeddings[diff_class_id]
            tune_embeddings = diff_class_embeddings[10:20]

            num_iterations = 1
            subset_size = 100

            indices = np.random.randint(tune_embeddings.shape[0], size=(num_iterations, subset_size))

            selected_subsets = tune_embeddings[indices]
            s1 = selected_subsets[0]
            print(s1.shape)
            print(selected_subsets.shape)
            # means_embeddings = np.mean(selected_subsets, axis=1)
            diff_cosine_distances = compare_with_template(s1, template)
            diff_distances.extend(list(diff_cosine_distances))

            # print("Cosine Distances:", diff_cosine_distances)
            # print("Shape of Cosine Distances:", cosine_distances.shape)
    # plt.plot(diff_distances, color='red')
    #
    # plt.show()

    dif = np.array(diff_distances).flatten()
    same = np.array(same_distances).flatten()
    np.save(f"tuning_distances_{num_train_subjects}/{margin_str}_{num_test_subjects}_dif.npy", dif)
    np.save(f"tuning_distances_{num_train_subjects}/{margin_str}_{num_test_subjects}_same.npy", same)
    low = 0.0
    high = 1.0
    epsilon = 0.001

    while high - low > epsilon / 100:
        threshold = (low + high) / 2
        FAR = 100 * len(dif[dif < threshold]) / len(dif)
        FRR = 100 * len(same[same > threshold]) / len(same)

        if abs(FAR - FRR) < epsilon:
            break

        if FAR > FRR:
            high = threshold
        else:
            low = threshold

    threshold = (low + high) / 2
    FAR = len(dif[dif < threshold]) / len(dif)
    FRR = len(same[same > threshold]) / len(same)
    FAR = "{:.7f}".format(FAR)
    FRR = "{:.7f}".format(FRR)

    print(f"Optimized Threshold: {threshold}")
    print(f"FAR {FAR}%")
    print(f"FRR {FRR}%")
    return threshold
