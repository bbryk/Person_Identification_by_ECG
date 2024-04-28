from torch.utils.data import DataLoader, Dataset
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from model import ResidualBlock, ResNet1D
from angular_margin_loss import AngularSoftmaxLoss
from treshold_tuning import tune_threshold
# from dataset import IDDataset
import time
import random
class IDDataset(Dataset):
    def __init__(self, data_folder,to_test, num_samples):
        self.data_paths = []
        self.labels = []
        ind = 0

        train_range = list(range(1, 600))
        test_range = list(range(601, 1122))

        sample_range = test_range if to_test else train_range
        seed = 1736
        if seed is not None:
            random.seed(seed)
            random.shuffle(sample_range)

        for label in test_range:

            subfolder_names = [int(name) for name in os.listdir(data_folder)
                               if os.path.isdir(os.path.join(data_folder, name))]
            # print(subfolder_names)
            if label not in subfolder_names:
                continue

            print(f"{ind}: {label}")

            ind += 1



            class_folder = os.path.join(data_folder, str(label))
            for filename in os.listdir(class_folder):
                if filename.endswith('.npy'):
                    self.data_paths.append(os.path.join(class_folder, filename))
                    self.labels.append(ind)
            if ind>num_samples-1:
                break

        print(f"IND: {ind}")

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        file_path = self.data_paths[idx]  # Capture the file path
        data = np.load(file_path)
        if data.ndim != 1:
            raise ValueError(f"Data must be 1-dimensional, found in file: {file_path}")

        # Reshape the data and convert to tensor
        data = data.reshape(1, -1)
        data_tensor = torch.from_numpy(data).float()

        # Check for NaN values in the data tensor, now including file path in the error message
        assert not torch.isnan(data_tensor).any(), f"NaN values found in data at index: {idx}, file: {file_path}"

        label = self.labels[idx] - 1
        return data_tensor, label
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




if __name__ == "__main__":

    for margin_str in ["01", "03", "05"]:
        # margin_str ="05"
        # for num_test in [20, 50, 100]:
        for num_test in [50, 100]:
            seed = 1736
            np.random.seed(seed)
            start_time = time.time()




            # margin_str = "01"
            num_train_subjects = 100
            # num_test = 250
            to_test = True
            batch_size = 200
            ENROLL_SIZE = 10
            THRESHOLD_TUNING_SIZE = 10
            data_folder_test = 'diplom_test/git_ecg_samples'
            model = ResNet1D()
            torch_dict = torch.load(f'git_test/models_{num_train_subjects}/ver2_m_{margin_str}_{num_train_subjects}.pth')

            model.load_state_dict(torch_dict)
            model.eval()
            # print(model)

            # data_folder_test = 'diplom_test/prev_ecg/ecg_test_samples_428Hz'
            dataset_test = IDDataset(data_folder_test, to_test=to_test, num_samples=num_test)
            dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)

            embeddings_test = []


            def hook_fn(module, input, output):
                embeddings_test.append(output.cpu().data.numpy())


            hook = model.fc.register_forward_hook(hook_fn)
            model.eval()
            with torch.no_grad():
                c = 0
                for inputs, labels in dataloader_test:
                    c += 1
                    print(f"{c}/{len(dataloader_test)}")
                    outputs = model(inputs)

            # np.save(f"embeddings/{num_train_subjects}/{margin_str}_{num_test}.npy", np.array(embeddings_test))
            #
            # print(a)
            #
            # embeddings = np.load(f"embeddings/400/{margin_str}_{num_test}.npy", allow_pickle=True)

            ###############################################################################
            ### EMBEDDINGS END ###
            ##############################################################################

            embeddings = np.array(embeddings_test)
            np.save("tst_emd", embeddings)
            print(embeddings.shape)
            same_distances = []
            for class_id in range(1, num_test):
                class_embeddings = embeddings[class_id]

                enroll_embeddings = class_embeddings[:10]
                test_embeddings = class_embeddings[20:]
                template = generate_template(enroll_embeddings)

                num_iterations = 100
                subset_size = 100

                indices = np.random.randint(test_embeddings.shape[0], size=(num_iterations, subset_size))

                selected_subsets = test_embeddings[indices]
                s1 = selected_subsets[0]
                means_embeddings = np.mean(selected_subsets, axis=1)
                same_cosine_distances = compare_with_template(s1, template)
                same_distances.extend(same_cosine_distances)


            # print("Cosine Distances:", same_distances)
            # print(len(same_distances))
            # plt.plot(same_distances, color='green')
            # plt.show()
            #
            diff_distances = []

            for class_id in range(1, num_test):
                class_embeddings = embeddings[class_id]
                enroll_embeddings = class_embeddings[:10]
                template = generate_template(enroll_embeddings)

                for diff_class_id in range(class_id + 1, num_test):
                    diff_class_embeddings = embeddings[diff_class_id]



                    num_iterations = 100
                    subset_size = 100

                    indices = np.random.randint(diff_class_embeddings.shape[0], size=(num_iterations, subset_size))

                    selected_subsets = diff_class_embeddings[indices]
                    s1 = selected_subsets[0]

                    means_embeddings = np.mean(selected_subsets, axis=1)
                    diff_cosine_distances = compare_with_template(s1, template)
                    diff_distances.extend(list(diff_cosine_distances))


            # plt.plot(diff_distances, color='red')
            #
            # plt.show()

            dif = np.array(diff_distances).flatten()
            same = np.array(same_distances).flatten()

            print(f"NUM_test: {num_test}\n same_len = {len(same)}\n dif_len = {len(dif)}\n\n")
            os.makedirs(f"cosine_distances", exist_ok=True)
            os.makedirs(f"cosine_distances/distances_{num_train_subjects}", exist_ok=True)

            np.save(f"cosine_distances/distances_{num_train_subjects}/{margin_str}_{num_test}_dif.npy", dif)
            np.save(f"cosine_distances/distances_{num_train_subjects}/{margin_str}_{num_test}_same.npy", same)
            np.save(f"cosine_distances/distances_{num_train_subjects}/{margin_str}_{num_test}_dif_mean.npy", np.mean(dif))
            np.save(f"cosine_distances/distances_{num_train_subjects}/{margin_str}_{num_test}_same_mean.npy", np.mean(same))
            print(np.mean(dif))
            print(np.mean(same))

            ########################################################################
            ### DISTANCE CALCULATION END ###
            ########################################################################

            optimized_threshold = tune_threshold(embeddings , margin_str, num_train_subjects, num_test, ENROLL_SIZE, THRESHOLD_TUNING_SIZE)




            threshold = optimized_threshold
            FAR = len(dif[dif < threshold]) / len(dif)
            FAR = "{:.7f}".format(100 * FAR)
            FRR = len(same[same > threshold]) / len(same)
            FRR = "{:.7f}".format(100 * FRR)
            print(f"FAR {FAR}%")
            print(f"FRR {FRR}%")
            os.makedirs(f"git_logs", exist_ok=True)

            log_file_path = f"git_logs/log_{num_train_subjects}_{num_test}_{margin_str}"
            with open(log_file_path, 'a') as file:
                file.write(f"Threshold: {threshold}\n")
                file.write(f"Mean Distance Different Subjects: {np.mean(dif)}\n")
                file.write(f"Mean Distance Same Subjects: {np.mean(same)}\n")
                file.write(f"False Acceptance Rate (FAR): {FAR}%\n")
                file.write(f"False Rejection Rate (FRR): {FRR}%\n")
                file.write("########################################################\n")

            print("Logging completed.")

            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Program took {elapsed_time} seconds to execute.")
