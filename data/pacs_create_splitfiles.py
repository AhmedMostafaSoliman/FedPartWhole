import os
from sklearn.model_selection import train_test_split

# Function to write data to a text file
def write_to_txt(file_path, image_paths, labels, indices):
    with open(file_path, 'w') as f:
        for img_path, label, index in zip(image_paths, labels, indices):
            f.write(f'{img_path} {label} {index}\n')

domain_abv = ['p', 'a', 'c', 's']
domain_name = ['photo', 'art_painting', 'cartoon', 'sketch']
out_path = '/home/ahmedm04/projects/distill_part_whole/FedPartWhole/datasets/PACS/split_files/'
# remove everything under out_path
os.system(f'rm -rf {out_path}')
# create it again
os.makedirs(out_path, exist_ok=True)

for domain_abv, domain_name in zip(domain_abv, domain_name):
    root = f'/home/ahmedm04/projects/distill_part_whole/FedPartWhole/datasets/PACS/raw_images/{domain_name}'
    #go over the images path where its folder with classes as folders and extract all image paths and their labels

    # Get the list of all the image paths
    image_paths = []
    labels = []
    indices = []
    for i, cls in enumerate(os.listdir(root)):
        for img_i, img in enumerate(sorted(os.listdir(os.path.join(root, cls)))):
            image_paths.append(os.path.join(domain_name, cls, img))
            labels.append(i+1)
            indices.append(img_i)

    # Write total data to a text file
    write_to_txt(os.path.join(out_path, f'{domain_name}_test_kfold.txt'), image_paths, labels, indices)

    # Split the data into train and test sets
    image_paths_train, image_paths_test, labels_train, labels_test, indices_train, indices_test = train_test_split(image_paths, labels, indices, test_size=0.1, random_state=42)

    # Write train data to a text file
    write_to_txt(os.path.join(out_path,f'{domain_name}_train_kfold.txt'), image_paths_train, labels_train, indices_train)

    # Write test data to a text file
    write_to_txt(os.path.join(out_path,f'{domain_name}_crossval_kfold.txt'), image_paths_test, labels_test, indices_test)