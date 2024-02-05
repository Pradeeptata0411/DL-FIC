import Preprocessing as skill
import pandas as pd

if __name__ == "__main__":
    #images_folder_path = 'E:\\KLU\\3rd year\\3_2\\deep learning\\Deep Learning Programs\\trainfolder'
    #images_folder_path = 'E:\\KLU\\3rd year\\3_2\\deep learning\\Deep Learning Programs\\train'
    images_folder_path = 'E:\\KLU\\3rd year\\3_2\\deep learning\\Deep Learning Programs\\imagefile'

    imdata = skill.PreProcess_Data()
    imdata.visualization_images(images_folder_path, 5)
    imagefile, label, df = imdata.preprocess(images_folder_path)
    csv_file_path = 'output.csv'
    df.to_csv(csv_file_path, index=False)
    print(f"CSV file saved at: {csv_file_path}")
    train_generator, test_generator, validation_generator = imdata.generate_train_test_images(imagefile, label)
    csv_file_path2 = 'output_test.csv'
    pd.DataFrame(test_generator.labels, columns=["Labels"]).to_csv(csv_file_path2, index=False)
    print(f"CSV file saved at: {csv_file_path2}")