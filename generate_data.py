import os

# 7 for training, 1 for validation, 1 for testing
# Frame: 1000~8000
id_train = ['00_00', '00_05', '00_09', '00_14', '00_15']
id_val = ['00_23']
id_test = ['00_27']

img_dir = 'hdImgs'
p = './data/panoptic'

def generate_train_data(path):
    train_txt = open('./data/train.txt', 'w')
    for s1 in os.listdir(path):
        f1 = os.path.join(s1,img_dir)
        for s2 in id_train:
            f2 = os.path.join(f1, s2)
            a = sorted(os.listdir(os.path.join(path, f2)))[1000:8000]
            for s3 in a:
                final_s = os.path.join(f2, s3)+'\n'
                train_txt.write(final_s)
    train_txt.close()

def generate_val_data(path):
    vla_txt = open('./data/val.txt', 'w')
    for s1 in os.listdir(path):
        f1 = os.path.join(s1,img_dir)
        for s2 in id_val:
            f2 = os.path.join(f1, s2)
            a = sorted(os.listdir(os.path.join(path, f2)))[1000:8000]
            for s3 in a:
                final_s = os.path.join(f2, s3)+'\n'
                vla_txt.write(final_s)
    vla_txt.close()

if __name__ == '__main__':
    print("Start generating....")
    generate_train_data(p)
    generate_val_data(p)
    print("Generated Done!")
