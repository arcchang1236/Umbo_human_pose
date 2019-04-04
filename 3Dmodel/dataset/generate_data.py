import os

# 8 for training, 1 for validation, 1 for testing
# Frame: 1000~6000

id_all = ['00_00', '00_05', '00_09', '00_14', '00_15', '00_23', '00_27']

def generate_data(path, start, end):
    txt = open(path, 'w')
    img_dir = 'hdImgs'
    p = '../data/panoptic'
    for s1 in os.listdir(p):
        f1 = os.path.join(s1,img_dir)
        for s2 in id_all:
            f2 = os.path.join(f1, s2)
            a = sorted(os.listdir(os.path.join(p, f2)))[start:end]
            for s3 in a:
                final_s = os.path.join(f2, s3)+'\n'
                txt.write(final_s)
    txt.close()


if __name__ == '__main__':
    print("Start generating....")
    
    generate_data('../data/train.txt', 1000, 5000)
    print("Generate Training Data Done!")

    generate_data('../data/val.txt', 5000, 5500)
    print("Generate Testing Data Done!")

    generate_data('../data/test.txt', 5500, 6000)
    print("Generate Testing Data Done!")
