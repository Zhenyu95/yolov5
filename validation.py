import os


TRUE_IMG_PATH = '/Users/zhenyu/Desktop/validation/images/'
TRUE_LABEL_PATH = '/Users/zhenyu/Desktop/validation/true_label/'
PRED_LABEL_PATH = '/Users/zhenyu/Desktop/validation/pred_label/'


def get_lists(raw_img_path=TRUE_IMG_PATH, raw_label_path=TRUE_LABEL_PATH, pred_label_path=PRED_LABEL_PATH):
    img_list = [f for f in os.listdir(raw_img_path) 
                if (os.path.isfile(os.path.join(raw_img_path, f)) and f.endswith('.jpg'))]
    return img_list


def map_label(img_list, label_path):
    label_map = {key:[0] for key in img_list}
    for key in label_map:
        try:
            with open(label_path+key[:-4]+'.txt') as f:
                label_map[key] = [list(map(float, line.rstrip().split())) for line in f]
        except FileNotFoundError:
            pass
    return label_map


def main():
    img_list = get_lists()
    true_label_map = map_label(img_list, label_path=TRUE_LABEL_PATH)
    pred_label_map = map_label(img_list, label_path=PRED_LABEL_PATH)
    total = 0
    false_negative = 0
    false_positive = 0
    for key in true_label_map:
        len_true = len(true_label_map[key])
        len_pred = len(pred_label_map[key])
        if len_true == len_pred:
            total += 1
        elif len_true > len_pred:
            total += 1
            false_negative += 1
        elif len_pred > len_true:
            total += 1
            false_positive += 1
    print(total, false_negative, false_positive)
    
    
if __name__ == '__main__':
    main()
            