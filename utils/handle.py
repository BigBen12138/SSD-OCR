from functools import cmp_to_key
import pathlib


def cmp(a, b):
    if a[0] == b[0]:
        return a[1] - b[1]
    return a[0] - b[0]

def get_rec_label() -> dict:
    import json
    import cv2
    path = pathlib.Path('./data/test')
    tmp = pathlib.Path('./img')
    res = dict()
    with open(path/'Label.txt', 'r') as f:
        while (line:=f.readline()):
            line = line.split('\t')
            name, data = line[0], json.loads(line[1])
            values = []
            image = cv2.imread(str(path/name))
            for i, info in enumerate(data):
                trans, points = info['transcription'], info['points']
                points = sorted(points, key=lambda x: (x[0], x[1]))
                values.append(trans)
                x1, y1 = list(map(min, *points))
                x2, y2 = list(map(max, *points))
                # print(x1, y1, x2, y2)
                # print(tmp/f"{name}-{i}.png")
                # im = image[y1:y2, x1:x2, :]
                # cv2.imwrite(str(tmp/f"{name}-{i}.png"), im)
                # print(sorted(points, key=cmp_to_key(cmp)))
            res[name] = values
            # print(len(values), name, values)
    return res

ch2en = {
    "鱼跃": "YuYue",
    "欧姆龙": "OuMuLong",
    # "omron": "OuMuLong"，
    "可浮": "KeFu"
}

if __name__ == '__main__':
    print(get_rec_label())
    ...