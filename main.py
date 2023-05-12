from pathlib import Path
import sys
import os

from ocr.ssdocr import OCR
from utils.general import LOGGER, is_picture, increment_path
from utils.handle import get_rec_label

BASE: Path

if __name__ == '__main__':
    if getattr(sys, 'frozen', False):
        BASE = Path(os.path.dirname(sys.executable)).resolve()
    elif __file__:
        BASE = Path(os.path.dirname(__file__)).resolve()
    LOGGER.info("Base Name: {}".format(BASE))
    ocr = OCR(det_model='/home/xla/code/python/SSD-OCR/models/det.onnx', 
              rec_model='/home/xla/code/python/SSD-OCR/models/rec.onnx')
    DATA_TEST = Path("./data/test")
    project=f"{BASE}/runs/detect"
    name="det"
    labels = get_rec_label()
    from collections import defaultdict
    for p, d, f in os.walk(DATA_TEST):
        cnt = error = 1
        save_dir = increment_path(Path(project) / name, exist_ok=False)  # increment run
        (save_dir).mkdir(parents=True, exist_ok=True)  # make dir
        LOGGER.info(f"save_dir: {save_dir}")
        error_dict = defaultdict(list)
        import time
        start_time = time.time()
        for file in f:
            pic = DATA_TEST / file
            if is_picture(pic):
                a, b = ocr.score(str(pic), list(map(int, labels[file])), save_dir)
                cnt += a
                error += b
                # predict = ocr.predict(str(pic), "./results")
                # expect = list(map(int, labels[file]))
                # for i, (x, y) in enumerate(zip(predict, expect)):
                #     if x != y:
                #         error_dict[file].append([i, (x, y)])
                #         error += 1
                #     cnt += 1
                # break
        end_time = time.time()
        with open(save_dir/'results.txt', 'w', encoding='utf-8') as tofile:
            import json
            tofile.write(json.dumps(error_dict, ensure_ascii=False))
        LOGGER.info(" [OCR] total: %s, error: %d, percent: %.2f%%", cnt, error, (1 - (error/ cnt)) * 100)
        cost = end_time - start_time
        LOGGER.info("[TIME] total: %f, fps: %d pic/s", cost, cnt / cost)
