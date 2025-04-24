import re
import os
from collections import defaultdict
import json
import csv
import numpy as np
from tqdm import tqdm
import argparse


def main(args):
    record = [json.loads(q) for q in open(args.result_file, "r")]

    mIoU = np.sum(
        [meters["iou"] for meters in record], axis=0
    ) / len(record)

    Acc25 = np.sum(
        [meters["tp25"] for meters in record], axis=0
    ) / len(record)

    Acc50 = np.sum(
        [meters["tp50"] for meters in record], axis=0
    ) / len(record)


    print("Val result: mIoU/Acc25/Acc50 {:.4f}/{:.4f}/{:.4f}".format(
                    mIoU, Acc25, Acc50
                ))
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotation-file', type=str)
    parser.add_argument('--result-file', type=str)
    parser.add_argument('--result-dir', type=str)
    args = parser.parse_args()

    main(args)
