import csv
import sys


def main():
    print(sys.argv)
    if len(sys.argv) != 3:
        print("You must pass 2 files as arguments.")
        print("compare_results.py results1 results2")
        exit(1)
    res1_path = sys.argv[1]
    res2_path = sys.argv[2]

    res1_file = open(res1_path, "r")
    res2_file = open(res2_path, "r")

    res1 = csv.DictReader(res1_file)
    res2 = csv.DictReader(res2_file)

    total_cls_differences = 0
    total_bbox_differences = 0

    for r1, r2 in zip(res1, res2):
        if r1["class"] != r2["class"]:
            total_cls_differences += 1
        if r1["bbox"] != r2["class"]:
            total_bbox_differences += 1

    print("cls differences:", total_cls_differences)
    print("bbox differences:", total_bbox_differences)

    res1_file.close()
    res2_file.close()


if __name__ == "__main__":
    main()
