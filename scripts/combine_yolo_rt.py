import csv
import sys


def averaged_bbox(bbox1, bbox2):
    y_min = (bbox1[0] + bbox2[0]) / 2
    x_min = (bbox1[1] + bbox2[1]) / 2
    y_max = (bbox1[2] + bbox2[2]) / 2
    x_max = (bbox1[3] + bbox2[3]) / 2
    return [y_min, x_min, y_max, x_max]


if len(sys.argv) != 4:
    print("You must pass 3 files as arguments.")
    print("compare_results.py results1 results2 output")
    exit(1)
res1_path = sys.argv[1]
res2_path = sys.argv[2]

yolo_model = sys.argv[1]
rt_model = sys.argv[2]
output = sys.argv[3]


yolo_file = open(yolo_model, "r")
rt_file = open(rt_model, "r")
output_file = open(output, "w")

yolo = csv.DictReader(yolo_file)
rt = csv.DictReader(rt_file)
output = csv.DictWriter(output_file, ["filename", "class", "bbox"])
output.writeheader()

for ym, rtm in zip(yolo, rt):
    if ym["class"] == "" and rtm["class"] == "":
        output.writerow(ym)
    elif ym["class"] == "" and rtm["class"] != "":
        output.writerow(rtm)
    elif ym["class"] != "" and rtm["class"] == "":
        output.writerow(ym)
    elif ym["class"] != rtm["class"]:
        # rtm_bbox = eval(rtm["bbox"])
        # ym_bbox = eval(ym["bbox"])
        # bbox = averaged_bbox(rtm_bbox, ym_bbox)
        # bbox = list(map(lambda x: 0 if x < 0 else x, bbox))
        row = ym
        output.writerow(row)
    else:
        output.writerow(ym)


yolo_file.close()
rt_file.close()
output_file.close()
