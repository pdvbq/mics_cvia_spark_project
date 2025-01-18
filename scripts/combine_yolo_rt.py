import csv

yolo_model = "./data/yolo11x_submission.csv"
rt_model = "./data/rtdetr_submission.csv"
output = "./data/output.csv"


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
    if ym["class"] != "":
        output.writerow(ym)
    if ym["class"] == "" and rtm["class"] != "":
        output.writerow(rtm)


yolo_file.close()
rt_file.close()
output_file.close()
