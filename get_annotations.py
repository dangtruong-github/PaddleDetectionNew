import json

annotation_path = "./output/infer_annotations.json"
file_path = "./output/infer_paths.json"
out_path = "./output/annotations.txt"

with open(annotation_path, "r") as f:
    bboxes = json.load(f)

with open(file_path, "r") as f:
    img_paths = json.load(f)

with open(out_path, "w") as f:
    for i in range(len(bboxes)):
        have_img = False
        for j in range(len(img_paths)):
            if bboxes[i]["image_id"] == img_paths[j]["image_id"]:
                f.write(img_paths[j]["image_path"])
                img_height = img_paths[j]["image_height"]
                img_width = img_paths[j]["image_width"]
                have_img = True
                break
        
        if not have_img:
            raise Exception("INVALID")

        f.write(" ")
        f.write(str(bboxes[i]["category_id"] % 4))
        f.write(" ")
        bbox_now = []
        for idx, box in enumerate(bboxes[i]["bbox"]):
            if idx % 2 == 1:
                bbox_now.append(box / img_height)
            else:
                bbox_now.append(box / img_width)
        
        bbox_now[0] += (bbox_now[2] / 2)
        bbox_now[1] += (bbox_now[3] / 2)

        for box in bbox_now:
            f.write("{}".format(box))
            f.write(" ")

        f.write("{}".format(bboxes[i]["score"]))
        f.write("\n")