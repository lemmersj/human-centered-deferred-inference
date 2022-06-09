import csv
import os
from IPython import embed
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
import sys

scenario_category = sys.argv[1]

class ScenarioGenerator:
    def __init__(self, scenario_id, bboxes):
        self.scenario_id = scenario_id
        self.bbox = bboxes
        self.cur_pick = None
        self.cur_idx = 0

        self.to_write_dict_list = []
    
    def find_bbox(self, x, y):
        cur_bbox = None
        for i in range(self.bbox.shape[0]):
            if x > self.bbox[i, 0] and x < self.bbox[i, 2] and y > self.bbox[i, 1] and y < self.bbox[i, 3]:
                if cur_bbox is None:
                    cur_bbox = self.bbox[i, :]
                else:
                    # If there's overlap, use the smaller bbox.
                    area_prev = (cur_bbox[2]-cur_bbox[0])*(cur_bbox[3]-cur_bbox[1])
                    area_cur = (self.bbox[i,2]-self.bbox[i,0])*(self.bbox[i,3]-self.bbox[i,1])
                    if area_cur < area_prev:
                        cur_bbox = self.bbox[i, :]
        if cur_bbox is None:
            embed()
        return cur_bbox
    def onclick(self, event):
        """Handles a click event."""
        x = event.xdata
        y = event.ydata
        assoc_bbox = self.find_bbox(x, y)

        if assoc_bbox is None:
            return

        if self.cur_pick is None:
            print(f"Setting pick to {assoc_bbox}")
            self.cur_pick = assoc_bbox
        else:
            print(f"Setting place to {assoc_bbox}")
            cur_place = assoc_bbox

            self.to_write_dict_list.append({'id':self.cur_idx,
                                            'pick_tlx': self.cur_pick[0],
                                            'pick_tly': self.cur_pick[1],
                                            'pick_brx': self.cur_pick[2],
                                            'pick_bry': self.cur_pick[3],
                                            'place_tlx': cur_place[0],
                                            'place_tly': cur_place[1],
                                            'place_brx': cur_place[2],
                                            'place_bry': cur_place[3]})
            self.cur_pick = None
            self.cur_idx += 1
    
    def write_file(self):
        with open(f"scenarios/{scenario_category}/{self.scenario_id}.csv", "w") as outfile:
            writer = csv.DictWriter(outfile, fieldnames=self.to_write_dict_list[0].keys())
            writer.writeheader()
            for row in self.to_write_dict_list:
                writer.writerow(row)


existing_files = os.listdir(f"scenarios/{scenario_category}")
read_dict = []
with open(f"scenarios/{scenario_category}/scenarios.csv", "r") as scenarios_file:
    reader = csv.DictReader(scenarios_file)
    for line in reader:
        read_dict.append(line)

# create a scenario for every image in scenarios.
for scenario in read_dict:
    print(scenario)
    if f"{scenario['id']}.csv" in existing_files:
        continue
    image_in = Image.open(f"../bottom-up-attention.pytorch/images/{scenario_category}/{scenario['image']}.jpg")
    bboxes = np.load(f"../bottom-up-attention.pytorch/extracted_features/{scenario_category}/{scenario['image']}.npz")['bbox']

    scenario_generator = ScenarioGenerator(scenario['id'], bboxes)
    draw = ImageDraw.Draw(image_in)
    for bbox in range(bboxes.shape[0]):
        draw.rectangle(bboxes[bbox, :])

    fig, ax = plt.subplots()

    cid = fig.canvas.mpl_connect('button_press_event', scenario_generator.onclick)
    ax.imshow(image_in)
    plt.show()
    scenario_generator.write_file()
