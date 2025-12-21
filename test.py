import json
# 换成你实际的 json 路径
with open("./ycb_ev_data/dataset/test_pbr/000000/scene_gt.json", "r") as f:
    data = json.load(f)
    print("Sample T:", data["0"][0]["cam_t_m2c"])