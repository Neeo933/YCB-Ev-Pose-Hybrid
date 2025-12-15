import os
import requests
from remotezip import RemoteZip

# ================= 配置 =================
PROXY_URL = "http://172.20.96.1:7890" 
url = "https://huggingface.co/datasets/paroj/ycbev_sd/resolve/main/test_pbr.zip"
target_ids = ["000000", "000001", "000002", "000003", "000004"]
output_dir = "./ycb_ev_data/test"
# =======================================

session = requests.Session()
session.proxies = {'http': PROXY_URL, 'https': PROXY_URL}
session.headers.update({'User-Agent': 'Mozilla/5.0'})

print(f"开始下载 PNG 事件图和标签...")

try:
    with RemoteZip(url, session=session) as zip:
        all_files = zip.namelist()
        files_to_extract = []
        
        for fname in all_files:
            # 筛选物体
            is_target = False
            for tid in target_ids:
                if f"/{tid}/" in fname or fname.startswith(f"{tid}/"):
                    is_target = True
                    break
            if not is_target: continue

            # 【新逻辑】只下载 ev_histogram (PNG图片) 和 标签
            if "ev_histogram/" in fname or "scene_gt" in fname or "obj_info" in fname:
                files_to_extract.append(fname)

        print(f"找到 {len(files_to_extract)} 个文件。开始下载...")
        zip.extractall(path=output_dir, members=files_to_extract)
        print("下载完成！")

except Exception as e:
    print("出错:", e)