import os
import requests
from remotezip import RemoteZip

# =======================================================
# 【关键修改】请在这里填入你刚才 curl 成功时的那个代理地址！
# 格式通常是 http://IP:端口
# 例如: "http://172.25.160.1:7890"
PROXY_URL = "http://172.20.96.1:7890"  
# =======================================================

url = "https://huggingface.co/datasets/paroj/ycbev_sd/resolve/main/test_pbr.zip"
target_objects = ["obj_000001", "obj_000002", "obj_000003", "obj_000004", "obj_000005"]
output_dir = "./ycb_ev_data/test"

print(f"正在配置强力代理: {PROXY_URL}")

# 1. 创建一个 Session 对象，并强制设置代理
session = requests.Session()
session.proxies = {
    'http': PROXY_URL,
    'https': PROXY_URL,
}

# 2. 伪装一下 User-Agent (有时候服务器会拦截 Python 脚本)
session.headers.update({
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
})

print(f"正在通过代理连接 Hugging Face... \n目标URL: {url}")

try:
    # 3. 将 session 传递给 RemoteZip
    with RemoteZip(url, session=session) as zip:
        print("连接成功！已获取文件列表。")
        
        all_files = zip.namelist()
        files_to_extract = []
        for fname in all_files:
            for obj_id in target_objects:
                if obj_id in fname:
                    files_to_extract.append(fname)
                    break
        
        if not files_to_extract:
            print("未找到匹配文件，请检查压缩包结构。")
        else:
            print(f"找到 {len(files_to_extract)} 个文件。")
            print("开始下载... (这可能需要几分钟，请耐心等待)")
            
            # 这里的下载也会自动使用上面的 session 和代理
            zip.extractall(path=output_dir, members=files_to_extract)
            
            print("下载完成！")

except Exception as e:
    print("\n还是报错了？请检查下面几点：")
    print(f"错误信息: {e}")
    print("1. 你的 PROXY_URL 填对了吗？必须要和 curl 能通的那个一模一样。")
    print("2. 你的 Windows 代理软件是否依然开着 'Allow LAN'？")