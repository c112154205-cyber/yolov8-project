from ultralytics import YOLO

# -------------------------------------------------------
# 重要：請確認你的 best.pt 在哪裡？
# 如果你跑了很多次，train 後面可能會有數字 (例如 train2, train3)
# 請去左邊資料夾 runs/detect/ 裡面看一下，選最新的那個資料夾
# -------------------------------------------------------
model_path = 'runs/detect/train/weights/best.pt' 

try:
    # 1. 載入模型
    model = YOLO(model_path)

    # 2. 設定預測來源
    # source = 0        -> 使用電腦的視訊鏡頭 (最推薦！很有趣)
    # source = 'test.jpg' -> 如果你有特定的圖片想測，就填圖片路徑
    source = 0

    print(f"正在載入模型：{model_path} ...")
    print("按 'q' 可以關閉視窗")

    # 3. 開始預測 (show=True 會跳出視窗, conf=0.5 代表信心大於 50% 才顯示)
    model.predict(source=source, show=True, conf=0.5)

except Exception as e:
    print("------------------------------------------------")
    print("❌ 發生錯誤！找不到模型檔案。")
    print(f"請檢查這個路徑對不對：{model_path}")
    print("------------------------------------------------")