from ultralytics import YOLO

if __name__ == '__main__':
    # 載入 YOLOv10n 模型
    model = YOLO('yolov10n.pt')

    # 開始訓練
    # data: 資料設定檔路徑
    # epochs: 訓練幾輪 (先設 30 試試看)
    # batch: 一次讀幾張圖 (4060 設 8 應該沒問題)
    # imgsz: 圖片大小
    model.train(data='dataset/data.yaml', epochs=30, batch=8, imgsz=640, device='0')