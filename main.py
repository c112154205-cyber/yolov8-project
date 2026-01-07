from ultralytics import YOLO
import os

def main():
    # ==========================================
    # 1. 設定路徑與環境
    # ==========================================
    # 取得目前這個 python 檔案所在的資料夾路徑
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # 假設你的資料集設定檔放在 dataset 資料夾內
    yaml_path = os.path.join(base_dir, 'dataset', 'data.yaml')
    
    print(f"目前工作路徑: {base_dir}")

    # ==========================================
    # 2. 載入模型
    # ==========================================
    print("正在載入 YOLOv8 模型...")
    # 第一次執行會自動下載 yolov8n.pt
    model = YOLO('yolov8n.pt') 

    # ==========================================
    # 3. 開始訓練
    # ==========================================
    print("開始訓練 YOLOv8...")
    train_name = 'yolov8_result'
    
    model.train(
        data=yaml_path,
        epochs=30,      
        imgsz=640,
        batch=16,
        project=base_dir,   # 專案根目錄
        name=train_name,    # 資料夾名稱
        exist_ok=True,      # 【關鍵】強制覆蓋，不會產生 result2, result3
        device=0,           # 如果報錯 CUDA error，請改成 'cpu'
        plots=True
    )
    
    # 組合出最佳權重檔的路徑
    best_weight_path = os.path.join(base_dir, train_name, 'weights', 'best.pt')
    print(f"訓練完成！最佳權重檔位於: {best_weight_path}")

    # 安全檢查：確認檔案真的存在
    if not os.path.exists(best_weight_path):
        print(f"❌ 錯誤：找不到檔案 {best_weight_path}")
        print("可能是訓練過程出錯，沒有產生權重檔。")
        return

    # ==========================================
    # 4. 驗證模型 (Validation)
    # ==========================================
    print("正在進行驗證...")
    model.val()

    # ==========================================
    # 5. 預測並自動存檔 (Prediction)
    # ==========================================
    print("開始進行預測測試...")
    
    try:
        # 重新載入剛剛練好的 best.pt
        best_model = YOLO(best_weight_path)

        # 執行預測
        # 使用 save=True，結果會存到預設的 runs/detect/predict 資料夾
        # 若要換成你自己的圖片，將 source 改成 r'C:\Users\...\圖片.jpg'
        best_model.predict(
            source='https://ultralytics.com/images/bus.jpg', 
            save=True  
        )
        
        print("-" * 30)
        print("🎉 全部完成！")
        print(f"1. 訓練好的模型在: {os.path.join(base_dir, train_name)}")
        print(f"2. 預測結果圖片在: {os.path.join(base_dir, 'runs', 'detect', 'predict')}")
        print("-" * 30)
        
    except Exception as e:
        print(f"預測階段發生錯誤: {e}")

if __name__ == '__main__':
    main()