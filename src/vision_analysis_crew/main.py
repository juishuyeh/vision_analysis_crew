#!/usr/bin/env python
import sys
import warnings

from datetime import datetime

from vision_analysis_crew.crew import VisionAnalysisCrew

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

# This main file is intended to be a way for you to run your
# crew locally, so refrain from adding unnecessary logic into this file.
# Replace with inputs you want to test with, it will automatically
# interpolate any tasks and agents information
import os
from vision_analysis_crew.crew import VisionAnalysisCrew

def run():
    """
    執行視覺分析 Crew
    """
    # 建立輸出目錄
    os.makedirs('output', exist_ok=True)
    
    # 設定分析參數
    inputs = {
        'image_path': 'https://cdn.orangenews.hk/images/2021/08/27/b7c5698f289740c184251124a04f5f99.jpg',  # 修改為您的圖片路徑
        'analysis_requirements': '''
        請特別關注以下方面：
        - 圖片中的文字內容（如果有的話）
        - 主要物體和元素的位置關係
        - 整體色彩搭配和視覺效果
        - 任何特殊的符號、標記或圖表
        '''
    }
    
    print("開始圖片分析...")
    print(f"分析圖片：{inputs['image_path']}")
    print("-" * 50)
    
    try:
        # 建立並執行 Crew
        result = VisionAnalysisCrew().crew().kickoff(inputs=inputs)
        
        print("\n" + "=" * 50)
        print("分析完成！")
        print("=" * 50)
        print(result.raw)
        
        print(f"\n詳細報告已儲存至：output/vision_analysis_report.md")
        
    except Exception as e:
        print(f"執行過程中發生錯誤：{str(e)}")
        print("請檢查：")
        print("1. LM Studio 是否已啟動並載入了 Qwen2.5-VL-7B 模型")
        print("2. 圖片路徑是否正確")
        print("3. 網路連接是否正常")
