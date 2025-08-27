import base64
import io
import os

import requests
from crewai.tools import BaseTool
from PIL import Image
from pydantic import BaseModel, Field


class VisionAnalysisInput(BaseModel):
    """輸入參數定義"""

    image_path: str = Field(..., description="圖片路徑（本地路徑或 URL）")
    query: str = Field(
        default="請詳細分析這張圖片的內容，包含物體、文字、場景等所有可見元素",
        description="對圖片的分析要求",
    )


class VisionAnalysisTool(BaseTool):
    name: str = "圖片視覺分析工具"
    description: str = (
        "分析圖片內容的專業工具。能夠識別物體、讀取文字、"
        "理解場景、分析圖表、提取結構化資訊等。"
        "支援本地圖片檔案和網路 URL。"
    )
    args_schema: type[BaseModel] = VisionAnalysisInput

    def _run(self, image_path: str, query: str) -> str:
        try:
            # 處理圖片
            if image_path.startswith("http"):
                response = requests.get(image_path, timeout=30)
                image = Image.open(io.BytesIO(response.content))
            else:
                if not os.path.exists(image_path):
                    return f"錯誤：找不到圖片檔案 {image_path}"
                image = Image.open(image_path)

            # 轉換為 RGB 格式（如果需要）
            if image.mode != "RGB":
                image = image.convert("RGB")

            # 將圖片轉換為 base64
            buffered = io.BytesIO()
            image.save(buffered, format="JPEG", quality=95)
            img_base64 = base64.b64encode(buffered.getvalue()).decode()

            # 構建 API 請求
            api_url = os.getenv("LM_STUDIO_BASE_URL", "http://localhost:1234/v1")

            payload = {
                "model": "qwen/qwen2.5-vl-7b",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"},
                            },
                            {"type": "text", "text": query},
                        ],
                    }
                ],
                "max_tokens": 4096,
                "temperature": 0.3,
                "stream": False,
            }

            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {os.getenv('LM_STUDIO_API_KEY', 'dummy')}",
            }

            # 發送請求到 LM Studio
            response = requests.post(
                f"{api_url}/chat/completions", headers=headers, json=payload, timeout=120
            )

            if response.status_code == 200:
                result = response.json()
                content = result["choices"][0]["message"]["content"]
                return content
            else:
                return f"API 請求失敗: {response.status_code} - {response.text}"

        except Exception as e:
            return f"圖片分析過程中發生錯誤: {str(e)}"
