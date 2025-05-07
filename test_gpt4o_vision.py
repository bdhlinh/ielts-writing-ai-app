import openai
import base64
import os
from dotenv import load_dotenv

# Load API key từ .env
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Đọc và mã hóa ảnh
image_path = "test_chart.png"  # ← thay bằng tên ảnh của bạn
with open(image_path, "rb") as f:
    base64_image = base64.b64encode(f.read()).decode("utf-8")

# Gửi ảnh đến GPT-4o
response = openai.chat.completions.create(
    model="gpt-4o",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What is shown in this image?"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}},
            ],
        }
    ],
    max_tokens=300,
)

# In ra kết quả
print("🧠 GPT-4o Vision Response:")
print(response.choices[0].message.content)
