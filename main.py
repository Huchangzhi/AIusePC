import base64
import json
import signal
from io import BytesIO
import pyautogui
import sys
from PIL import ImageGrab, Image
from openai import OpenAI
import logging
import numpy as np
import requests
import time
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

# 屏幕参数配置
SCREEN_WIDTH, SCREEN_HEIGHT = pyautogui.size()

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# 定义JSON Schema格式约束
RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {
            "type": "string",
            "enum": ["question", "mouse_move", "clipboard", "task_complete", "mouse_click", "mouse_right_click", "mouse_double_click", "keyboard_input"]  # 添加 keyboard_input
        },
        "content": {
            "type": ["string", "array"],
            "items": {"type": "integer"}
        },
        "reasoning": {
            "type": "string",
            "description": "解释这一步操作的目的和思路"
        }
    },
    "required": ["action"]
}

def signal_handler(sig, frame):
    """处理Ctrl+C退出"""
    logging.info("用户终止程序")
    sys.exit(0)

def build_prompt(task, previous_reasoning="", asked_confirmation=False):
    """构建带格式约束的提示词，包含上下文"""
    prompt_text = f"""你是一个自动化操作助手，请根据用户任务、屏幕截图以及之前的操作备注生成 JSON 指令。

    当前任务：{task} (使用鼠标点击和键盘输入来完成任务)

    {"之前的操作备注：" + previous_reasoning if previous_reasoning else "这是第一次操作"}

    当任务比较复杂时，一次只进行一次操作。当你发现任务已经完成，请返回任务完成指令。

    请严格按照以下 JSON Schema 格式返回结果，不要包含任何其他文本：
    {json.dumps(RESPONSE_SCHEMA, indent=2)}

    **JSON Schema说明：**
    - action (string):  指定执行的动作，必须是以下之一: question, mouse_move, clipboard, task_complete, mouse_click, mouse_right_click, mouse_double_click, keyboard_input。
    - content (string | array):  根据 action 的不同，content 的类型和格式也不同。
    - reasoning (string): 解释这一步操作的目的和思路。在下一次调用时，该备注将作为上下文传递给你。

    **操作规范：**
    - question:  content 是你想询问用户的文本问题。只允许询问一次！ 例如: {{"action": "question", "content": "请确认是否要继续操作？", "reasoning": "询问用户是否要继续操作"}}
    - mouse_move: content 是一个包含两个整数的数组，表示鼠标移动的目标坐标 (x, y)。例如: {{"action": "mouse_move", "content": [100, 200], "reasoning": "将鼠标移动到指定位置，准备点击"}}
    - mouse_click: content 是一个包含两个整数的数组，表示鼠标点击的坐标 (x, y)。例如: {{"action": "mouse_click", "content": [100, 200], "reasoning": "点击指定位置的按钮"}}
    - mouse_right_click: content 是一个包含两个整数的数组，表示鼠标右键点击的坐标 (x, y)。例如: {{"action": "mouse_right_click", "content": [100, 200], "reasoning": "右键点击指定位置"}}
    - mouse_double_click: content 是一个包含两个整数的数组，表示鼠标双击的坐标 (x, y)。例如: {{"action": "mouse_double_click", "content": [100, 200], "reasoning": "双击指定位置"}}
    - clipboard: content 是要复制到剪贴板的文本内容，复制完后请手动粘贴到剪贴板。然后输入 1 继续 。例如: {{"action": "clipboard", "content": "Hello, world!", "reasoning": "已经将文本复制到剪贴板，请手动粘贴到文本框"}}
    - keyboard_input: content 是要输入的文本字符串。例如: {{"action": "keyboard_input", "content": "example@email.com", "reasoning": "输入邮箱地址"}}
    - task_complete:  content 只能是 "success" (任务成功完成) 或 "error" (任务遇到无法恢复的错误)。例如: {{"action": "task_complete", "content": "success", "reasoning": "任务成功完成"}}

    **重要：**
    - 注意移动与点击鼠标操作时若要点击的按钮较大，请点击中间，以确保精确点击
    - 你的输出必须是完全符合 JSON Schema 格式的 JSON 字符串，不要包含任何多余的文字、注释或解释，不要使用markdown代码块，一定不要，请直接输出json!!!。
    - 坐标 (x, y) 必须是整数。
    - 务必在 reasoning 字段中清晰地说明你的操作目的和思路以及下一步应该做什么，已经已经完成了什么。
    - 如果已经询问过用户是否要关闭计算机，并且用户已确认，则不要再次询问，直接按照上述步骤执行点击操作！
    {"已经询问过用户是否要关闭计算机，用户已确认，请按照指定步骤执行点击操作" if asked_confirmation else ""}
    """
    return {"type": "text", "text": prompt_text}

def validate_response(response):
    """验证响应格式"""
    try:
        parsed = json.loads(response)
        # 验证action字段
        if parsed['action'] not in RESPONSE_SCHEMA['properties']['action']['enum']:
            return {"valid": False, "error": "无效的操作类型"}

        # 验证content格式
        if parsed['action'] == 'mouse_move' or parsed['action'] == 'mouse_click' or parsed['action'] == 'mouse_right_click' or parsed['action'] == 'mouse_double_click':
            if not isinstance(parsed['content'], list) or len(parsed['content']) != 2:
                return {"valid": False, "error": "坐标格式错误"}
            try:
                [int(x) for x in parsed['content']]
            except ValueError:
                return {"valid": False, "error": "坐标必须为整数"}

        # 验证 keyboard_input content 格式
        if parsed['action'] == 'keyboard_input':
            if not isinstance(parsed['content'], str):
                return {"valid": False, "error": "keyboard_input 的 content 必须是字符串"}

        # 验证 reasoning 字段
        if 'reasoning' not in parsed:
            return {"valid": False, "error": "缺少 reasoning 字段"}
        if not isinstance(parsed['reasoning'], str):
            return {"valid": False, "error": "reasoning 字段必须是字符串"}

        return {"valid": True, "data": parsed}
    except json.JSONDecodeError:
        return {"valid": False, "error": "JSON解析失败"}
    except KeyError:
        return {"valid": False, "error": "缺少必要字段"}

def take_screenshot():
    """智能截图并返回元数据"""
    try:
        # 原始截图
        original_image = ImageGrab.grab()
        original_size = original_image.size

        # 不进行压缩
        buffered = BytesIO()
        original_image.save(buffered, format="PNG")  # 保存为 PNG 格式，保留更多信息

        return {
            "image_data": "data:image/png;base64," + base64.b64encode(buffered.getvalue()).decode(),  # 修改为 PNG 格式
            "scale_x": 1,  # 不进行缩放
            "scale_y": 1,  # 不进行缩放
            "original_size": original_size,
            "compressed_size": original_size  # 压缩尺寸与原始尺寸相同
        }
    except Exception as e:
        logging.error(f"截图处理失败：{e}")
        return None

def send_to_ai(messages):
    """增强型API请求"""
    client = OpenAI(
        base_url='https://api-inference.modelscope.cn/v1',
        api_key='1234567',
        timeout=30
    )

    max_retries = 3
    for retry in range(max_retries):
        try:
            response = client.chat.completions.create(
                model='Qwen/Qwen2.5-VL-72B-Instruct',
                messages=messages,
                stream=False
            )

            # 打印 API 响应
            raw_content = response.choices[0].message.content
            print(f"原始API响应: {raw_content}")

            # 移除 ```json 标签
            if raw_content.startswith("```json"):
                raw_content = raw_content[6:]
            if raw_content.endswith("```"):
                raw_content = raw_content[:-3]

            # 自动坐标转换
            parsed = json.loads(raw_content)

            return parsed
        except json.JSONDecodeError as e:
            logging.error(f"JSON 解析失败：{e}")
            return {"action": "error", "content": str(e), "reasoning": "JSON解析失败"}
        except Exception as e:
            logging.error(f"API请求异常：{str(e)}")
            if retry < max_retries - 1:
                logging.info(f"Retrying in 1 second...")
                time.sleep(1)
            else:
                return {"action": "error", "content": "Max retries exceeded", "reasoning": "达到最大重试次数"}
    return {"action": "error", "content": "Max retries exceeded", "reasoning": "达到最大重试次数"}

def process_response(response, messages, image_meta, asked_confirmation):
    """处理响应并返回状态"""
    action = response.get("action")
    content = response.get("content", "")
    reasoning = response.get("reasoning", "")  # 获取 reasoning

    if action == "question":
        if asked_confirmation:
            print("⚠️  AI 再次提问，存在逻辑错误！")
            return "error", "AI 再次提问，存在逻辑错误！"
        user_input = input(f"❓ 问题：{content}\n您的回答 (输入exit退出): ")
        if user_input.lower() in ['exit', 'quit']:
            return "exit", None
        messages.append({
            'role': 'user',
            'content': [{'type': 'text', 'text': user_input}]
        })
        return "continue", reasoning

    elif action == "mouse_move":
        try:
            x, y = content
            # 坐标转换：乘以 2
            x = round((x)*2 )
            y = round((y)*2)
            # 边界检查
            x = max(0, min(x, SCREEN_WIDTH-1))
            y = max(0, min(y, SCREEN_HEIGHT-1))

            pyautogui.click(x, y)
            print(f"🖱️ 单击鼠标：({x}, {y})")
            # debug_image(image_meta['image_data'], content, (x, y))  # 添加 debug_image 调用
        except Exception as e:
            logging.error(f"鼠标单击失败：{e}")
            print(f"⚠️ 鼠标单击失败: {e}")
            return "error", "鼠标单击失败"
        return "continue", reasoning
    # elif action == "mouse_move":
    #     try:
    #         x, y = content
    #         # 坐标转换：乘以 2
    #         x = round((x)*2 )
    #         y = round((y)*2)
    #         # 边界检查
    #         x = max(0, min(x, SCREEN_WIDTH-1))
    #         y = max(0, min(y, SCREEN_HEIGHT-1))

    #         pyautogui.moveTo(x, y)
    #         print(f"🖱️ 校准后坐标：({x}, {y})")
    #         # debug_image(image_meta['image_data'], content, (x, y))  # 添加 debug_image 调用
    #     except Exception as e:
    #         logging.error(f"鼠标移动失败：{e}")
    #         print(f"⚠️ 鼠标移动失败: {e}")
    #         return "error", "鼠标移动失败"
    #     return "continue", reasoning

    elif action == "mouse_click":
        try:
            x, y = content
            # 坐标转换：乘以 2
            x = round((x)*2 )
            y = round((y)*2)
            # 边界检查
            x = max(0, min(x, SCREEN_WIDTH-1))
            y = max(0, min(y, SCREEN_HEIGHT-1))

            pyautogui.click(x, y)
            print(f"🖱️ 单击鼠标：({x}, {y})")
            # debug_image(image_meta['image_data'], content, (x, y))  # 添加 debug_image 调用
        except Exception as e:
            logging.error(f"鼠标单击失败：{e}")
            print(f"⚠️ 鼠标单击失败: {e}")
            return "error", "鼠标单击失败"
        return "continue", reasoning

    elif action == "mouse_right_click":
        try:
            x, y = content
            # 坐标转换：乘以 2
            x = round((x)*2 )
            y = round((y)*2)
            # 边界检查
            x = max(0, min(x, SCREEN_WIDTH-1))
            y = max(0, min(y, SCREEN_HEIGHT-1))

            pyautogui.rightClick(x, y) # 使用 pyautogui.rightClick()
            print(f"🖱️ 右键单击鼠标：({x}, {y})")
            # debug_image(image_meta['image_data'], content, (x, y))  # 添加 debug_image 调用
        except Exception as e:
            logging.error(f"鼠标右键单击失败：{e}")
            print(f"⚠️ 鼠标右键单击失败: {e}")
            return "error", "鼠标右键单击失败"
        return "continue", reasoning

    elif action == "mouse_double_click":
        try:
            x, y = content
            # 坐标转换：乘以 2
            x = round((x)*2 )
            y = round((y)*2)
            # 边界检查
            x = max(0, min(x, SCREEN_WIDTH-1))
            y = max(0, min(y, SCREEN_HEIGHT-1))

            pyautogui.doubleClick(x, y) # 使用 pyautogui.doubleClick()
            print(f"🖱️ 双击鼠标：({x}, {y})")
            # debug_image(image_meta['image_data'], content, (x, y))  # 添加 debug_image 调用
        except Exception as e:
            logging.error(f"鼠标双击失败：{e}")
            print(f"⚠️ 鼠标双击失败: {e}")
            return "error", "鼠标双击失败"
        return "continue", reasoning

    elif action == "clipboard":
        print(f"📋 AI 建议输入文本: {content}")
        input("请手动将文本复制到剪贴板并粘贴到文本框，然后输入 1 继续：")
        return "continue", reasoning

    elif action == "keyboard_input":
        try:
            pyautogui.write(content)
            print(f"⌨️ 键盘输入：{content}")
        except Exception as e:
            logging.error(f"键盘输入失败：{e}")
            print(f"⚠️ 键盘输入失败: {e}")
            return "error", "键盘输入失败"
        return "continue", reasoning

    elif action == "task_complete":
        print("✅ 任务已完成" if content != "error" else "🛑 任务异常终止")
        return "complete", reasoning

    elif action == "error":
        print(f"❌ 错误：{content}")
        return "error", reasoning

    else:
        print(f"❌ 未知操作类型：{action}")
        return "error", "未知操作类型"

def debug_image(image_data, ai_coords, actual_coords):
    """在一个窗口中显示图片和标记点"""
    try:
        # 从 base64 编码的图片数据中提取图片
        image_data = base64.b64decode(image_data.split(',')[1])
        image = Image.open(BytesIO(image_data))

        # 创建 figure 和 axes
        fig, ax = plt.subplots(1)

        # 显示图片
        ax.imshow(image)

        # 标记 AI 指定的坐标
        ax.plot(ai_coords[0], ai_coords[1], 'ro', markersize=8, label='AI 坐标')

        # 标记实际点击的坐标
        ax.plot(actual_coords[0], actual_coords[1], 'bo', markersize=8, label='实际坐标')

        # 添加图例
        ax.legend()

        # 显示窗口
        plt.show()
    except Exception as e:
        logging.error(f"显示图片失败：{e}")
        print(f"⚠️ 显示图片失败: {e}")

class SMMS(object):
    # init
    def __init__(self, username, password):
        self.username = username
        self.password = password
        self.token = None
        self.profile = None
        self.history = None
        self.upload_history = None
        self.url = None
        self.headers = None  # 初始化 headers
        self.root = 'https://sm.ms/api/v2/'  # 定义 root 变量

    # user
    def get_api_token(self):
        data = {
            'username': self.username,
            'password': self.password,
        }
        url = self.root+'token'  # 使用 self.root
        try:
            res = requests.post(url, data=data).json()
            if res['success']:
                self.token = res['data']['token']
                self.headers = {'Authorization': self.token}
                print("API Token 获取成功")
            else:
                print(f"API Token 获取失败: {res['message']}")
                return False
            # print(json.dumps(res, indent=4))
            return True
        except Exception as e:
            print(f"API Token 获取异常: {e}")
            return False

    # user
    def get_user_profile(self):
        url = self.root+'profile'  # 使用 self.root
        try:
            res = requests.get(url, headers=self.headers).json()  # 使用 get 方法
            if res['success']:
                self.profile = res['data']
                print(json.dumps(res, indent=4))
            else:
                print(f"获取用户资料失败: {res['message']}")
        except Exception as e:
            print(f"获取用户资料异常: {e}")

    # image
    def clear_temporary_history(self):
        data = {
            'format': 'json'
        }
        url = self.root+'clear'  # 使用 self.root
        try:
            res = requests.get(url, data=data).json()
            print(json.dumps(res, indent=4))
        except Exception as e:
            print(f"清除临时历史记录异常: {e}")

    # image
    def view_temporary_history(self):
        url = self.root+'history'  # 使用 self.root
        try:
            res = requests.get(url).json()
            self.history = res['data']
            print(json.dumps(res, indent=4))
        except Exception as e:
            print(f"查看临时历史记录异常: {e}")

    # image
    def delete_image(self, hash):
        url = self.root+'delete/'+hash  # 使用 self.root
        try:
            res = requests.get(url, headers=self.headers).json()  # 添加 headers
            print(json.dumps(res, indent=4))
        except Exception as e:
            print(f"删除图片异常: {e}")

    # image
    def view_upload_history(self):
        url = self.root+'upload_history'  # 使用 self.root
        try:
            res = requests.get(url, headers=self.headers).json()
            self.upload_history = res['data']
            print(json.dumps(res, indent=4))
        except Exception as e:
            print(f"查看上传历史记录异常: {e}")

    # image
    def upload_image(self, image_data):
        try:
            # 将 base64 编码的图片数据解码为 bytes
            image_bytes = base64.b64decode(image_data.split(',')[1])
            files = {'smfile': ('image.jpg', image_bytes)}  # 使用 image_bytes
            url = self.root+'upload'  # 使用 self.root
            res = requests.post(url, files=files, headers=self.headers).json()
            if res['success']:
                self.url = res['data']['url']
                self.hash = res['data']['hash']  # 保存 hash 值
                print(f"图片上传成功，URL: {self.url}")
                return self.url, self.hash
            else:
                print(f"图片上传失败: {res['message']}")
                return None, None
        except Exception as e:
            print(f"图片上传异常: {e}")
            return None, None

    def delete_uploaded_image(self, hash_value):
        """删除已上传的图片"""
        if not hash_value:
            print("没有可删除的图片 Hash 值")
            return

        url = f"{self.root}delete/{hash_value}"  # 使用 self.root
        try:
            response = requests.get(url, headers=self.headers)
            response_json = response.json()
            if response_json["success"]:
                print(f"图片 {hash_value} 删除成功")
            else:
                print(f"图片 {hash_value} 删除失败: {response_json['message']}")
        except requests.exceptions.RequestException as e:
            print(f"删除图片 {hash_value} 发生异常: {e}")

# 修改 main 函数
def main():
    # 注册信号处理
    signal.signal(signal.SIGINT, signal_handler)

    task = input("🎯 请描述您的目标任务 (输入exit退出): ")
    if task.lower() in ['exit', 'quit']:
        sys.exit(0)

    # 初始化 SMMS 图床
    smms = SMMS('你的smms用户名', '你的smms密码')
    if not smms.get_api_token():
        print("❌ SMMS API Token 获取失败，请检查用户名和密码")
        sys.exit(1)

    image_url = None
    image_hash = None
    previous_reasoning = "" # 初始化 previous_reasoning
    asked_confirmation = False # 初始化 asked_confirmation

    retry_count = 0
    max_retries = 3

    while retry_count < max_retries:
        # 删除之前上传的图片
        if image_hash:
            smms.delete_uploaded_image(image_hash)
            image_hash = None
            image_url = None

        # 处理截图
        image_meta = take_screenshot()
        if not image_meta:
            print("❌ 截图处理失败，请尝试减少屏幕内容或手动截图")
            sys.exit(1)

        # 上传图片到 SMMS
        image_url, image_hash = smms.upload_image(image_meta['image_data'])
        if not image_url:
            print("❌ 图片上传失败，将不使用图片进行任务")
            messages = [{
                "role": "user",
                "content": [build_prompt(task, previous_reasoning, asked_confirmation)]
            }]
        else:
            # 初始化对话历史：包含图片 URL 和任务提示
            messages = [{
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": image_url,  # 使用 image_url 字段
                    },
                    build_prompt(task, previous_reasoning, asked_confirmation)
                ]
            }]

        response = send_to_ai(messages)
        status, reasoning = process_response(response, messages, image_meta, asked_confirmation)

        if status == "complete":
            break
        elif status == "exit":
            print("🛑 用户主动终止任务")
            break
        elif status == "error":
            retry_count += 1
            if retry_count >= max_retries:
                print("❌ 已达到最大重试次数")
                break
            retry = input(f"🔄 是否重试？(y/n) [{retry_count}/{max_retries}]: ").lower()
            if retry != 'y':
                break
        else:
            previous_reasoning = reasoning # 保存 reasoning，下一次使用
            if response.get("action") == "question":
                asked_confirmation = True # 标记已经提问过
            elif asked_confirmation:
                # 如果已经提问过，则按照步骤执行点击操作
                # 这里需要根据 AI 返回的坐标，执行点击操作
                pass

    # 任务完成后删除图片
    if image_hash:
        smms.delete_uploaded_image(image_hash)

if __name__ == "__main__":
    main()
