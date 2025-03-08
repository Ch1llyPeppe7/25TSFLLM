from flask import Flask, request, jsonify, render_template
import pandas as pd
from io import StringIO

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('template.html')


@app.route('/process_data', methods=['POST'])
def process_data():
    # 处理上传的CSV文件或文本数据
    file = request.files.get('file')
    data_text = request.form.get('data_text')

    if file:
        # 读取CSV文件
        df = pd.read_csv(file)
    elif data_text:
        # 处理文本数据
        data_lines = data_text.splitlines()
        data = [line.split(',') for line in data_lines]
        df = pd.DataFrame(data)
    
    # 假设通道信息在 DataFrame 中有特定的列，进行检查
    # 例如，假设第一列是时间戳，后面的列是通道数据
    channels = df.columns[1:].tolist()  # 假设通道从第二列开始
    
    return jsonify({"channels": channels})

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    channels = data.get("channels", [])

    # 执行预测逻辑
    # 这里你可以根据所选择的通道执行相关的预测操作
    prediction_result = {"prediction": "预测结果示例"}
    
    return jsonify(prediction_result)

if __name__ == '__main__':
    app.run(debug=True)
