import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from flask import Flask, request, render_template_string
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import io
import base64

# =============================
# Load and preprocess dataset
file_path = r"C:\Users\user\Desktop\Task_Timing_Log_1000_Entries.xlsx"
df = pd.read_excel(file_path)

df['Start Time'] = pd.to_datetime(df['Start Time'])
df['End Time'] = pd.to_datetime(df['End Time'])
df['Duration'] = (df['End Time'] - df['Start Time']).dt.total_seconds() / 60.0
df['Task Name'] = df['Task Name'].str.strip().str.lower()
df = df.sort_values(by=['Task Name', 'Start Time'])
df['Repetition'] = df.groupby('Task Name').cumcount() + 1

task_to_id = {task: i for i, task in enumerate(df['Task Name'].unique())}
df['Task ID'] = df['Task Name'].map(task_to_id)

X = df[['Repetition', 'Duration', 'Task ID']].values.astype(np.float32)
y = df['Quality Score'].values.astype(np.float32).reshape(-1, 1)

X_tensor = torch.tensor(X)
y_tensor = torch.tensor(y)

# =============================
# Neural Network Model
class LearningRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3, 12)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(12, 8)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(8, 1)

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        return self.fc3(x)

model = LearningRegressor()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Train the model
for epoch in range(300):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_tensor)
    loss = criterion(outputs, y_tensor)
    loss.backward()
    optimizer.step()

# =============================
# Flask App
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    result = ''
    image_current = ''
    image_improved = ''

    if request.method == 'POST':
        task_input = request.form['task'].strip().lower()
        repetition_input = int(request.form['repetition'])
        duration_input = float(request.form['duration'])

        if task_input not in task_to_id:
            result = "<p style='color: red;'>❌ Unknown task name. Please check your input.</p>"
        else:
            task_id_input = task_to_id[task_input]
            input_tensor = torch.tensor([[repetition_input, duration_input, float(task_id_input)]])
            predicted_score = model(input_tensor).item()

            # Dynamic adjustment logic
            if predicted_score < 4:
                improved_duration = duration_input + 30
                improved_repetition = repetition_input + 2
            elif predicted_score < 7:
                improved_duration = duration_input + 20
                improved_repetition = repetition_input + 1
            else:
                improved_duration = duration_input + 10
                improved_repetition = repetition_input

            input_tensor_improved = torch.tensor([[improved_repetition, improved_duration, float(task_id_input)]])
            improved_score = model(input_tensor_improved).item()

            def label_text(score):
                if score >= 7:
                    return "Good 🚀"
                elif score >= 4:
                    return "Moderate 🙂"
                else:
                    return "Average 😐"

            label = label_text(predicted_score)
            improved_label = label_text(improved_score)

            if predicted_score >= 7:
                suggestion = "Keep up your learning style! Try spaced repetition for even better retention."
            elif predicted_score >= 4:
                suggestion = "Increase focus time slightly or try active recall techniques."
            else:
                suggestion = "Try longer sessions with breaks, and explain the topic aloud."

            # Current Visualization
            plt.figure(figsize=(6, 4))
            plt.gca().add_patch(patches.Rectangle((0, 0), 20, 4, color='red', alpha=0.1))
            plt.gca().add_patch(patches.Rectangle((0, 4), 20, 3, color='yellow', alpha=0.1))
            plt.gca().add_patch(patches.Rectangle((0, 7), 20, 3, color='lightgreen', alpha=0.1))
            plt.scatter(repetition_input, predicted_score, color='orange', s=200, edgecolor='black')
            plt.text(repetition_input, predicted_score + 0.3, f"{predicted_score:.2f}", fontsize=10, ha='center')
            plt.title(f"📊 Current Score: '{task_input.title()}'")
            plt.xlabel("Repetition")
            plt.ylabel("Predicted Score")
            plt.ylim(0, 10.5)
            plt.tight_layout()
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            image_current = base64.b64encode(buf.read()).decode('utf-8')
            plt.close()

            # Improved Visualization
            plt.figure(figsize=(6, 4))
            plt.gca().add_patch(patches.Rectangle((0, 0), 20, 4, color='red', alpha=0.1))
            plt.gca().add_patch(patches.Rectangle((0, 4), 20, 3, color='yellow', alpha=0.1))
            plt.gca().add_patch(patches.Rectangle((0, 7), 20, 3, color='lightgreen', alpha=0.1))
            plt.scatter(improved_repetition, improved_score, color='green', s=200, edgecolor='black')
            plt.text(improved_repetition, improved_score + 0.3, f"{improved_score:.2f}", fontsize=10, ha='center')
            plt.title("🔮 Improved Prediction")
            plt.xlabel("Repetition")
            plt.ylabel("Predicted Score")
            plt.ylim(0, 10.5)
            plt.tight_layout()
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            image_improved = base64.b64encode(buf.read()).decode('utf-8')
            plt.close()

            result = f"""
                <h3>📘 Prediction Summary</h3>
                <p><b>Task:</b> {task_input.title()}</p>
                <p><b>Predicted Score:</b> {predicted_score:.2f} → <b>{label}</b></p>
                <p><b>Suggestion:</b> {suggestion}</p>
                <hr>
                <h4>🔮 What-if Scenario</h4>
                <p>If you study +{int(improved_duration - duration_input)} min and +{int(improved_repetition - repetition_input)} repetition:</p>
                <p><b>New Score:</b> {improved_score:.2f} → <b>{improved_label}</b></p>
            """

    return render_template_string('''
    <html>
    <head>
        <title>🧠 Learning Score Predictor</title>
        <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;600&display=swap" rel="stylesheet">
        <style>
            body {
                font-family: 'Poppins', sans-serif;
                background-color: #121212;
                color: #f0f0f0;
                padding: 50px;
                animation: fadeIn 0.8s ease-in;
                overflow-y: auto;
            }
            input[type="text"], input[type="number"] {
                width: 300px;
                padding: 10px;
                margin-top: 8px;
                margin-bottom: 20px;
                border-radius: 8px;
                border: 1px solid #888;
                font-size: 15px;
                background-color: #1f1f1f;
                color: #fff;
                transition: box-shadow 0.3s ease;
            }
            input[type="text"]:focus, input[type="number"]:focus {
                box-shadow: 0 0 10px #2196f3;
                outline: none;
            }
            input[type="submit"] {
                background: linear-gradient(to right, #2196f3, #21cbf3);
                color: white;
                padding: 10px 25px;
                border: none;
                border-radius: 6px;
                font-size: 15px;
                cursor: pointer;
                transition: transform 0.3s ease;
            }
            input[type="submit"]:hover {
                transform: scale(1.05);
            }
            .result-box {
                background-color: #1f1f1f;
                padding: 20px;
                border-radius: 12px;
                margin-top: 30px;
                box-shadow: 0 0 10px rgba(255,255,255,0.05);
            }
            img {
                border-radius: 10px;
                margin-top: 20px;
                max-width: 100%;
            }
            @keyframes fadeIn {
                from {opacity: 0; transform: translateY(10px);}
                to {opacity: 1; transform: translateY(0);}
            }
        </style>
    </head>
    <body>
        <h2>🎯 Learning Quality Score Predictor</h2>
        <form method="post">
            <label>🧠 Task Name:</label><br>
            <input type="text" name="task" required><br>
            <label>🔁 Repetition Number:</label><br>
            <input type="number" name="repetition" required><br>
            <label>⏱ Duration (minutes):</label><br>
            <input type="number" step="any" name="duration" required><br>
            <input type="submit" value="Predict">
        </form>

        {% if result %}
        <div class="result-box">
            {{result | safe}}
            {% if image_current %}
                <h3>📈 Current Score Visualization</h3>
                <img src="data:image/png;base64,{{image_current}}">
                <h3>📈 What-if Score Visualization</h3>
                <img src="data:image/png;base64,{{image_improved}}">
            {% endif %}
        </div>
        {% endif %}
    </body>
    </html>
    ''', result=result, image_current=image_current, image_improved=image_improved)

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
