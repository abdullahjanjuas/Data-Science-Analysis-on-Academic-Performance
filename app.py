from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)
model = joblib.load('content/rf_pipeline.joblib')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        data = {
            'Gender': request.form['gender'],
            'Area': request.form['area'],
            'Living Situation': request.form['living'],
            'Higher Secondary Education': request.form['hs_edu'],
            'Higher Secondary Subjects': request.form['hs_subject'],
            'Higher Secondary Marks Percentage (%)': float(request.form['hs_marks']),
            'NET Marks': float(request.form['net_marks']),
            'Parents Education Level': int(request.form['parents_edu']),
            'Family Income': int(request.form['income']),
            'House Size (Sqft)': int(request.form['house_size']),
            'Family Size': float(request.form['family_size']),
            'Stress Level': int(request.form['stress']),
            'Confidence Level': int(request.form['confidence']),
            'Study Consistency Level': int(request.form['consistency']),
            'Participation in Extra-Curricular Activities': int(request.form['extra']),
            'Available Emotional Support': int(request.form['emotional']),
            'Sleeping Level': int(request.form['sleep']),
            'Attendance Level': int(request.form['attendance']),
            'Access to Resources': int(request.form['resources']),
            'Understanding of Subject': int(request.form['understanding']),
            'Interest in Degree': int(3),
            'Personal Space (Sqft)': float(request.form['personal_space'])
        }

        input_df = pd.DataFrame([data])
        prediction = model.predict(input_df)[0]
        return render_template('form.html', prediction=round(prediction, 2))

    return render_template('form.html', prediction=None)

if __name__ == '__main__':
    app.run(debug=True)
