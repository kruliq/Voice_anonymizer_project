import os
import subprocess
from flask import Flask, request, render_template, send_from_directory, jsonify
from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32MB limit
ALLOWED_EXTENSIONS = {'wav'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_filtered_files():
    return [f for f in os.listdir(UPLOAD_FOLDER) if f != '.gitignore']

# Create uploads directory with proper permissions
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER, mode=0o755)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'status': 'error', 'message': 'No file part', 'files': get_filtered_files()})
        file = request.files['file']
        if file.filename == '':
            return jsonify({'status': 'error', 'message': 'No selected file', 'files': get_filtered_files()})
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            os.chmod(file_path, 0o644)
            # Run the external script
            output_file = os.path.join(app.config['UPLOAD_FOLDER'], f"{filename.rsplit('.', 1)[0]}_anon.wav")
            process = subprocess.Popen(
                ['python3', '/home/tm_user/Voice_anonymizer_project/base_functionality/web/base_test.py', '-i', file_path, '-o', output_file],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            stdout, stderr = process.communicate()
            if process.returncode == 0:
                return jsonify({'status': 'success', 'message': stdout.decode('utf-8'), 'files': get_filtered_files()})
            else:
                return jsonify({'status': 'error', 'message': stderr.decode('utf-8'), 'files': get_filtered_files()})
        return jsonify({'status': 'error', 'message': 'File type not allowed', 'files': get_filtered_files()})
    
    uploaded_files = get_filtered_files()
    return render_template('index.html', uploaded_files=uploaded_files)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], secure_filename(filename))

if __name__ == '__main__':
    app.run(debug=True)